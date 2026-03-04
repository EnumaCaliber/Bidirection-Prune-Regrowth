"""
RL-based Iterative Regrowth  —  v3
===================================
Key changes vs v2
─────────────────
1. SSIM uses **feature-map** SSIM (BlockwiseFeatureExtractor + compute_block_ssim),
   same as single_layer_analysis.py.  Threshold still selects layers each iteration.

2. Baseline accuracy is **pre-loaded** from the iterative-pruning table (Excel).
   A linear interpolator fills in any sparsity that is not an exact table entry.
   No need to evaluate the un-regrowed model each iteration.

3. Reward = (accuracy_after_regrowth - baseline_at_that_sparsity)  [percentage points]
   → positive = better than the pruning baseline, negative = worse.

4. Budget fraction is the FIRST agent decision per episode (same as v2).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import copy
import os
import time
import wandb
import random
from torch.distributions import Categorical
from collections import deque
from tqdm import tqdm
import math

from utils.model_loader import model_loader
from utils.data_loader import data_loader
from utils.analysis_utils import (
    load_model_name, prune_weights_reparam, count_pruned_params,
    BlockwiseFeatureExtractor, compute_block_ssim,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-loaded Baseline  (from the iterative-pruning Excel table)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Per-model baseline tables  (sparsity → test_accuracy %) ─────────────────
ITERATIVE_BASELINE_TABLES = {
    'densenet': {
        0.30:   93.22,
        0.51:   93.53,
        0.657:  93.96,
        0.7599: 93.90,
        0.8319: 93.97,
        0.8824: 93.96,
        0.9176: 93.60,
        0.9424: 93.74,
        0.9596: 93.52,
        0.9718: 93.46,
        0.9802: 93.00,
        0.9862: 92.30,
        0.9903: 91.83,
        0.9932: 90.48,
        0.9953: 88.09,
    },
    'effnet': {
        0.30:   89.42,
        0.51:   89.42,
        0.657:  89.39,
        0.7599: 89.33,
        0.8319: 89.57,
        0.8824: 89.56,
        0.9176: 89.75,
        0.9424: 89.48,
        0.9596: 89.25,
        0.9718: 88.87,
        0.9802: 88.35,
        0.9862: 87.54,
        0.9903: 86.21,
        0.9932: 83.89,
        0.9953: 80.14,
    },
    'vgg16': {
        0.30:   91.58,
        0.51:   91.93,
        0.657:  91.96,
        0.7599: 92.28,
        0.8319: 92.27,
        0.8824: 92.40,
        0.9176: 92.65,
        0.9424: 92.51,
        0.9596: 92.27,
        0.9718: 92.01,
        0.9802: 91.76,
        0.9862: 91.29,
        0.9903: 90.63,
        0.9932: 89.89,
        0.9953: 89.22,
    },
    'resnet20': {
        0.30:   91.61,
        0.51:   91.48,
        0.657:  91.59,
        0.7599: 91.09,
        0.8319: 90.48,
        0.8824: 89.65,
        0.9176: 88.70,
        0.9424: 87.68,
        0.9596: 86.38,
        0.9718: 84.94,
        0.9802: 82.51,
        0.9862: 79.27,
        0.9903: 76.07,
        0.9932: 69.04,
        0.9953: 60.64,
    },
}


class BaselineInterpolator:
    """
    Linear interpolation over discrete (sparsity, accuracy) baseline points.

    Usage
    ─────
    bi = BaselineInterpolator(ITERATIVE_BASELINE_TABLES['resnet20'])
    acc = bi.get_baseline_acc(0.982)   # → interpolated accuracy %
    """

    def __init__(self, table: dict):
        self.points = {float(k): float(v) for k, v in table.items()}
        pts = sorted(self.points.items())
        print(f"  [Baseline] loaded {len(pts)} points: "
              f"sparsity {pts[0][0]:.4f}→{pts[-1][0]:.4f}")

    def get_baseline_acc(self, sparsity: float) -> float:
        pts = sorted(self.points.items())
        if sparsity <= pts[0][0]:
            return pts[0][1]
        if sparsity >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            s1, a1 = pts[i]
            s2, a2 = pts[i + 1]
            if s1 <= sparsity <= s2:
                t = (sparsity - s1) / (s2 - s1 + 1e-12)
                return a1 + t * (a2 - a1)
        return pts[-1][1]

    def __repr__(self):
        return f"BaselineInterpolator({len(self.points)} pts)"


# ═══════════════════════════════════════════════════════════════════════════════
# Feature-map SSIM Layer Selector  (mirrors single_layer_analysis.py)
# ═══════════════════════════════════════════════════════════════════════════════

class SSIMLayerSelector:
    """
    Per-iteration SSIM-based search-space selection.

    Every iteration:
      1. Scan ALL layers with weight_mask in the current sparse model.
      2. Compute feature-map SSIM vs pretrained for each layer.
      3. Layers with SSIM < threshold (default -1.0) enter the RL search space.
      4. Fallback to the single worst layer if nothing qualifies.

    No fixed candidate pool — the search space adapts to the current model
    state each iteration as regrowth gradually repairs degraded layers.
    """

    @staticmethod
    def update_search_space(sparse_model, pretrained_model,
                            data_loader_ref,
                            threshold: float = -1.0,
                            num_batches: int = 64) -> tuple:
        """
        Returns (selected_layers, ssim_dict).

        selected_layers : layers whose feature-SSIM < threshold
        ssim_dict       : {layer_name: ssim_score} for ALL masked layers
        """
        # Collect every layer that has a pruning mask
        all_masked = [name for name, m in sparse_model.named_modules()
                      if hasattr(m, 'weight_mask') and len(name) > 0]

        block_dict = {'all_layers': all_masked}
        ext_pre  = BlockwiseFeatureExtractor(pretrained_model, block_dict)
        ext_spar = BlockwiseFeatureExtractor(sparse_model,     block_dict)

        with torch.no_grad():
            feats_pre  = ext_pre.extract_block_features(data_loader_ref,  num_batches=num_batches)
            feats_spar = ext_spar.extract_block_features(data_loader_ref, num_batches=num_batches)

        ssim_raw   = compute_block_ssim(feats_pre, feats_spar)
        block_ssim = ssim_raw.get('all_layers', {})

        ssim_dict, selected = {}, []
        for lname in all_masked:
            score = float(block_ssim.get(lname, 0.0))
            ssim_dict[lname] = score
            if score < threshold:
                selected.append(lname)

        print(f"\n  ── SSIM Layer Selection  (threshold < {threshold:+.3f}) ──")
        for n, s in ssim_dict.items():
            flag = "  ← SEARCH" if n in selected else ""
            print(f"    {n}: {s:+.4f}{flag}")

        if not selected:
            worst    = min(ssim_dict, key=ssim_dict.get)
            selected = [worst]
            print(f"  No layer below {threshold:+.3f} → fallback: {worst} ({ssim_dict[worst]:+.4f})")

        print(f"  Search space: {len(selected)}/{len(all_masked)} layers\n")
        return selected, ssim_dict


# ═══════════════════════════════════════════════════════════════════════════════
# Saliency  (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════════

class SaliencyComputer:
    """Saliency(θ) = (∂L/∂θ)² · θ²  —  computed ONCE from pretrained model."""

    def __init__(self, model, criterion, device='cuda'):
        self.model     = model
        self.criterion = criterion
        self.device    = device
        self.accumulated_grads = {}
        self.grad_count        = 0

    def reset(self):
        self.accumulated_grads = {}
        self.grad_count        = 0

    def compute_saliency_scores(self, data_loader, target_layers):
        self.model.eval()
        self.reset()
        print("\nComputing saliency (one-time, pretrained model)…")

        module_dict = dict(self.model.named_modules())
        for lname in target_layers:
            m = module_dict.get(lname)
            if m is not None and hasattr(m, 'weight'):
                self.accumulated_grads[lname] = torch.zeros(
                    m.weight.shape, device=self.device)

        for inputs, labels in tqdm(data_loader, desc="  grad accum"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            loss  = self.criterion(self.model(inputs), labels)
            grads = torch.autograd.grad(loss, self.model.parameters(),
                                        create_graph=False, allow_unused=True)
            for param, grad in zip(self.model.parameters(), grads):
                if grad is None:
                    continue
                for name, p in self.model.named_parameters():
                    if p is param:
                        for lname in target_layers:
                            if name == f"{lname}.weight":
                                self.accumulated_grads[lname] += (
                                    grad.pow(2).detach() * param.data.pow(2).detach())
                                break
                        break
            self.grad_count += 1

        saliency_dict = {}
        for lname in target_layers:
            if lname in self.accumulated_grads:
                sal = self.accumulated_grads[lname] / max(self.grad_count, 1)
                saliency_dict[lname] = sal.cpu()
                print(f"  {lname}: mean={sal.mean():.3e}  max={sal.max():.3e}")
        print("Saliency done.\n")
        return saliency_dict


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM Controller
# ═══════════════════════════════════════════════════════════════════════════════

class RegrowthAgent(nn.Module):
    """
    LSTM controller with two separate decoders:
      - budget_decoder : Step 0     → budget_space_size logits
      - alloc_decoder  : Steps 1..N → alloc_space_size  logits

    LSTM input dim = max(budget_space, alloc_space); prev_logits are
    zero-padded to this size so the hidden state is always consistent.
    """

    def __init__(self, budget_space_size, alloc_space_size,
                 hidden_size, context_dim, device='cuda'):
        super().__init__()
        self.DEVICE       = device
        self.nhid         = hidden_size
        self.budget_space = budget_space_size
        self.alloc_space  = alloc_space_size
        self.input_dim    = max(budget_space_size, alloc_space_size)

        self.lstm           = nn.LSTMCell(self.input_dim + context_dim, hidden_size)
        self.budget_decoder = nn.Linear(hidden_size, budget_space_size)
        self.alloc_decoder  = nn.Linear(hidden_size, alloc_space_size)
        self.hidden         = self.init_hidden()

    def forward(self, prev_logits, context_vec, step='alloc'):
        """
        step='budget' -> budget_decoder (Step 0)
        step='alloc'  -> alloc_decoder  (Steps 1..N)
        """
        if prev_logits.dim() == 1: prev_logits = prev_logits.unsqueeze(0)
        if context_vec.dim() == 1: context_vec = context_vec.unsqueeze(0)
        pad = self.input_dim - prev_logits.shape[-1]
        if pad > 0:
            prev_logits = F.pad(prev_logits, (0, pad))
        h, c = self.lstm(torch.cat([prev_logits, context_vec], dim=-1), self.hidden)
        self.hidden = (h, c)
        return self.budget_decoder(h) if step == 'budget' else self.alloc_decoder(h)

    def init_hidden(self):
        return (torch.zeros(1, self.nhid, device=self.DEVICE),
                torch.zeros(1, self.nhid, device=self.DEVICE))


# ═══════════════════════════════════════════════════════════════════════════════
# Saliency-based Regrowth
# ═══════════════════════════════════════════════════════════════════════════════

class SaliencyBasedRegrowth:
    @staticmethod
    @torch.no_grad()
    def apply_regrowth(model, layer_name, saliency_tensor, num_weights,
                       init_strategy='zero', device='cuda'):
        module = dict(model.named_modules()).get(layer_name)
        if module is None or not hasattr(module, 'weight_mask'):
            return 0, []

        mask  = module.weight_mask
        sal   = saliency_tensor.to(device)
        pruned = (mask == 0)
        if not pruned.any():
            return 0, []

        sal_m = sal.clone()
        sal_m[~pruned] = -float('inf')
        flat = sal_m.flatten()
        k    = min(num_weights, (flat > -float('inf')).sum().item())
        if k == 0:
            return 0, []

        _, top_k  = torch.topk(flat, k=k)
        regrown   = []
        for fi in top_k:
            idx = np.unravel_index(fi.cpu().item(), sal.shape)
            regrown.append(idx)
            mask[idx] = 1.0
            wp = getattr(module, 'weight_orig', module.weight)
            if init_strategy == 'zero':
                wp.data[idx] = 0.0
            elif init_strategy == 'kaiming':
                fi_n, _ = nn.init._calculate_fan_in_and_fan_out(wp)
                b = np.sqrt(6.0 / fi_n)
                wp.data[idx] = torch.empty(1).uniform_(-b, b).item()
            elif init_strategy == 'xavier':
                fi_n, fo_n = nn.init._calculate_fan_in_and_fan_out(wp)
                b = np.sqrt(6.0 / (fi_n + fo_n))
                wp.data[idx] = torch.empty(1).uniform_(-b, b).item()
        return len(regrown), regrown


# ═══════════════════════════════════════════════════════════════════════════════
# RL Policy Gradient
# ═══════════════════════════════════════════════════════════════════════════════

class RegrowthPolicyGradient:
    """
    Changes vs v1
    ─────────────
    • Budget fraction = Step 0 action  (linspace min_budget_frac … max_budget_frac)
    • Reward = acc_after_regrowth − baseline_interp(sparsity)   [pp, /100]
    • baseline_interp is the pre-loaded iterative-pruning table
    • target_layers updated externally each iteration via SSIM
    """

    def __init__(self, config, model_sparse, saliency_dict,
                 target_layers, train_loader, test_loader, device,
                 baseline_interp: BaselineInterpolator,
                 total_weights: int,
                 wandb_run=None):

        self.NUM_EPOCHS         = config['num_epochs']
        self.ALPHA              = config['learning_rate']
        self.HIDDEN_SIZE        = config['hidden_size']
        self.BETA               = config['entropy_coef']
        self.REWARD_TEMPERATURE = config.get('reward_temperature', 0.01)
        self.DEVICE             = device
        self.BUDGET_SPACE      = config['budget_space_size']
        self.ALLOC_SPACE       = config['alloc_space_size']
        self.NUM_STEPS          = len(target_layers)
        self.CONTEXT_DIM        = config.get('context_dim', 3)
        self.BASELINE_DECAY     = config.get('baseline_decay', 0.9)

        self.model_sparse     = model_sparse.to(device)
        self.target_layers    = target_layers
        self.train_loader     = train_loader
        self.test_loader      = test_loader

        self.layer_capacities = config['layer_capacities']
        self.total_capacity   = max(sum(self.layer_capacities), 1)
        self.init_strategy    = config.get('init_strategy', 'zero')

        self.early_stop_patience   = config.get('early_stop_patience', 40)
        self.min_epochs            = config.get('min_epochs', 50)
        self.reward_std_threshold  = config.get('reward_std_threshold', 0.002)
        self.reward_window_size    = config.get('reward_window_size', 20)

        self._best_model_state = None
        self._best_reward_seen = float('-inf')

        self.run              = wandb_run
        self.model_name       = config.get('model_name')
        self.method           = config.get('method', 'iterative')
        self.checkpoint_dir   = config.get('checkpoint_dir', './rl_checkpoints')
        self.model_sparsity   = config.get('model_sparsity', '0.98')

        self.use_entropy_schedule = config.get('use_entropy_schedule', True)
        self.start_beta           = config.get('start_beta', 0.4)
        self.end_beta             = config.get('end_beta', 0.004)
        self.decay_fraction       = config.get('decay_fraction', 0.4)

        # ── Pre-loaded baseline ────────────────────────────────────────────────
        self.baseline_interp = baseline_interp
        self.total_weights   = total_weights

        # ── Budget fractions ───────────────────────────────────────────────────
        min_f = config.get('min_budget_frac', 0.001)
        max_f = config.get('max_budget_frac', 0.010)
        self.budget_fracs = np.linspace(min_f, max_f, self.BUDGET_SPACE).tolist()
        print(f"  Budget fracs: [{min_f:.4f}…{max_f:.4f}]  "
              f"({self.BUDGET_SPACE} options)")

        # ── Saliency (pretrained, fixed) ───────────────────────────────────────
        self.saliency_dict = saliency_dict

        # ── Agent ─────────────────────────────────────────────────────────────
        self.agent = RegrowthAgent(
            budget_space_size=self.BUDGET_SPACE,
            alloc_space_size=self.ALLOC_SPACE,
            hidden_size=self.HIDDEN_SIZE,
            context_dim=self.CONTEXT_DIM,
            device=self.DEVICE,
        ).to(self.DEVICE)

        self.adam            = optim.Adam(self.agent.parameters(), lr=self.ALPHA)
        self.reward_baseline = None   # running baseline for advantage
        self.layer_priority  = [(n, i) for i, n in enumerate(target_layers)]

        print(f"  Search-space layers ({len(target_layers)}):")
        for n, i in self.layer_priority:
            print(f"    {i+1}. {n}  cap={self.layer_capacities[i]}")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def get_entropy_coef(self, epoch):
        if not self.use_entropy_schedule:
            return self.BETA
        de = self.NUM_EPOCHS * self.decay_fraction
        if epoch < de:
            return self.start_beta - (self.start_beta - self.end_beta) * (epoch / de)
        return self.end_beta

    def _create_model_copy(self, src):
        m = model_loader(self.model_name, self.DEVICE)
        prune_weights_reparam(m)
        m.load_state_dict(src.state_dict())
        return m

    def get_best_model(self):
        if self._best_model_state is None:
            return None
        m = model_loader(self.model_name, self.DEVICE)
        prune_weights_reparam(m)
        m.load_state_dict(self._best_model_state)
        return m

    def calculate_sparsity(self, model):
        total, pruned = 0, 0
        for _, m in model.named_modules():
            if hasattr(m, 'weight_mask'):
                total  += m.weight_mask.numel()
                pruned += (m.weight_mask == 0).sum().item()
        return (100.0 * pruned / total if total > 0 else 0.0), total, pruned

    def evaluate_model(self, model, full_eval=False):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                if not full_eval and i >= 20: break
                x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                _, pred = model(x).max(1)
                total   += y.size(0)
                correct += pred.eq(y).sum().item()
        return 100.0 * correct / total

    def mini_finetune(self, model, epochs=40, lr=3e-4):
        model.train()
        opt  = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        crit = nn.CrossEntropyLoss()
        best_acc, best_state = 0.0, None
        for _ in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                opt.zero_grad()
                crit(model(x), y).backward()
                opt.step()
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in self.test_loader:
                    x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                    _, pred = model(x).max(1)
                    total   += y.size(0)
                    correct += pred.eq(y).sum().item()
            acc = 100.0 * correct / total
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(model.state_dict())
            model.train()
        if best_state:
            model.load_state_dict(best_state)
        model.eval()

    def calculate_loss(self, budget_logits, alloc_logits, wlp, beta):
        loss = -torch.mean(wlp)
        # Entropy over budget choices
        p_b  = F.softmax(budget_logits, dim=1)
        ent_b = -torch.mean(torch.sum(p_b * F.log_softmax(budget_logits, dim=1), dim=1))
        # Entropy over alloc choices
        p_a  = F.softmax(alloc_logits, dim=1)
        ent_a = -torch.mean(torch.sum(p_a * F.log_softmax(alloc_logits, dim=1), dim=1))
        ent  = (ent_b + ent_a) / 2.0
        return loss - beta * ent, ent

    # ── Main loop ─────────────────────────────────────────────────────────────
    def solve_environment(self, resume_from=None):
        t0 = time.time()
        best_reward, best_alloc, best_regrow = float('-inf'), None, None
        best_frac, best_reward_ep, start_ep  = None, 0, 0
        reward_window = deque(maxlen=self.reward_window_size)
        stop_reason   = ""

        if resume_from and os.path.exists(resume_from):
            ckpt = torch.load(resume_from)
            self.agent.load_state_dict(ckpt['agent_state_dict'])
            self.adam.load_state_dict(ckpt['optimizer_state_dict'])
            start_ep      = ckpt['epoch'] + 1
            best_reward   = ckpt['best_reward']
            best_alloc    = ckpt['best_allocation']
            best_regrow   = ckpt['best_regrow_indices']
            best_frac     = ckpt.get('best_budget_frac')
            if 'reward_baseline' in ckpt:
                self.reward_baseline = ckpt['reward_baseline']
            print(f"Resumed ep {start_ep}, best={best_reward:+.4f}")

        for epoch in range(start_ep, self.NUM_EPOCHS):
            (ep_wlp, ep_budget_logits, ep_alloc_logits, reward,
             alloc, sparsity, regrow_idx, frac) = self.play_episode(epoch)

            reward_window.append(reward)
            if reward > best_reward:
                best_reward, best_reward_ep = reward, epoch
                best_alloc, best_regrow, best_frac = alloc, copy.deepcopy(regrow_idx), frac
                self._save_best_alloc(epoch, best_reward, best_alloc, best_regrow, best_frac)

            beta      = self.get_entropy_coef(epoch)
            loss, ent = self.calculate_loss(ep_budget_logits, ep_alloc_logits, ep_wlp, beta)
            self.adam.zero_grad()
            loss.backward()
            self.adam.step()

            no_imp  = epoch - best_reward_ep
            rwd_std = float(np.std(list(reward_window))) if len(reward_window) > 1 else float('inf')

            pfx = f"iter_{self.model_sparsity}"
            if self.run:
                self.run.log({
                    f"{pfx}/epoch":        epoch + 1,
                    f"{pfx}/reward_pp":    reward * 100,
                    f"{pfx}/best_pp":      best_reward * 100,
                    f"{pfx}/loss":         loss.item(),
                    f"{pfx}/entropy":      ent.item(),
                    f"{pfx}/beta":         beta,
                    f"{pfx}/budget_frac":  frac,
                    f"{pfx}/sparsity":     sparsity,
                    f"{pfx}/no_improve":   no_imp,
                    f"{pfx}/rwd_std":      rwd_std,
                })

            print(f"[{self.model_sparsity}] Ep {epoch+1:3d}/{self.NUM_EPOCHS} | "
                  f"Rwd={reward*100:+.2f}pp | Best={best_reward*100:+.2f}pp | "
                  f"Frac={frac:.4f} | Loss={loss.item():.4f} | "
                  f"Ent={ent.item():.4f} | NoImp={no_imp} | Std={rwd_std:.5f}")

            if no_imp >= self.early_stop_patience:
                stop_reason = f"NoImp {self.early_stop_patience}"
                print(f"\n>>> Early stop: {stop_reason}")
                break
            if (epoch >= self.min_epochs
                    and len(reward_window) >= self.reward_window_size
                    and rwd_std < self.reward_std_threshold):
                stop_reason = f"Std {rwd_std:.5f} < {self.reward_std_threshold}"
                print(f"\n>>> Early stop: {stop_reason}")
                break

        print(f"\nBest: {best_reward*100:+.2f}pp  frac={best_frac}"
              + (f"  [{stop_reason}]" if stop_reason else ""))
        return best_alloc, best_reward, best_regrow, best_frac

    # ── Episode ───────────────────────────────────────────────────────────────
    def play_episode(self, epoch):
        self.agent.hidden = self.agent.init_hidden()
        prev_logits       = torch.zeros(1, self.BUDGET_SPACE, device=self.DEVICE)
        all_log_probs        = []
        budget_masked_logits = []   # size BUDGET_SPACE
        alloc_masked_logits  = []   # size ALLOC_SPACE

        # ── Step 0: Budget ────────────────────────────────────────────────────
        b_ctx     = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float,
                                  device=self.DEVICE).unsqueeze(0)
        b_logits  = self.agent(prev_logits, b_ctx, step='budget').squeeze(0)
        b_dist    = Categorical(probs=F.softmax(b_logits, dim=0))
        b_action  = b_dist.sample()
        sel_frac  = self.budget_fracs[b_action.item()]
        target_rg = max(1, min(int(self.total_weights * sel_frac),
                               sum(self.layer_capacities)))

        all_log_probs.append(b_dist.log_prob(b_action))
        budget_masked_logits.append(b_logits)
        prev_logits = b_logits.unsqueeze(0)  # budget_space sized; agent will pad

        print(f"  [Budget] frac={sel_frac:.4f} → {target_rg} weights")

        # ── Steps 1…N: Per-layer allocation ───────────────────────────────────
        ratio_opts     = (torch.arange(self.ALLOC_SPACE, device=self.DEVICE,
                                       dtype=torch.float) / (self.ALLOC_SPACE - 1))
        remaining      = target_rg
        sel_counts, pnames = [], []

        for p_idx, (lname, orig_idx) in enumerate(self.layer_priority):
            cap = int(self.layer_capacities[orig_idx])
            ctx = torch.tensor([
                (p_idx + 1) / (self.NUM_STEPS + 1),
                cap / self.total_capacity,
                remaining / target_rg if target_rg > 0 else 0.0,
            ], dtype=torch.float, device=self.DEVICE).unsqueeze(0)

            logits    = self.agent(prev_logits, ctx, step='alloc').squeeze(0)
            eff_max   = min(cap, remaining)
            c_opts    = torch.round(ratio_opts * eff_max).to(torch.long)
            feasible  = c_opts <= remaining
            if not feasible.any(): feasible[0] = True

            masked    = torch.where(feasible, logits, torch.full_like(logits, -1e9))
            dist      = Categorical(probs=F.softmax(masked, dim=0))
            action    = dist.sample()
            chosen    = int(c_opts[action].item())
            chosen    = min(chosen, cap, remaining)
            remaining = max(remaining - chosen, 0)

            sel_counts.append(chosen)
            pnames.append(lname)
            all_log_probs.append(dist.log_prob(action))
            alloc_masked_logits.append(masked)
            prev_logits = logits.unsqueeze(0)

        ep_log_probs = torch.stack(all_log_probs)
        # Keep budget and alloc logits separate (different sizes)
        ep_budget_logits = torch.stack(budget_masked_logits)  # [1, BUDGET_SPACE]
        ep_alloc_logits  = torch.stack(alloc_masked_logits)   # [N, ALLOC_SPACE]
        allocation   = {n: int(c) for n, c in zip(pnames, sel_counts) if c > 0}
        print(f"  [Alloc] {sum(sel_counts)}/{target_rg} | {allocation}")

        # ── Apply regrowth ────────────────────────────────────────────────────
        model_copy    = self._create_model_copy(self.model_sparse)
        regrow_indices = {}
        for lname, num_w in allocation.items():
            sal = self.saliency_dict.get(lname)
            if sal is not None:
                _, idxs = SaliencyBasedRegrowth.apply_regrowth(
                    model=model_copy, layer_name=lname, saliency_tensor=sal,
                    num_weights=num_w, init_strategy=self.init_strategy,
                    device=self.DEVICE)
                regrow_indices[lname] = idxs

        self.mini_finetune(model_copy, epochs=40)
        accuracy         = self.evaluate_model(model_copy, full_eval=True)
        sparsity, _, _   = self.calculate_sparsity(model_copy)

        # ── Reward: improvement over pre-loaded baseline ──────────────────────
        sp_frac      = sparsity / 100.0
        baseline_acc = self.baseline_interp.get_baseline_acc(sp_frac)
        improvement  = accuracy - baseline_acc      # percentage points
        reward       = improvement / 100.0          # ≈ [-1, +1]

        print(f"  [Reward] acc={accuracy:.2f}%  "
              f"baseline@sp={sp_frac:.4f}→{baseline_acc:.2f}%  "
              f"Δ={improvement:+.2f}pp  reward={reward:+.4f}")

        if reward > self._best_reward_seen:
            self._best_reward_seen = reward
            self._best_model_state = copy.deepcopy(model_copy.state_dict())
            self._save_best_model(epoch, reward, accuracy, model_copy,
                                  allocation, sel_frac)

        # ── RL advantage ──────────────────────────────────────────────────────
        if self.reward_baseline is None:
            self.reward_baseline = reward
        adv = float(np.clip(
            (reward - self.reward_baseline) / max(self.REWARD_TEMPERATURE, 1e-6),
            -100.0, 100.0))
        self.reward_baseline = (self.BASELINE_DECAY * self.reward_baseline
                                + (1 - self.BASELINE_DECAY) * reward)

        adv_t    = torch.tensor(adv, device=self.DEVICE, dtype=torch.float)
        ep_wlp   = torch.sum(ep_log_probs * adv_t).unsqueeze(0)

        return ep_wlp, ep_budget_logits, ep_alloc_logits, reward, allocation, sparsity, regrow_indices, sel_frac

    # ── Checkpointing ─────────────────────────────────────────────────────────
    def _save_best_model(self, epoch, reward, accuracy, model, allocation, frac):
        d = os.path.join(self.checkpoint_dir,
                         f'{self.model_name}/{self.method}/{self.model_sparsity}')
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f'best_ep{epoch+1}_rwd{reward*100:+.2f}pp.pth')
        torch.save({
            'epoch': epoch, 'reward': reward, 'accuracy': accuracy,
            'model_state_dict': model.state_dict(),
            'allocation': allocation, 'budget_frac': frac,
            'timestamp': time.time(),
        }, p)
        print(f"  ✓ Best model: {p}")
        if self.run:
            self.run.log({"best_reward_pp": reward * 100,
                          "best_epoch": epoch + 1,
                          "best_budget_frac": frac})

    def _save_best_alloc(self, epoch, reward, alloc, regrow, frac):
        d = os.path.join(self.checkpoint_dir,
                         f'{self.model_name}/{self.method}/{self.model_sparsity}')
        os.makedirs(d, exist_ok=True)
        torch.save({
            'epoch': epoch, 'best_reward': reward,
            'best_allocation': alloc, 'best_regrow_indices': regrow,
            'best_budget_frac': frac,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.adam.state_dict(),
            'reward_baseline': self.reward_baseline,
            'timestamp': time.time(),
        }, os.path.join(d, 'best_allocation.pth'))
        print(f"  ✓ Best alloc: {reward*100:+.2f}pp  frac={frac:.4f} @ ep {epoch+1}")


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def get_layer_capacities(model, target_layers):
    caps = []
    for name in target_layers:
        m = dict(model.named_modules()).get(name)
        caps.append(int((m.weight_mask == 0).sum().item())
                    if (m is not None and hasattr(m, 'weight_mask')) else 0)
    return caps


def quick_eval(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            _, pred = model(x).max(1)
            total   += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total


def get_sparsity(model):
    total, pruned = 0, 0
    for _, m in model.named_modules():
        if hasattr(m, 'weight_mask'):
            total  += m.weight_mask.numel()
            pruned += (m.weight_mask == 0).sum().item()
    return 100.0 * pruned / total if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_name',   type=str,   default='resnet20')
    parser.add_argument('--data_dir', type=str,   default='./data')
    parser.add_argument('--method',   type=str,   default='iterative')

    # Sparsity
    parser.add_argument('--start_sparsity',  type=float, default=0.9903)
    parser.add_argument('--target_sparsity', type=float, default=0.98)
    parser.add_argument('--num_iters',       type=int,   default=20)

    # RL
    parser.add_argument('--num_epochs',        type=int,   default=300)
    parser.add_argument('--learning_rate',     type=float, default=3e-4)
    parser.add_argument('--hidden_size',       type=int,   default=64)
    parser.add_argument('--entropy_coef',      type=float, default=0.5)
    parser.add_argument('--reward_temperature',type=float, default=0.005)
    parser.add_argument('--start_beta',        type=float, default=0.40)
    parser.add_argument('--end_beta',          type=float, default=0.04)
    parser.add_argument('--decay_fraction',    type=float, default=0.4)
    parser.add_argument('--budget_space_size', type=int,   default=5,
                        help='Number of discrete budget options (Step 0)')
    parser.add_argument('--alloc_space_size',  type=int,   default=11,
                        help='Number of discrete allocation options per layer (Steps 1..N)')

    # Budget search
    parser.add_argument('--min_budget_frac', type=float, default=0.001)
    parser.add_argument('--max_budget_frac', type=float, default=0.005)

    # SSIM layer selection
    parser.add_argument('--ssim_threshold', type=float, default=-0.1,
                        help='Layers with feature-SSIM < this value enter the RL '
                             'search space each iteration. Feature-SSIM can go below -1.')
    parser.add_argument('--ssim_num_batches', type=int, default=64,
                        help='Batches used for feature-SSIM computation per iteration')

    # Regrowth
    parser.add_argument('--init_strategy', type=str, default='zero',
                        choices=['zero', 'kaiming', 'xavier', 'magnitude'])

    # Early stopping
    parser.add_argument('--early_stop_patience',  type=int,   default=40)
    parser.add_argument('--min_epochs',            type=int,   default=50)
    parser.add_argument('--reward_std_threshold',  type=float, default=0.002)
    parser.add_argument('--reward_window_size',    type=int,   default=20)

    # Misc
    parser.add_argument('--save_dir',    type=str, default='./rl_checkpoints')
    parser.add_argument('--resume',      type=str, default=None)
    parser.add_argument('--resume_iter', type=int, default=0)
    parser.add_argument('--seed',        type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    train_loader, _, test_loader = data_loader(data_dir=args.data_dir)

    # ── Initial checkpoint path (per model) ──────────────────────────────────

    initial_ckpt = (f'./{args.m_name}/ckpt_after_prune_0.3_epoch_finetune_40/'
                    f'pruned_finetuned_mask_0.9953.pth')

    # ── Pre-loaded baseline (per model) ──────────────────────────────────────
    assert args.m_name in ITERATIVE_BASELINE_TABLES, \
        f"No baseline table for '{args.m_name}'. Add it to ITERATIVE_BASELINE_TABLES."
    baseline_interp = BaselineInterpolator(ITERATIVE_BASELINE_TABLES[args.m_name])

    # ── Total prunable weights ────────────────────────────────────────────────
    _ref = model_loader(args.m_name, device)
    prune_weights_reparam(_ref)
    _ref.load_state_dict(torch.load(initial_ckpt))
    total_weights, _, _ = count_pruned_params(_ref)
    del _ref
    print(f"Total prunable weights: {total_weights:,}\n")

    # ── wandb ─────────────────────────────────────────────────────────────────
    run = wandb.init(
        project="ICCAD_saliency_iterative_v3",
        name=(f"{args.m_name}_{args.start_sparsity:.4f}"
              f"_ssim{args.ssim_threshold}"),
        config=vars(args) | {"total_weights": total_weights},
    )

    # ── Pretrained (dense) model ──────────────────────────────────────────────
    print("Loading pretrained (dense) model…")
    model_pretrained = model_loader(args.m_name, device)
    load_model_name(model_pretrained, f'./{args.m_name}/checkpoint', args.m_name)
    model_pretrained.eval()

    # ── Saliency: computed ONCE on pretrained for ALL masked layers ───────────
    # We don't know the search space yet (it's determined per-iteration by SSIM),
    # so compute saliency for every layer that has a weight_mask.
    print("\nLoading initial sparse model to enumerate all masked layers…")
    _init_sparse = model_loader(args.m_name, device)
    prune_weights_reparam(_init_sparse)
    _init_sparse.load_state_dict(torch.load(initial_ckpt))
    all_masked_layers = [name for name, m in _init_sparse.named_modules()
                         if hasattr(m, 'weight_mask') and len(name) > 0]
    del _init_sparse
    print(f"  Found {len(all_masked_layers)} masked layers → computing saliency…")

    print("\n" + "=" * 70)
    print("ONE-TIME saliency on ALL masked layers (pretrained model)")
    print("=" * 70)
    saliency_dict = SaliencyComputer(
        model=model_pretrained,
        criterion=nn.CrossEntropyLoss(),
        device=device,
    ).compute_saliency_scores(
        data_loader=train_loader,
        target_layers=all_masked_layers,
    )
    print("=" * 70 + "\n")

    # ── Load starting sparse model ────────────────────────────────────────────
    current_model = model_loader(args.m_name, device)
    prune_weights_reparam(current_model)

    if args.resume_iter > 0:
        ckp = os.path.join(args.save_dir,
                           f'{args.m_name}/{args.method}'
                           f'/iter_{args.resume_iter - 1}/best_grown_model.pth')
        assert os.path.exists(ckp), f"Resume ckpt not found: {ckp}"
        current_model.load_state_dict(torch.load(ckp))
        print(f"Resumed from iter {args.resume_iter}: {ckp}")
    else:
        current_model.load_state_dict(torch.load(initial_ckpt))

    acc0 = quick_eval(current_model, test_loader, device)
    sp0  = get_sparsity(current_model)
    print(f"Starting → Acc: {acc0:.2f}%  Sparsity: {sp0:.2f}%")
    print(f"Baseline at sp={sp0/100:.4f}: {baseline_interp.get_baseline_acc(sp0/100):.2f}%\n")

    run.log({"iter_summary/iteration": 0,
             "iter_summary/accuracy":  acc0,
             "iter_summary/sparsity":  sp0})

    # ═════════════════════════════════════════════════════════════════════════
    # Iterative loop
    # ═════════════════════════════════════════════════════════════════════════
    for iter_idx in range(args.resume_iter, args.num_iters):
        cur_sp  = get_sparsity(current_model)
        cur_acc = quick_eval(current_model, test_loader, device)
        bline   = baseline_interp.get_baseline_acc(cur_sp / 100.0)

        print(f"\n{'#' * 70}")
        print(f"  ITER {iter_idx+1}/{args.num_iters}  |  "
              f"sp={cur_sp:.2f}%  acc={cur_acc:.2f}%  baseline={bline:.2f}%  "
              f"gap={cur_acc-bline:+.2f}pp")
        print(f"{'#' * 70}\n")

        # (1) Feature-SSIM: select search space (all masked layers with SSIM < threshold)
        target_layers, ssim_scores = SSIMLayerSelector.update_search_space(
            sparse_model=current_model,
            pretrained_model=model_pretrained,
            data_loader_ref=test_loader,
            threshold=args.ssim_threshold,
            num_batches=args.ssim_num_batches,
        )
        if run:
            for lname, sc in ssim_scores.items():
                run.log({f"ssim/{lname}": sc, "ssim_iter": iter_idx + 1})

        # (2) Layer capacities
        layer_capacities = get_layer_capacities(current_model, target_layers)
        if sum(layer_capacities) == 0:
            print("  All pruned weights restored. Stopping.")
            break

        sp_label = f"iter{iter_idx}_sp{cur_sp/100:.4f}"

        config = {
            'num_epochs':           args.num_epochs,
            'learning_rate':        args.learning_rate,
            'hidden_size':          args.hidden_size,
            'entropy_coef':         args.entropy_coef,
            'budget_space_size':    args.budget_space_size,
            'alloc_space_size':     args.alloc_space_size,
            'layer_capacities':     layer_capacities,
            'model_name':           args.m_name,
            'reward_temperature':   args.reward_temperature,
            'checkpoint_dir':       args.save_dir,
            'start_beta':           args.start_beta,
            'end_beta':             args.end_beta,
            'decay_fraction':       args.decay_fraction,
            'init_strategy':        args.init_strategy,
            'early_stop_patience':  args.early_stop_patience,
            'min_epochs':           args.min_epochs,
            'reward_std_threshold': args.reward_std_threshold,
            'reward_window_size':   args.reward_window_size,
            'model_sparsity':       sp_label,
            'method':               args.method,
            'min_budget_frac':      args.min_budget_frac,
            'max_budget_frac':      args.max_budget_frac,
        }

        pg = RegrowthPolicyGradient(
            config=config,
            model_sparse=current_model,
            saliency_dict=saliency_dict,
            target_layers=target_layers,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            baseline_interp=baseline_interp,
            total_weights=total_weights,
            wandb_run=run,
        )

        resume_path = args.resume if (iter_idx == args.resume_iter and args.resume) else None
        best_alloc, best_reward, _, best_frac = pg.solve_environment(
            resume_from=resume_path)

        best_model = pg.get_best_model() or current_model
        iter_acc   = quick_eval(best_model, test_loader, device)
        iter_sp    = get_sparsity(best_model)
        iter_base  = baseline_interp.get_baseline_acc(iter_sp / 100.0)

        print(f"\n  [Iter {iter_idx+1}] acc={iter_acc:.2f}%  "
              f"sp={iter_sp:.2f}%  Δbaseline={iter_acc-iter_base:+.2f}pp  "
              f"frac={best_frac}")

        run.log({
            "iter_summary/iteration":         iter_idx + 1,
            "iter_summary/best_reward_pp":    best_reward * 100,
            "iter_summary/accuracy":          iter_acc,
            "iter_summary/sparsity":          iter_sp,
            "iter_summary/baseline_acc":      iter_base,
            "iter_summary/delta_baseline_pp": iter_acc - iter_base,
            "iter_summary/best_budget_frac":  best_frac,
            "iter_summary/n_search_layers":   len(target_layers),
        })

        save_dir  = os.path.join(args.save_dir, f'{args.m_name}/{args.method}/iter_{iter_idx}')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'best_grown_model.pth')
        torch.save(best_model.state_dict(), save_path)
        print(f"  ✓ Saved → {save_path}")

        current_model = best_model

        if iter_sp <= args.target_sparsity * 100 + 0.1:
            print(f"  Target sparsity {args.target_sparsity*100:.1f}% reached.")
            break

    # ── Final ─────────────────────────────────────────────────────────────────
    final_acc  = quick_eval(current_model, test_loader, device)
    final_sp   = get_sparsity(current_model)
    final_base = baseline_interp.get_baseline_acc(final_sp / 100.0)

    print(f"\n{'=' * 70}")
    print(f"DONE  |  Acc: {final_acc:.2f}%  Sp: {final_sp:.2f}%  "
          f"Baseline: {final_base:.2f}%  Δ={final_acc-final_base:+.2f}pp")
    print(f"{'=' * 70}")

    run.log({"final/accuracy":          final_acc,
             "final/sparsity":          final_sp,
             "final/delta_baseline_pp": final_acc - final_base})
    run.finish()


if __name__ == '__main__':
    main()