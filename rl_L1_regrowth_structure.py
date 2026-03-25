"""
Structured RL-based Iterative Regrowth  —  v3 (Structured)
============================================================
Based on v3, adapted for structural pruning (torch-pruning).

Key differences vs original v3
────────────────────────────────
1. No weight_mask / prune_weights_reparam.
   Pruning is physical channel removal via torch-pruning.

2. SaliencyComputer (grad²×weight²) replaced by
   TaylorChannelScorer:
     Taylor score = |mean_grad × dense_weight| per output channel
     → selects which pruned channels to restore

3. SaliencyBasedRegrowth (mask flip) replaced by
   apply_config (rebuild from dense model).

4. Sparsity = channel sparsity  (not weight mask ratio).

5. Capacities = # pruned channels per layer  (not # masked weights).

6. SSIM layer selection unchanged (BlockwiseFeatureExtractor).

7. Baseline table + interpolator unchanged.

8. Budget Step-0 action + LSTM controller unchanged.
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
import torch_pruning as tp

from utils.model_loader import model_loader
from utils.data_loader import data_loader
from utils.analysis_utils import (
    load_model_name,
    BlockwiseFeatureExtractor,
    compute_block_ssim,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline Interpolator  (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════════════════

ITERATIVE_BASELINE_TABLES = {
    'vgg16': {
        0.042: 91.74,
        0.081: 91.71,
        0.12: 91.64,
        0.161: 91.64,
        0.2: 91.68,
        0.239: 91.64,
        0.278: 91.63,
        0.319: 91.32,
        0.358: 91.45,
        0.397: 91.41,
        0.438: 91.1,
        0.477: 91.1,
        0.516: 90.63,
        0.555: 90.47,
        0.597: 90.07,
        0.636: 89.59,
        0.675: 89.18,
        0.714: 88.22,
        0.754: 87.23,
        0.793: 86.3,
        0.833: 84.15,
        0.874: 82.23,
        0.913: 76.24,
        0.952: 65.53,
        0.99: 31.9,
    },

}


class BaselineInterpolator:
    def __init__(self, table: dict):
        self.points = {float(k): float(v) for k, v in table.items()}
        pts = sorted(self.points.items())
        print(f"  [Baseline] {len(pts)} pts: "
              f"sp {pts[0][0]:.4f}→{pts[-1][0]:.4f}")

    def get_baseline_acc(self, sparsity: float) -> float:
        pts = sorted(self.points.items())
        if sparsity <= pts[0][0]:  return pts[0][1]
        if sparsity >= pts[-1][0]: return pts[-1][1]
        for i in range(len(pts) - 1):
            s1, a1 = pts[i]; s2, a2 = pts[i + 1]
            if s1 <= sparsity <= s2:
                t = (sparsity - s1) / (s2 - s1 + 1e-12)
                return a1 + t * (a2 - a1)
        return pts[-1][1]


# ═══════════════════════════════════════════════════════════════════════════════
# Structured Pruning Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_original_channels(dense_model):
    return {name: m.out_channels
            for name, m in dense_model.named_modules()
            if isinstance(m, nn.Conv2d)}


def get_current_config(model):
    return {name: m.out_channels
            for name, m in model.named_modules()
            if isinstance(m, nn.Conv2d)}


def get_pruned_channel_indices(dense_model, pruned_model, target_layers):
    """
    对比 dense 和 pruned 权重，找出每层被剪掉的通道索引
    返回 {layer_name: [ch_idx, ...]}
    """
    pruned_indices = {}
    d_mods = dict(dense_model.named_modules())
    p_mods = dict(pruned_model.named_modules())

    for name in target_layers:
        d_m = d_mods.get(name)
        p_m = p_mods.get(name)
        if d_m is None or p_m is None or not isinstance(d_m, nn.Conv2d):
            pruned_indices[name] = []
            continue

        d_w = d_m.weight.data   # [C_dense, ...]
        p_w = p_m.weight.data   # [C_pruned, ...]

        kept = []
        # in_channels 可能也被剪了，取 min 对齐后再比较
        min_in = min(p_w.shape[1], d_w.shape[1])
        for pw in p_w:
            diffs = [(pw[:min_in] - d_w[j][:min_in]).abs().sum().item()
                     for j in range(d_w.shape[0])]
            kept.append(int(torch.tensor(diffs).argmin()))

        pruned_indices[name] = [i for i in range(d_m.out_channels)
                                if i not in kept]
    return pruned_indices


def apply_config(dense_model, config, original_channels, example_inputs):
    """从 dense 模型按 config 重建结构化子网"""
    model = copy.deepcopy(dense_model)
    pruning_ratio_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in config:
            sp = 1 - config[name] / original_channels[name]
            if sp > 0:
                pruning_ratio_dict[module] = sp

    if pruning_ratio_dict:
        pruner = tp.pruner.MagnitudePruner(
            model, example_inputs,
            importance=tp.importance.MagnitudeImportance(p=1),
            iterative_steps=1,
            pruning_ratio=0,
            pruning_ratio_dict=pruning_ratio_dict,
        )
        pruner.step()
    return model


def compute_channel_sparsity(model, original_channels):
    """channel sparsity = 被删通道 / 原始总通道"""
    total, remaining = 0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and name in original_channels:
            total     += original_channels[name]
            remaining += m.out_channels
    return (1 - remaining / total) if total > 0 else 0.0


def get_target_layers(dense_model, pruned_model):
    """
    自动发现被结构化剪枝过的层
    对比 dense 和 pruned，通道数减少的 Conv2d 层就是目标层
    等价于原 v3 的 weight_mask 发现逻辑
    """
    target_layers = []
    d_mods = dict(dense_model.named_modules())
    for name, p_m in pruned_model.named_modules():
        if not isinstance(p_m, nn.Conv2d):
            continue
        d_m = d_mods.get(name)
        if d_m is None:
            continue
        if p_m.out_channels < d_m.out_channels:  # 通道数变少 → 被剪过
            target_layers.append(name)
    return target_layers


def get_layer_capacities_structured(dense_model, pruned_model, target_layers):
    """每层还能加回多少通道"""
    caps = []
    d_mods = dict(dense_model.named_modules())
    p_mods = dict(pruned_model.named_modules())
    for name in target_layers:
        d_m = d_mods.get(name)
        p_m = p_mods.get(name)
        if d_m and p_m and isinstance(d_m, nn.Conv2d):
            caps.append(max(d_m.out_channels - p_m.out_channels, 0))
        else:
            caps.append(0)
    return caps


# ═══════════════════════════════════════════════════════════════════════════════
# SSIM Layer Selector  (unchanged from v3, wraps BlockwiseFeatureExtractor)
# ═══════════════════════════════════════════════════════════════════════════════

class SSIMLayerSelector:
    """
    Per-iteration SSIM-based search-space selection.
    Scans all Conv2d layers (instead of weight_mask layers in original v3).
    """

    @staticmethod
    def update_search_space(sparse_model, pretrained_model,
                            data_loader_ref, target_layers,
                            threshold: float = -1.0,
                            num_batches: int = 64) -> tuple:
        """
        Returns (selected_layers, ssim_dict).
        selected_layers: layers with feature-SSIM < threshold
        ssim_dict: {layer_name: ssim_score}
        """
        block_dict = {'all_layers': target_layers}
        ext_pre  = BlockwiseFeatureExtractor(pretrained_model, block_dict)
        ext_spar = BlockwiseFeatureExtractor(sparse_model,     block_dict)

        with torch.no_grad():
            feats_pre  = ext_pre.extract_block_features(
                data_loader_ref, num_batches=num_batches)
            feats_spar = ext_spar.extract_block_features(
                data_loader_ref, num_batches=num_batches)

        ssim_raw   = compute_block_ssim(feats_pre, feats_spar)
        block_ssim = ssim_raw.get('all_layers', {})

        ssim_dict, selected = {}, []
        for lname in target_layers:
            score = float(block_ssim.get(lname, 0.0))
            ssim_dict[lname] = score
            if score < threshold:
                selected.append(lname)

        print(f"\n  ── SSIM Layer Selection (threshold < {threshold:+.3f}) ──")
        for n, s in ssim_dict.items():
            flag = "  ← SEARCH" if n in selected else ""
            print(f"    {n}: {s:+.4f}{flag}")

        if not selected:
            worst    = min(ssim_dict, key=ssim_dict.get)
            selected = [worst]
            print(f"  Fallback → {worst} ({ssim_dict[worst]:+.4f})")

        print(f"  Search space: {len(selected)}/{len(target_layers)} layers\n")
        return selected, ssim_dict


# ═══════════════════════════════════════════════════════════════════════════════
# Taylor Channel Scorer  (replaces SaliencyComputer)
# ═══════════════════════════════════════════════════════════════════════════════

class TaylorChannelScorer:
    """
    Taylor score per output channel = |mean_grad × dense_weight|
    Computed from the current sparse model's gradients +
    the dense model's weights.

    Called once per outer iteration (not per episode).
    """

    def __init__(self, dense_model, device='cuda'):
        self.dense_model = dense_model
        self.device      = device

    def compute(self, sparse_model, target_layers,
                data_loader, n_batches=10):
        """
        Returns {layer_name: {ch_idx: taylor_score}}
        for every channel in pruned_ch_indices.
        """
        # Accumulate gradients on sparse model
        sparse_model.train()
        sparse_model.zero_grad()
        crit = nn.CrossEntropyLoss()

        for i, (inputs, targets) in enumerate(data_loader):
            if i >= n_batches:
                break
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            loss = crit(sparse_model(inputs), targets)
            loss.backward()

        sparse_model.eval()

        d_mods = dict(self.dense_model.named_modules())
        p_mods = dict(sparse_model.named_modules())
        channel_scores = {}

        for layer_name in target_layers:
            d_m = d_mods.get(layer_name)
            p_m = p_mods.get(layer_name)

            if d_m is None or p_m is None or not isinstance(d_m, nn.Conv2d):
                channel_scores[layer_name] = {}
                continue
            if p_m.weight.grad is None:
                channel_scores[layer_name] = {}
                continue

            p_grad   = p_m.weight.grad.detach()   # [C_pruned, C_in, k, k]
            d_weight = d_m.weight.detach()         # [C_dense,  C_in, k, k]

            # Pruned channels don't exist in sparse model →
            # use mean gradient of existing channels as proxy
            mean_grad = p_grad.mean(dim=0)         # [C_in, k, k]

            # Score every channel in dense model
            # (caller will filter to pruned-only channels)
            scores = {}
            for ch_idx in range(d_m.out_channels):
                d_w    = d_weight[ch_idx]          # [C_in, k, k]
                min_in = min(mean_grad.shape[0], d_w.shape[0])
                taylor = (mean_grad[:min_in] * d_w[:min_in]).abs().sum().item()
                scores[ch_idx] = taylor

            channel_scores[layer_name] = scores
            print(f"  {layer_name}: {d_m.out_channels} ch, "
                  f"max_taylor={max(scores.values()):.3e}")

        return channel_scores


def select_channels_by_taylor(channel_scores, pruned_ch_indices,
                               layer_name, n_restore):
    """从被剪掉的通道里按 Taylor score 选 top-n"""
    all_scores  = channel_scores.get(layer_name, {})
    pruned_chs  = pruned_ch_indices.get(layer_name, [])
    if not all_scores or not pruned_chs or n_restore == 0:
        return []

    # 只在被剪掉的通道里排序
    candidate_scores = {ch: all_scores[ch] for ch in pruned_chs if ch in all_scores}
    if not candidate_scores:
        return []

    n = min(n_restore, len(candidate_scores))
    sorted_ch = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return [ch for ch, _ in sorted_ch[:n]]


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM Controller  (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════════════════

class RegrowthAgent(nn.Module):
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
        if prev_logits.dim() == 1: prev_logits = prev_logits.unsqueeze(0)
        if context_vec.dim() == 1: context_vec = context_vec.unsqueeze(0)
        pad = self.input_dim - prev_logits.shape[-1]
        if pad > 0:
            prev_logits = F.pad(prev_logits, (0, pad))
        h, c = self.lstm(
            torch.cat([prev_logits, context_vec], dim=-1), self.hidden)
        self.hidden = (h, c)
        return self.budget_decoder(h) if step == 'budget' else self.alloc_decoder(h)

    def init_hidden(self):
        return (torch.zeros(1, self.nhid, device=self.DEVICE),
                torch.zeros(1, self.nhid, device=self.DEVICE))


# ═══════════════════════════════════════════════════════════════════════════════
# RL Policy Gradient  (structured version)
# ═══════════════════════════════════════════════════════════════════════════════

class StructuredRegrowthPG:
    """
    Differences vs original v3 RegrowthPolicyGradient
    ──────────────────────────────────────────────────
    • No weight_mask / saliency_dict.
    • channel_scores (Taylor) + pruned_ch_indices decide which channels to add.
    • apply_config rebuilds model from dense; no mask flipping.
    • Sparsity = channel sparsity.
    • Budget = # channels to restore (not # weights).
    • total_channels replaces total_weights for budget scaling.
    """

    def __init__(self, config, model_sparse, dense_model,
                 channel_scores, pruned_ch_indices,
                 original_channels, example_inputs,
                 target_layers, train_loader, test_loader,
                 device, baseline_interp, total_channels,
                 wandb_run=None):

        self.NUM_EPOCHS         = config['num_epochs']
        self.ALPHA              = config['learning_rate']
        self.HIDDEN_SIZE        = config['hidden_size']
        self.BETA               = config['entropy_coef']
        self.REWARD_TEMPERATURE = config.get('reward_temperature', 0.01)
        self.DEVICE             = device
        self.BUDGET_SPACE       = config['budget_space_size']
        self.ALLOC_SPACE        = config['alloc_space_size']
        self.NUM_STEPS          = len(target_layers)
        self.CONTEXT_DIM        = config.get('context_dim', 3)
        self.BASELINE_DECAY     = config.get('baseline_decay', 0.9)

        self.model_sparse       = model_sparse
        self.dense_model        = dense_model
        self.target_layers      = target_layers
        self.train_loader       = train_loader
        self.test_loader        = test_loader

        self.layer_capacities   = config['layer_capacities']
        self.total_capacity     = max(sum(self.layer_capacities), 1)

        # Taylor channel scores + pruned indices
        self.channel_scores     = channel_scores
        self.pruned_ch_indices  = pruned_ch_indices
        self.original_channels  = original_channels
        self.example_inputs     = example_inputs

        # Current config (per-layer channel counts)
        self.current_config     = get_current_config(model_sparse)

        self.early_stop_patience   = config.get('early_stop_patience', 40)
        self.min_epochs            = config.get('min_epochs', 50)
        self.reward_std_threshold  = config.get('reward_std_threshold', 0.002)
        self.reward_window_size    = config.get('reward_window_size', 20)
        self.finetune_epochs       = config.get('finetune_epochs', 5)

        self._best_model        = None
        self._best_reward_seen  = float('-inf')

        self.run              = wandb_run
        self.model_name       = config.get('model_name')
        self.method           = config.get('method', 'structured_iterative')
        self.checkpoint_dir   = config.get('checkpoint_dir', './structured_rl_ckpts')
        self.model_sparsity   = config.get('model_sparsity', '0.5')

        self.use_entropy_schedule = config.get('use_entropy_schedule', True)
        self.start_beta     = config.get('start_beta', 0.4)
        self.end_beta       = config.get('end_beta', 0.004)
        self.decay_fraction = config.get('decay_fraction', 0.4)

        # Baseline
        self.baseline_interp = baseline_interp
        self.total_channels  = total_channels  # for budget scaling

        # Budget options: # channels to restore
        min_ch = max(1, int(total_channels * config.get('min_budget_frac', 0.001)))
        max_ch = max(2, int(total_channels * config.get('max_budget_frac', 0.010)))
        self.budget_options = list(range(min_ch,
                                         max_ch + 1,
                                         max(1, (max_ch - min_ch) //
                                             (self.BUDGET_SPACE - 1))))[:self.BUDGET_SPACE]
        while len(self.budget_options) < self.BUDGET_SPACE:
            self.budget_options.append(self.budget_options[-1])
        print(f"  Budget options (channels): {self.budget_options}")

        self.agent = RegrowthAgent(
            budget_space_size=self.BUDGET_SPACE,
            alloc_space_size=self.ALLOC_SPACE,
            hidden_size=self.HIDDEN_SIZE,
            context_dim=self.CONTEXT_DIM,
            device=self.DEVICE,
        ).to(self.DEVICE)

        self.adam            = optim.Adam(self.agent.parameters(), lr=self.ALPHA)
        self.reward_baseline = None
        self.layer_priority  = [(n, i) for i, n in enumerate(target_layers)]

        print(f"  Layer capacities:")
        for n, i in self.layer_priority:
            print(f"    {n}: cap={self.layer_capacities[i]}")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def get_entropy_coef(self, epoch):
        if not self.use_entropy_schedule:
            return self.BETA
        de = self.NUM_EPOCHS * self.decay_fraction
        if epoch < de:
            return self.start_beta - (self.start_beta - self.end_beta) * (epoch / de)
        return self.end_beta

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

    def mini_finetune(self, model, epochs=5, lr=3e-4):
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
            acc = self.evaluate_model(model, full_eval=True)
            if acc > best_acc:
                best_acc   = acc
                best_state = copy.deepcopy(model.state_dict())
            model.train()
        if best_state:
            model.load_state_dict(best_state)
        model.eval()

    def calculate_loss(self, budget_logits, alloc_logits, wlp, beta):
        loss  = -torch.mean(wlp)
        p_b   = F.softmax(budget_logits, dim=1)
        ent_b = -torch.mean(torch.sum(p_b * F.log_softmax(budget_logits, dim=1), dim=1))
        p_a   = F.softmax(alloc_logits,  dim=1)
        ent_a = -torch.mean(torch.sum(p_a * F.log_softmax(alloc_logits,  dim=1), dim=1))
        ent   = (ent_b + ent_a) / 2.0
        return loss - beta * ent, ent

    # ── Main RL loop ─────────────────────────────────────────────────────────
    def solve_environment(self):
        best_reward, best_config = float('-inf'), None
        best_reward_ep, best_frac = 0, None
        reward_window = deque(maxlen=self.reward_window_size)
        stop_reason   = ""

        for epoch in range(self.NUM_EPOCHS):
            (ep_wlp, ep_budget_logits, ep_alloc_logits,
             reward, new_config, sparsity, frac) = self.play_episode(epoch)

            reward_window.append(reward)
            if reward > best_reward:
                best_reward, best_reward_ep = reward, epoch
                best_config, best_frac = new_config, frac
                self._save_best(epoch, reward, new_config, frac)

            beta      = self.get_entropy_coef(epoch)
            loss, ent = self.calculate_loss(
                ep_budget_logits, ep_alloc_logits, ep_wlp, beta)
            self.adam.zero_grad()
            loss.backward()
            self.adam.step()

            no_imp  = epoch - best_reward_ep
            rwd_std = float(np.std(list(reward_window))) \
                      if len(reward_window) > 1 else float('inf')

            pfx = f"iter_{self.model_sparsity}"
            if self.run:
                self.run.log({
                    f"{pfx}/epoch":       epoch + 1,
                    f"{pfx}/reward_pp":   reward * 100,
                    f"{pfx}/best_pp":     best_reward * 100,
                    f"{pfx}/loss":        loss.item(),
                    f"{pfx}/entropy":     ent.item(),
                    f"{pfx}/beta":        beta,
                    f"{pfx}/budget_ch":   frac,
                    f"{pfx}/sparsity":    sparsity,
                    f"{pfx}/no_improve":  no_imp,
                    f"{pfx}/rwd_std":     rwd_std,
                })

            print(f"[{self.model_sparsity}] Ep {epoch+1:3d}/{self.NUM_EPOCHS} | "
                  f"Rwd={reward*100:+.2f}pp | Best={best_reward*100:+.2f}pp | "
                  f"BudgetCh={frac} | Loss={loss.item():.4f} | "
                  f"Ent={ent.item():.4f} | NoImp={no_imp} | Std={rwd_std:.5f}")

            if no_imp >= self.early_stop_patience:
                stop_reason = f"NoImp {self.early_stop_patience}"
                break
            if (epoch >= self.min_epochs
                    and len(reward_window) >= self.reward_window_size
                    and rwd_std < self.reward_std_threshold):
                stop_reason = f"Std {rwd_std:.5f} < threshold"
                break

        print(f"\nBest: {best_reward*100:+.2f}pp  budget_ch={best_frac}"
              + (f"  [{stop_reason}]" if stop_reason else ""))
        return best_config, best_reward, best_frac

    # ── Episode ───────────────────────────────────────────────────────────────
    def play_episode(self, epoch):
        t0 = time.time()
        self.agent.hidden = self.agent.init_hidden()
        prev_logits       = torch.zeros(1, self.BUDGET_SPACE, device=self.DEVICE)
        all_log_probs        = []
        budget_masked_logits = []
        alloc_masked_logits  = []

        # ── Step 0: Budget（要加回多少通道）──────────────────────────────────
        b_ctx    = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float,
                                 device=self.DEVICE).unsqueeze(0)
        b_logits = self.agent(prev_logits, b_ctx, step='budget').squeeze(0)
        b_dist   = Categorical(probs=F.softmax(b_logits, dim=0))
        b_action = b_dist.sample()
        target_ch = self.budget_options[b_action.item()]
        target_ch = min(target_ch, sum(self.layer_capacities))

        all_log_probs.append(b_dist.log_prob(b_action))
        budget_masked_logits.append(b_logits)
        prev_logits = b_logits.unsqueeze(0)

        print(f"  [Budget] {target_ch} channels to restore")

        # ── Steps 1…N: Per-layer allocation ───────────────────────────────────
        ratio_opts = (torch.arange(self.ALLOC_SPACE, device=self.DEVICE,
                                   dtype=torch.float) / (self.ALLOC_SPACE - 1))
        remaining  = target_ch
        sel_counts, pnames = [], []

        for p_idx, (lname, orig_idx) in enumerate(self.layer_priority):
            cap = int(self.layer_capacities[orig_idx])
            ctx = torch.tensor([
                (p_idx + 1) / (self.NUM_STEPS + 1),
                cap / self.total_capacity if self.total_capacity > 0 else 0.0,
                remaining / target_ch     if target_ch > 0 else 0.0,
            ], dtype=torch.float, device=self.DEVICE).unsqueeze(0)

            logits   = self.agent(prev_logits, ctx, step='alloc').squeeze(0)
            eff_max  = min(cap, remaining)
            c_opts   = torch.round(ratio_opts * eff_max).to(torch.long)
            feasible = c_opts <= remaining
            if not feasible.any(): feasible[0] = True

            masked   = torch.where(feasible, logits, torch.full_like(logits, -1e9))
            dist     = Categorical(probs=F.softmax(masked, dim=0))
            action   = dist.sample()
            chosen   = int(c_opts[action].item())
            chosen   = min(chosen, cap, remaining)
            remaining = max(remaining - chosen, 0)

            sel_counts.append(chosen)
            pnames.append(lname)
            all_log_probs.append(dist.log_prob(action))
            alloc_masked_logits.append(masked)
            prev_logits = logits.unsqueeze(0)

        ep_log_probs         = torch.stack(all_log_probs)
        ep_budget_logits     = torch.stack(budget_masked_logits)
        ep_alloc_logits      = torch.stack(alloc_masked_logits)
        allocation           = {n: int(c)
                                for n, c in zip(pnames, sel_counts) if c > 0}
        print(f"  [Alloc] {sum(sel_counts)}/{target_ch} | {allocation}")

        # ── Taylor 选通道 → 更新 config ───────────────────────────────────────
        new_config = copy.copy(self.current_config)
        for lname, n_restore in allocation.items():
            chs = select_channels_by_taylor(
                self.channel_scores, self.pruned_ch_indices, lname, n_restore)
            if chs:
                new_config[lname] = self.current_config[lname] + len(chs)
                print(f"    {lname}: +{len(chs)} ch "
                      f"(taylor top3: {chs[:3]})")

        # ── 从 dense 重建子网 ─────────────────────────────────────────────────
        new_model = apply_config(
            self.dense_model, new_config,
            self.original_channels, self.example_inputs
        ).to(self.DEVICE)

        # ── finetune ──────────────────────────────────────────────────────────
        self.mini_finetune(new_model, epochs=self.finetune_epochs)

        accuracy = self.evaluate_model(new_model, full_eval=True)
        sparsity = compute_channel_sparsity(new_model, self.original_channels)

        # ── Reward = acc - baseline  (pp / 100) ───────────────────────────────
        baseline_acc = self.baseline_interp.get_baseline_acc(sparsity)
        improvement  = accuracy - baseline_acc
        reward       = improvement / 100.0

        print(f"  [Reward] acc={accuracy:.2f}%  "
              f"baseline@sp={sparsity:.4f}→{baseline_acc:.2f}%  "
              f"Δ={improvement:+.2f}pp  reward={reward:+.4f}")

        if reward > self._best_reward_seen:
            self._best_reward_seen = reward
            self._best_model       = copy.deepcopy(new_model)

        # ── Advantage ─────────────────────────────────────────────────────────
        if self.reward_baseline is None:
            self.reward_baseline = reward
        adv = float(np.clip(
            (reward - self.reward_baseline) / max(self.REWARD_TEMPERATURE, 1e-6),
            -100.0, 100.0))
        self.reward_baseline = (self.BASELINE_DECAY * self.reward_baseline
                                + (1 - self.BASELINE_DECAY) * reward)

        adv_t  = torch.tensor(adv, device=self.DEVICE, dtype=torch.float)
        ep_wlp = torch.sum(ep_log_probs * adv_t).unsqueeze(0)

        print(f"  [{time.time()-t0:.1f}s]")
        return ep_wlp, ep_budget_logits, ep_alloc_logits, \
               reward, new_config, sparsity, target_ch

    def _save_best(self, epoch, reward, config, budget_ch):
        d = os.path.join(self.checkpoint_dir,
                         f'{self.model_name}/{self.method}/{self.model_sparsity}')
        os.makedirs(d, exist_ok=True)
        if self._best_model is not None:
            # 移除所有 hook 再保存，避免 pickle 报错
            model_to_save = copy.deepcopy(self._best_model)
            for m in model_to_save.modules():
                m._forward_hooks.clear()
                m._backward_hooks.clear()
                m._forward_pre_hooks.clear()
            torch.save(model_to_save,
                       os.path.join(d, f'best_model_rwd{reward*100:+.2f}pp.pth'))
        torch.save({
            'epoch': epoch, 'reward': reward,
            'config': config, 'budget_ch': budget_ch,
        }, os.path.join(d, 'best_allocation.pth'))
        print(f"  ✓ Best {reward*100:+.2f}pp  budget_ch={budget_ch}  saved → {d}")
        if self.run:
            self.run.log({"best_reward_pp": reward * 100,
                          "best_epoch": epoch + 1})


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_name',   type=str,   default='vgg16')
    parser.add_argument('--data_dir', type=str,   default='./data')
    parser.add_argument('--method',   type=str,   default='structured_iterative')

    # Sparsity
    parser.add_argument('--start_sp',    type=float, default=0.7,
                        help='起始 channel sparsity（对应已剪好的模型）')
    parser.add_argument('--target_sp',   type=float, default=0.3,
                        help='目标 channel sparsity')
    parser.add_argument('--num_iters',   type=int,   default=20)

    # RL
    parser.add_argument('--num_epochs',          type=int,   default=300)
    parser.add_argument('--learning_rate',        type=float, default=3e-4)
    parser.add_argument('--hidden_size',          type=int,   default=64)
    parser.add_argument('--entropy_coef',         type=float, default=0.5)
    parser.add_argument('--reward_temperature',   type=float, default=0.005)
    parser.add_argument('--start_beta',           type=float, default=0.40)
    parser.add_argument('--end_beta',             type=float, default=0.04)
    parser.add_argument('--decay_fraction',       type=float, default=0.4)
    parser.add_argument('--budget_space_size',    type=int,   default=5)
    parser.add_argument('--alloc_space_size',     type=int,   default=11)
    parser.add_argument('--min_budget_frac',      type=float, default=0.001)
    parser.add_argument('--max_budget_frac',      type=float, default=0.010)

    # SSIM
    parser.add_argument('--ssim_threshold',   type=float, default=0.0,
                        help='Layers with SSIM < threshold enter search space')
    parser.add_argument('--ssim_num_batches', type=int,   default=64)

    # Taylor
    parser.add_argument('--taylor_batches', type=int, default=10)

    # Finetune
    parser.add_argument('--finetune_epochs', type=int, default=5)

    # Early stopping
    parser.add_argument('--early_stop_patience',  type=int,   default=40)
    parser.add_argument('--min_epochs',            type=int,   default=50)
    parser.add_argument('--reward_std_threshold',  type=float, default=0.002)
    parser.add_argument('--reward_window_size',    type=int,   default=20)

    # Misc
    parser.add_argument('--save_dir',    type=str, default='./structured_rl_ckpts')
    parser.add_argument('--resume_iter', type=int, default=0)
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    train_loader, _, test_loader = data_loader(data_dir=args.data_dir)

    # 真实数据作为 example_inputs
    example_inputs, _ = next(iter(train_loader))
    example_inputs = example_inputs[:1].to(device)

    # ── Checkpoint 路径（任意模型通用）──────────────────────────────────────
    initial_ckpt = (f'./{args.m_name}/ckpt_structured_iterative/'
                    f'step22_sp0.874.pth')

    # ── Baseline interpolator ─────────────────────────────────────────────────
    assert args.m_name in ITERATIVE_BASELINE_TABLES, \
        f"No baseline table for '{args.m_name}'"
    baseline_interp = BaselineInterpolator(ITERATIVE_BASELINE_TABLES[args.m_name])

    # ── Dense model（原始预训练）──────────────────────────────────────────────
    dense_model = model_loader(args.m_name, device)
    cktp = f'pretrain_{args.m_name}_ckpt.pth'
    checkpoint = torch.load(f'./{args.m_name}/checkpoint/{cktp}', weights_only=False)
    dense_model.load_state_dict(checkpoint['net'])
    dense_model.eval()

    original_channels = get_original_channels(dense_model)

    # ── 起始稀疏模型（结构化保存整个 model）──────────────────────────────────
    current_model = torch.load(initial_ckpt, map_location=device, weights_only=False)
    current_model.eval()

    # ── 自动发现目标层（对比 dense 和 pruned，通道数减少的层）────────────────
    # 等价于原 v3 的 weight_mask 自动发现逻辑
    target_layers = get_target_layers(dense_model, current_model)
    print(f"\n自动发现 {len(target_layers)} 个被剪层：")
    for name in target_layers:
        d_ch = original_channels[name]
        p_ch = dict(current_model.named_modules())[name].out_channels
        print(f"  {name}: {d_ch} → {p_ch} ch  (pruned {d_ch - p_ch})")

    total_channels = sum(original_channels[n] for n in target_layers
                         if n in original_channels)
    print(f"Total channels in target layers: {total_channels}")

    acc0 = quick_eval(current_model, test_loader, device)
    sp0  = compute_channel_sparsity(current_model, original_channels)
    print(f"\n起点 → Acc: {acc0:.2f}%  ChannelSparsity: {sp0:.4f}")
    print(f"Baseline@sp={sp0:.4f}: {baseline_interp.get_baseline_acc(sp0):.2f}%\n")

    run = wandb.init(
        project="structured_ssim_taylor_rl_v3",
        name=f"{args.m_name}_sp{args.start_sp:.2f}→{args.target_sp:.2f}",
        config=vars(args) | {"total_channels": total_channels},
    )
    run.log({"iter_summary/iteration": 0,
             "iter_summary/accuracy":  acc0,
             "iter_summary/sparsity":  sp0})

    # ═════════════════════════════════════════════════════════════════════════
    # Iterative regrowth loop
    # ═════════════════════════════════════════════════════════════════════════
    for iter_idx in range(args.resume_iter, args.num_iters):
        cur_sp  = compute_channel_sparsity(current_model, original_channels)
        cur_acc = quick_eval(current_model, test_loader, device)
        bline   = baseline_interp.get_baseline_acc(cur_sp)

        print(f"\n{'#'*70}")
        print(f"  ITER {iter_idx+1}/{args.num_iters}  |  "
              f"sp={cur_sp:.4f}  acc={cur_acc:.2f}%  "
              f"baseline={bline:.2f}%  gap={cur_acc-bline:+.2f}pp")
        print(f"{'#'*70}\n")

        if cur_sp <= args.target_sp:
            print("目标稀疏度已达到，停止。")
            break

        # (1) SSIM 选层
        selected_layers, ssim_scores = SSIMLayerSelector.update_search_space(
            sparse_model=current_model,
            pretrained_model=dense_model,
            data_loader_ref=test_loader,
            target_layers=target_layers,
            threshold=args.ssim_threshold,
            num_batches=args.ssim_num_batches,
        )
        if run:
            for lname, sc in ssim_scores.items():
                run.log({f"ssim/{lname}": sc, "ssim_iter": iter_idx + 1})

        # (2) Taylor 计算通道重要性（每轮更新）
        print("\nComputing Taylor channel scores…")
        scorer = TaylorChannelScorer(dense_model=dense_model, device=device)
        channel_scores = scorer.compute(
            sparse_model=current_model,
            target_layers=selected_layers,
            data_loader=train_loader,
            n_batches=args.taylor_batches,
        )

        # (3) 被剪通道索引（每轮更新）
        pruned_ch_indices = get_pruned_channel_indices(
            dense_model, current_model, selected_layers)

        # (4) 每层容量
        layer_capacities = get_layer_capacities_structured(
            dense_model, current_model, selected_layers)

        if sum(layer_capacities) == 0:
            print("所有通道已恢复，停止。")
            break

        sp_label = f"iter{iter_idx}_sp{cur_sp:.4f}"
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
            'early_stop_patience':  args.early_stop_patience,
            'min_epochs':           args.min_epochs,
            'reward_std_threshold': args.reward_std_threshold,
            'reward_window_size':   args.reward_window_size,
            'model_sparsity':       sp_label,
            'method':               args.method,
            'min_budget_frac':      args.min_budget_frac,
            'max_budget_frac':      args.max_budget_frac,
            'finetune_epochs':      args.finetune_epochs,
        }

        pg = StructuredRegrowthPG(
            config=config,
            model_sparse=current_model,
            dense_model=dense_model,
            channel_scores=channel_scores,
            pruned_ch_indices=pruned_ch_indices,
            original_channels=original_channels,
            example_inputs=example_inputs,
            target_layers=selected_layers,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            baseline_interp=baseline_interp,
            total_channels=total_channels,
            wandb_run=run,
        )

        best_config, best_reward, best_frac = pg.solve_environment()

        best_model = pg._best_model or current_model
        iter_acc   = quick_eval(best_model, test_loader, device)
        iter_sp    = compute_channel_sparsity(best_model, original_channels)
        iter_base  = baseline_interp.get_baseline_acc(iter_sp)

        print(f"\n  [Iter {iter_idx+1}] acc={iter_acc:.2f}%  "
              f"sp={iter_sp:.4f}  Δbaseline={iter_acc-iter_base:+.2f}pp  "
              f"budget_ch={best_frac}")

        run.log({
            "iter_summary/iteration":         iter_idx + 1,
            "iter_summary/best_reward_pp":    best_reward * 100,
            "iter_summary/accuracy":          iter_acc,
            "iter_summary/sparsity":          iter_sp,
            "iter_summary/baseline_acc":      iter_base,
            "iter_summary/delta_baseline_pp": iter_acc - iter_base,
            "iter_summary/n_search_layers":   len(selected_layers),
        })

        # 保存（结构化必须存整个 model，先清 hook）
        save_dir = os.path.join(args.save_dir,
                                f'{args.m_name}/{args.method}/iter_{iter_idx}')
        os.makedirs(save_dir, exist_ok=True)
        model_to_save = copy.deepcopy(best_model)
        for m in model_to_save.modules():
            m._forward_hooks.clear()
            m._backward_hooks.clear()
            m._forward_pre_hooks.clear()
        torch.save(model_to_save, os.path.join(save_dir, 'best_model.pth'))
        print(f"  ✓ Saved → {save_dir}/best_model.pth")

        current_model = best_model

    # ── Final ─────────────────────────────────────────────────────────────────
    final_acc  = quick_eval(current_model, test_loader, device)
    final_sp   = compute_channel_sparsity(current_model, original_channels)
    final_base = baseline_interp.get_baseline_acc(final_sp)

    print(f"\n{'='*70}")
    print(f"DONE | Acc={final_acc:.2f}%  Sp={final_sp:.4f}  "
          f"Baseline={final_base:.2f}%  Δ={final_acc-final_base:+.2f}pp")
    print(f"{'='*70}")

    run.log({"final/accuracy":          final_acc,
             "final/sparsity":          final_sp,
             "final/delta_baseline_pp": final_acc - final_base})
    run.finish()


if __name__ == '__main__':
    main()