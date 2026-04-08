"""
Structured RL-based One-Shot Regrowth
======================================
One-shot version of the structured regrowth pipeline.
Reward = accuracy - fixed threshold (no baseline interpolation).
Budget controlled by --sparsity_delta (e.g. 0.025 = drop sparsity 2.5pp).

Fix: added sparsity verification after model rebuild + new_config completeness check.
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
import torch_pruning as tp

from utils.model_loader import model_loader
from utils.data_loader import data_loader
from utils.analysis_utils import (
    BlockwiseFeatureExtractor,
    compute_block_ssim,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def compute_channel_sparsity(model, original_channels):
    total, remaining = 0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and name in original_channels:
            total     += original_channels[name]
            remaining += m.out_channels
    return (1 - remaining / total) if total > 0 else 0.0


def get_target_layers(dense_model, pruned_model):
    target_layers = []
    d_mods = dict(dense_model.named_modules())
    for name, p_m in pruned_model.named_modules():
        if not isinstance(p_m, nn.Conv2d):
            continue
        d_m = d_mods.get(name)
        if d_m and p_m.out_channels < d_m.out_channels:
            target_layers.append(name)
    return target_layers


def get_layer_capacities(dense_model, pruned_model, target_layers):
    d_mods = dict(dense_model.named_modules())
    p_mods = dict(pruned_model.named_modules())
    caps = []
    for name in target_layers:
        d_m, p_m = d_mods.get(name), p_mods.get(name)
        if d_m and p_m and isinstance(d_m, nn.Conv2d):
            caps.append(max(d_m.out_channels - p_m.out_channels, 0))
        else:
            caps.append(0)
    return caps


def get_pruned_channel_indices(dense_model, pruned_model, target_layers):
    pruned_indices = {}
    d_mods = dict(dense_model.named_modules())
    p_mods = dict(pruned_model.named_modules())

    for name in target_layers:
        d_m = d_mods.get(name)
        p_m = p_mods.get(name)
        if d_m is None or p_m is None or not isinstance(d_m, nn.Conv2d):
            pruned_indices[name] = []
            continue

        d_w = d_m.weight.data
        p_w = p_m.weight.data
        min_in = min(p_w.shape[1], d_w.shape[1])

        kept = []
        for pw in p_w:
            diffs = [(pw[:min_in] - d_w[j][:min_in]).abs().sum().item()
                     for j in range(d_w.shape[0])]
            kept.append(int(torch.tensor(diffs).argmin()))

        pruned_indices[name] = [i for i in range(d_m.out_channels) if i not in kept]

    return pruned_indices


def apply_config(dense_model, config, original_channels, example_inputs):
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


def apply_config_with_weight_transfer(current_model, dense_model,
                                      new_config, original_channels,
                                      example_inputs):
    new_model = apply_config(dense_model, new_config, original_channels, example_inputs)

    cur_mods = dict(current_model.named_modules())
    with torch.no_grad():
        for name, new_m in new_model.named_modules():
            if not isinstance(new_m, nn.Conv2d):
                continue
            cur_m = cur_mods.get(name)
            if cur_m is None:
                continue
            n_out = min(new_m.weight.shape[0], cur_m.weight.shape[0])
            n_in  = min(new_m.weight.shape[1], cur_m.weight.shape[1])
            new_m.weight[:n_out, :n_in] = cur_m.weight[:n_out, :n_in].data
            if new_m.bias is not None and cur_m.bias is not None:
                new_m.bias[:n_out] = cur_m.bias[:n_out].data

        for name, new_m in new_model.named_modules():
            if not isinstance(new_m, nn.BatchNorm2d):
                continue
            cur_m = cur_mods.get(name)
            if cur_m is None:
                continue
            n = min(new_m.num_features, cur_m.num_features)
            new_m.weight[:n]       = cur_m.weight[:n].data
            new_m.bias[:n]         = cur_m.bias[:n].data
            new_m.running_mean[:n] = cur_m.running_mean[:n].data
            new_m.running_var[:n]  = cur_m.running_var[:n].data

    return new_model


# ═══════════════════════════════════════════════════════════════════════════════
# [FIX] Verify actual channel counts after rebuild
# ═══════════════════════════════════════════════════════════════════════════════

def verify_rebuild(new_model, new_config, original_channels, pruned_sp, label="new_model"):
    """
    Compare intended new_config vs actual channel counts in rebuilt model.
    Prints warnings if torch_pruning didn't honor our requested config
    (can happen with EfficientNet due to structural coupling constraints).
    Returns actual sparsity.
    """
    actual_config = get_current_config(new_model)
    actual_sp     = compute_channel_sparsity(new_model, original_channels)
    mismatch      = 0

    print(f"\n  ── Rebuild Verification [{label}] ──")
    print(f"  pruned_sp={pruned_sp:.4f}  actual_sp={actual_sp:.4f}  "
          f"Δ={pruned_sp - actual_sp:.4f}")

    for name in new_config:
        intended = new_config[name]
        actual   = actual_config.get(name, -1)
        orig     = original_channels.get(name, -1)
        if intended != actual:
            mismatch += 1
            print(f"  !! MISMATCH {name}: intended={intended}  actual={actual}  "
                  f"orig={orig}")

    if mismatch == 0:
        print(f"  ✓ All {len(new_config)} layers match intended config")
    else:
        print(f"  !! {mismatch} layers did NOT match — "
              f"torch_pruning enforced structural constraints")
    print()
    return actual_sp


# ═══════════════════════════════════════════════════════════════════════════════
# SSIM Layer Selector
# ═══════════════════════════════════════════════════════════════════════════════

class SSIMLayerSelector:
    @staticmethod
    def update_search_space(sparse_model, pretrained_model,
                            data_loader_ref, target_layers,
                            threshold: float = 0.0,
                            num_batches: int = 64) -> tuple:
        block_dict = {'all_layers': target_layers}
        ext_pre  = BlockwiseFeatureExtractor(pretrained_model, block_dict)
        ext_spar = BlockwiseFeatureExtractor(sparse_model,     block_dict)

        with torch.no_grad():
            feats_pre  = ext_pre.extract_block_features(data_loader_ref,  num_batches=num_batches)
            feats_spar = ext_spar.extract_block_features(data_loader_ref, num_batches=num_batches)

        ssim_raw   = compute_block_ssim(feats_pre, feats_spar)
        block_ssim = ssim_raw.get('all_layers', {})

        ssim_dict, selected = {}, []
        for lname in target_layers:
            score = float(block_ssim.get(lname, 0.5))
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
# Taylor Channel Scorer
# ═══════════════════════════════════════════════════════════════════════════════

class TaylorChannelScorer:
    def __init__(self, dense_model, device='cuda'):
        self.dense_model = dense_model
        self.device      = device

    def compute(self, sparse_model, target_layers, data_loader, n_batches=10):
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

        d_mods         = dict(self.dense_model.named_modules())
        channel_scores = {}

        for layer_name in target_layers:
            d_m = d_mods.get(layer_name)
            p_m = dict(sparse_model.named_modules()).get(layer_name)

            if d_m is None or p_m is None or not isinstance(d_m, nn.Conv2d):
                channel_scores[layer_name] = {}
                continue
            if p_m.weight.grad is None:
                channel_scores[layer_name] = {}
                continue

            p_grad    = p_m.weight.grad.detach()
            d_weight  = d_m.weight.detach()
            mean_grad = p_grad.mean(dim=0)

            scores = {}
            for ch_idx in range(d_m.out_channels):
                d_w    = d_weight[ch_idx]
                min_in = min(mean_grad.shape[0], d_w.shape[0])
                scores[ch_idx] = (mean_grad[:min_in] * d_w[:min_in]).abs().sum().item()

            channel_scores[layer_name] = scores
            print(f"  {layer_name}: {d_m.out_channels} ch, "
                  f"max_taylor={max(scores.values()):.3e}")

        return channel_scores


def select_channels_by_taylor(channel_scores, pruned_ch_indices, layer_name, n_restore):
    all_scores = channel_scores.get(layer_name, {})
    pruned_chs = pruned_ch_indices.get(layer_name, [])
    if not all_scores or not pruned_chs or n_restore == 0:
        return []
    candidates = {ch: all_scores[ch] for ch in pruned_chs if ch in all_scores}
    if not candidates:
        return []
    n = min(n_restore, len(candidates))
    return [ch for ch, _ in sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:n]]


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM Controller
# ═══════════════════════════════════════════════════════════════════════════════

class RegrowthAgent(nn.Module):
    def __init__(self, budget_space_size, alloc_space_size,
                 hidden_size, context_dim, device='cuda'):
        super().__init__()
        self.DEVICE    = device
        self.nhid      = hidden_size
        self.input_dim = max(budget_space_size, alloc_space_size)

        self.lstm           = nn.LSTMCell(self.input_dim + context_dim, hidden_size)
        self.budget_decoder = nn.Linear(hidden_size, budget_space_size)
        self.alloc_decoder  = nn.Linear(hidden_size, alloc_space_size)
        self.hidden         = self.init_hidden()

    def forward(self, prev_logits, context_vec, step='alloc'):
        if prev_logits.dim() == 1: prev_logits = prev_logits.unsqueeze(0)
        if context_vec.dim()  == 1: context_vec = context_vec.unsqueeze(0)
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
# One-Shot Structured Regrowth PG
# ═══════════════════════════════════════════════════════════════════════════════

class OneshotStructuredRegrowthPG:

    def __init__(self, config, model_sparse, dense_model,
                 channel_scores, pruned_ch_indices,
                 original_channels, example_inputs,
                 target_layers, train_loader, test_loader,
                 device, wandb_run=None):

        # ── Hyperparameters ───────────────────────────────────────────────────
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

        # ── Accuracy threshold ────────────────────────────────────────────────
        self.acc_threshold = config['acc_threshold']

        # ── Models & data ─────────────────────────────────────────────────────
        self.model_sparse      = model_sparse
        self.dense_model       = dense_model
        self.target_layers     = target_layers
        self.train_loader      = train_loader
        self.test_loader       = test_loader
        self.channel_scores    = channel_scores
        self.pruned_ch_indices = pruned_ch_indices
        self.original_channels = original_channels
        self.example_inputs    = example_inputs
        self.current_config    = get_current_config(model_sparse)

        # ── [FIX] Record pruned model sparsity for verification ───────────────
        self.pruned_sp = compute_channel_sparsity(model_sparse, original_channels)

        # ── Capacities ────────────────────────────────────────────────────────
        self.layer_capacities = config['layer_capacities']
        self.total_capacity   = max(sum(self.layer_capacities), 1)

        # ── Early stopping ────────────────────────────────────────────────────
        self.early_stop_patience  = config.get('early_stop_patience', 40)
        self.min_epochs           = config.get('min_epochs', 50)
        self.reward_std_threshold = config.get('reward_std_threshold', 0.002)
        self.reward_window_size   = config.get('reward_window_size', 20)
        self.finetune_epochs      = config.get('finetune_epochs', 5)

        # ── Tracking ──────────────────────────────────────────────────────────
        self._best_model       = None
        self._best_reward_seen = float('-inf')

        # ── Logging / checkpointing ───────────────────────────────────────────
        self.run            = wandb_run
        self.model_name     = config.get('model_name')
        self.method         = config.get('method', 'structured_oneshot')
        self.checkpoint_dir = config.get('checkpoint_dir', './structured_oneshot_rl_ckpts')
        self.model_sparsity = config.get('model_sparsity', '0.5')

        # ── Entropy schedule ──────────────────────────────────────────────────
        self.use_entropy_schedule = config.get('use_entropy_schedule', True)
        self.start_beta           = config.get('start_beta', 0.4)
        self.end_beta             = config.get('end_beta', 0.04)
        self.decay_fraction       = config.get('decay_fraction', 0.4)

        # ── Budget options ────────────────────────────────────────────────────
        target_ch = config['target_restore_ch']
        self.budget_options = [config['target_restore_ch']] * self.BUDGET_SPACE
        while len(self.budget_options) < self.BUDGET_SPACE:
            self.budget_options.append(self.budget_options[-1])

        print(f"  Sparsity delta : {config.get('sparsity_delta', '?')}")
        print(f"  Target restore : {target_ch} channels")
        print(f"  Budget options : {self.budget_options}")

        # ── LSTM controller ───────────────────────────────────────────────────
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

        print(f"  Accuracy threshold: {self.acc_threshold:.2f}%")
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

    # ── Main RL loop ──────────────────────────────────────────────────────────

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
                best_config, best_frac      = new_config, frac
                self._save_best(epoch, reward, new_config, frac)

            beta      = self.get_entropy_coef(epoch)
            loss, ent = self.calculate_loss(
                ep_budget_logits, ep_alloc_logits, ep_wlp, beta)
            self.adam.zero_grad()
            loss.backward()
            self.adam.step()

            no_imp  = epoch - best_reward_ep
            rwd_std = float(np.std(list(reward_window))) if len(reward_window) > 1 else float('inf')

            if self.run:
                self.run.log({
                    'epoch':      epoch + 1,
                    'reward_pp':  reward * 100,
                    'best_pp':    best_reward * 100,
                    'loss':       loss.item(),
                    'entropy':    ent.item(),
                    'beta':       beta,
                    'budget_ch':  frac,
                    'sparsity':   sparsity,
                    'no_improve': no_imp,
                    'rwd_std':    rwd_std,
                })

            print(f"Ep {epoch + 1:3d}/{self.NUM_EPOCHS} | "
                  f"Rwd={reward * 100:+.2f}pp | Best={best_reward * 100:+.2f}pp | "
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

        print(f"\nBest: {best_reward * 100:+.2f}pp  budget_ch={best_frac}"
              + (f"  [{stop_reason}]" if stop_reason else ""))
        return best_config, best_reward, best_frac

    # ── Episode ───────────────────────────────────────────────────────────────

    def play_episode(self, epoch):
        t0 = time.time()
        self.agent.hidden = self.agent.init_hidden()
        prev_logits       = torch.zeros(1, self.BUDGET_SPACE, device=self.DEVICE)
        all_log_probs     = []
        budget_masked_logits = []
        alloc_masked_logits  = []

        # ── Step 0: Budget ────────────────────────────────────────────────────
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
        ratio_opts = (torch.arange(self.ALLOC_SPACE, device=self.DEVICE, dtype=torch.float)
                      / (self.ALLOC_SPACE - 1))
        remaining  = target_ch
        sel_counts, pnames = [], []

        for p_idx, (lname, orig_idx) in enumerate(self.layer_priority):
            cap = int(self.layer_capacities[orig_idx])
            ctx = torch.tensor([
                (p_idx + 1) / (self.NUM_STEPS + 1),
                cap / self.total_capacity if self.total_capacity > 0 else 0.0,
                remaining / target_ch    if target_ch > 0 else 0.0,
            ], dtype=torch.float, device=self.DEVICE).unsqueeze(0)

            logits   = self.agent(prev_logits, ctx, step='alloc').squeeze(0)
            eff_max  = min(cap, remaining)
            c_opts   = torch.round(ratio_opts * eff_max).to(torch.long)
            feasible = c_opts <= remaining
            if not feasible.any(): feasible[0] = True

            masked = torch.where(feasible, logits, torch.full_like(logits, -1e9))
            dist   = Categorical(probs=F.softmax(masked, dim=0))
            action = dist.sample()
            chosen = int(c_opts[action].item())
            chosen = min(chosen, cap, remaining)
            remaining = max(remaining - chosen, 0)

            sel_counts.append(chosen)
            pnames.append(lname)
            all_log_probs.append(dist.log_prob(action))
            alloc_masked_logits.append(masked)
            prev_logits = logits.unsqueeze(0)

        ep_log_probs     = torch.stack(all_log_probs)
        ep_budget_logits = torch.stack(budget_masked_logits)
        ep_alloc_logits  = torch.stack(alloc_masked_logits)
        allocation = {n: int(c) for n, c in zip(pnames, sel_counts) if c > 0}
        print(f"  [Alloc] {sum(sel_counts)}/{target_ch} | {allocation}")

        # ── Taylor channel selection → update config ──────────────────────────
        new_config = copy.copy(self.current_config)
        for lname, n_restore in allocation.items():
            chs = select_channels_by_taylor(
                self.channel_scores, self.pruned_ch_indices, lname, n_restore)
            if chs:
                new_config[lname] = self.current_config[lname] + len(chs)
                print(f"    {lname}: +{len(chs)} ch (taylor top3: {chs[:3]})")

        # ── [FIX] Print intended config delta before rebuild ──────────────────
        intended_delta = sum(
            new_config.get(n, 0) - self.current_config.get(n, 0)
            for n in new_config
        )
        print(f"  [Config] intended total channel gain: +{intended_delta}")

        # ── Rebuild model with weight transfer ────────────────────────────────
        new_model = apply_config_with_weight_transfer(
            current_model=self.model_sparse,
            dense_model=self.dense_model,
            new_config=new_config,
            original_channels=self.original_channels,
            example_inputs=self.example_inputs,
        ).to(self.DEVICE)

        # ── [FIX] Verify actual sparsity after rebuild ────────────────────────
        # Only do full verification on epoch 0 to avoid verbose logs
        if epoch == 0:
            actual_sp = verify_rebuild(
                new_model, new_config, self.original_channels,
                self.pruned_sp, label=f"ep{epoch}"
            )
        else:
            actual_sp = compute_channel_sparsity(new_model, self.original_channels)

        # ── [FIX] Warn if sparsity didn't change as expected ─────────────────
        expected_sp = self.pruned_sp - (
            intended_delta / max(sum(self.original_channels.values()), 1)
        )
        if abs(actual_sp - self.pruned_sp) < 0.001:
            print(f"  !! WARNING: sparsity unchanged after rebuild! "
                  f"pruned={self.pruned_sp:.4f}  actual={actual_sp:.4f}  "
                  f"expected≈{expected_sp:.4f}")
            print(f"  !! This likely means torch_pruning enforced structural "
                  f"constraints (e.g. EfficientNet depthwise coupling).")
            print(f"  !! Consider using --ssim_threshold=-1 to select ALL layers "
                  f"together, or reduce --sparsity_delta.")
        else:
            print(f"  [Sparsity] pruned={self.pruned_sp:.4f} → "
                  f"actual={actual_sp:.4f}  Δ={self.pruned_sp - actual_sp:.4f}")

        # ── Mini finetune ─────────────────────────────────────────────────────
        self.mini_finetune(new_model, epochs=self.finetune_epochs)

        accuracy = self.evaluate_model(new_model, full_eval=True)
        sparsity = compute_channel_sparsity(new_model, self.original_channels)

        # ── Reward = acc - fixed threshold ────────────────────────────────────
        reward = (accuracy - self.acc_threshold) / 100.0

        print(f"  [Reward] acc={accuracy:.2f}%  "
              f"threshold={self.acc_threshold:.2f}%  "
              f"Δ={accuracy - self.acc_threshold:+.2f}pp  reward={reward:+.4f}")

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

        print(f"  [{time.time() - t0:.1f}s]")
        return (ep_wlp, ep_budget_logits, ep_alloc_logits,
                reward, new_config, sparsity, target_ch)

    def _save_best(self, epoch, reward, config, budget_ch):
        d = os.path.join(self.checkpoint_dir,
                         f'{self.model_name}/{self.method}/{self.model_sparsity}')
        os.makedirs(d, exist_ok=True)
        if self._best_model is not None:
            model_to_save = copy.deepcopy(self._best_model)
            for m in model_to_save.modules():
                m._forward_hooks.clear()
                m._backward_hooks.clear()
                m._forward_pre_hooks.clear()
            torch.save(model_to_save,
                       os.path.join(d, f'best_model_rwd{reward * 100:+.2f}pp.pth'))
        torch.save({'epoch': epoch, 'reward': reward,
                    'config': config, 'budget_ch': budget_ch},
                   os.path.join(d, 'best_allocation.pth'))
        print(f"  ✓ Best {reward * 100:+.2f}pp  budget_ch={budget_ch}  saved → {d}")
        if self.run:
            self.run.log({"best_reward_pp": reward * 100, "best_epoch": epoch + 1})


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
    parser.add_argument('--m_name',      type=str,   default='effnet')
    parser.add_argument('--data_dir',    type=str,   default='./data')
    parser.add_argument('--method',      type=str,   default='structured_oneshot')
    parser.add_argument('--pruned_ckpt', type=str,   default="./effnet/ckpt_after_prune_structured_oneshot/"
                                                             "pruned_structured_l1_sp0.9_it1.pth")

    # Threshold
    parser.add_argument('--acc_threshold', type=float, default=71.31,
                        help='Reward = acc - threshold (pp)')

    # Sparsity delta
    parser.add_argument('--sparsity_delta', type=float, default=0.04,
                        help='Target sparsity drop, e.g. 0.025 = restore 2.5%% of original channels')

    # RL
    parser.add_argument('--num_epochs',         type=int,   default=300)
    parser.add_argument('--learning_rate',      type=float, default=3e-4)
    parser.add_argument('--hidden_size',        type=int,   default=64)
    parser.add_argument('--entropy_coef',       type=float, default=0.5)
    parser.add_argument('--reward_temperature', type=float, default=0.005)
    parser.add_argument('--start_beta',         type=float, default=0.40)
    parser.add_argument('--end_beta',           type=float, default=0.04)
    parser.add_argument('--decay_fraction',     type=float, default=0.4)
    parser.add_argument('--budget_space_size',  type=int,   default=5)
    parser.add_argument('--alloc_space_size',   type=int,   default=11)

    # SSIM
    parser.add_argument('--ssim_threshold',   type=float, default=0.0)
    parser.add_argument('--ssim_num_batches', type=int,   default=64)

    # Taylor
    parser.add_argument('--taylor_batches', type=int, default=10)

    # Finetune
    parser.add_argument('--finetune_epochs', type=int, default=40)

    # Early stopping
    parser.add_argument('--early_stop_patience',  type=int,   default=40)
    parser.add_argument('--min_epochs',           type=int,   default=50)
    parser.add_argument('--reward_std_threshold', type=float, default=0.002)
    parser.add_argument('--reward_window_size',   type=int,   default=20)

    # Misc
    parser.add_argument('--save_dir', type=str, default='./structured_rl_ckpts')
    parser.add_argument('--seed',     type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    train_loader, _, test_loader = data_loader(data_dir=args.data_dir)
    example_inputs, _ = next(iter(train_loader))
    example_inputs = example_inputs[:1].to(device)

    # ── Dense (pretrained) model ──────────────────────────────────────────────
    dense_model = model_loader(args.m_name, device)
    ckpt = torch.load(f'./{args.m_name}/checkpoint/pretrain_{args.m_name}_ckpt.pth',
                      weights_only=False)
    dense_model.load_state_dict(ckpt['net'])
    dense_model.eval()
    original_channels = get_original_channels(dense_model)

    # ── Pruned model ──────────────────────────────────────────────────────────
    assert args.pruned_ckpt, "Provide --pruned_ckpt path to a structured-pruned model."
    pruned_model = torch.load(args.pruned_ckpt, map_location=device, weights_only=False)
    pruned_model.eval()

    sp0  = compute_channel_sparsity(pruned_model, original_channels)
    acc0 = quick_eval(pruned_model, test_loader, device)
    print(f"\nStarting point → Acc: {acc0:.2f}%  ChannelSparsity: {sp0:.4f}")
    print(f"Threshold: {args.acc_threshold:.2f}%  Δ={acc0 - args.acc_threshold:+.2f}pp\n")

    # ── Target layers ─────────────────────────────────────────────────────────
    target_layers = get_target_layers(dense_model, pruned_model)
    print(f"Found {len(target_layers)} pruned layers:")
    for name in target_layers:
        d_ch = original_channels[name]
        p_ch = dict(pruned_model.named_modules())[name].out_channels
        print(f"  {name}: {d_ch} → {p_ch}  (pruned {d_ch - p_ch})")

    # ── target_restore_ch：由 sparsity_delta 决定 ─────────────────────────────
    total_original_ch = sum(original_channels.values())
    layer_capacities  = get_layer_capacities(dense_model, pruned_model, target_layers)
    target_restore_ch = int(total_original_ch * args.sparsity_delta)
    target_restore_ch = min(target_restore_ch, sum(layer_capacities))

    print(f"\nSparsity delta   : {args.sparsity_delta:.4f}")
    print(f"Total original ch: {total_original_ch}")
    print(f"Target restore ch: {target_restore_ch}  (cap={sum(layer_capacities)})\n")

    if sum(layer_capacities) == 0:
        print("No channels to restore. Exiting.")
        return

    # ── SSIM layer selection ──────────────────────────────────────────────────
    selected_layers, ssim_scores = SSIMLayerSelector.update_search_space(
        sparse_model=pruned_model,
        pretrained_model=dense_model,
        data_loader_ref=test_loader,
        target_layers=target_layers,
        threshold=args.ssim_threshold,
        num_batches=args.ssim_num_batches,
    )

    # ── Taylor channel scores ─────────────────────────────────────────────────
    print("Computing Taylor channel scores…")
    scorer = TaylorChannelScorer(dense_model=dense_model, device=device)
    channel_scores = scorer.compute(
        sparse_model=pruned_model,
        target_layers=selected_layers,
        data_loader=train_loader,
        n_batches=args.taylor_batches,
    )

    pruned_ch_indices = get_pruned_channel_indices(dense_model, pruned_model, selected_layers)
    layer_capacities  = get_layer_capacities(dense_model, pruned_model, selected_layers)

    # ── [FIX] Sanity check: does apply_config actually change sparsity? ───────
    print("\n── Sanity check: apply_config on selected_layers ──")
    test_config = get_current_config(pruned_model)
    for lname in selected_layers:
        cap = dict(zip(target_layers, get_layer_capacities(dense_model, pruned_model, target_layers)))
        restore_n = min(10, cap.get(lname, 0))
        if restore_n > 0:
            test_config[lname] = test_config[lname] + restore_n

    test_model = apply_config(dense_model, test_config, original_channels, example_inputs)
    test_sp    = compute_channel_sparsity(test_model, original_channels)
    print(f"  pruned_sp={sp0:.4f}  after_test_restore_sp={test_sp:.4f}  "
          f"Δ={sp0 - test_sp:.4f}")
    if abs(test_sp - sp0) < 0.001:
        print("  !! WARNING: apply_config didn't change sparsity in sanity check!")
        print("  !! Check if torch_pruning respects per-layer pruning_ratio_dict")
        print("  !! for this model architecture (EfficientNet depthwise constraints).")
    else:
        print("  ✓ apply_config correctly changes sparsity")
    del test_model
    print()

    # ── WandB ─────────────────────────────────────────────────────────────────
    run = wandb.init(
        project="structured_oneshot_rl_regrowth",
        name=f"{args.m_name}_oneshot_sp{sp0:.3f}_delta{args.sparsity_delta}",
        config=vars(args) | {"start_acc": acc0, "start_sp": sp0,
                             "target_restore_ch": target_restore_ch},
    )

    # ── RL config ─────────────────────────────────────────────────────────────
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
        'model_sparsity':       f"sp{sp0:.3f}",
        'method':               args.method,
        'finetune_epochs':      args.finetune_epochs,
        'acc_threshold':        args.acc_threshold,
        'target_restore_ch':    target_restore_ch,
        'sparsity_delta':       args.sparsity_delta,
    }

    pg = OneshotStructuredRegrowthPG(
        config=config,
        model_sparse=pruned_model,
        dense_model=dense_model,
        channel_scores=channel_scores,
        pruned_ch_indices=pruned_ch_indices,
        original_channels=original_channels,
        example_inputs=example_inputs,
        target_layers=selected_layers,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        wandb_run=run,
    )

    best_config, best_reward, best_frac = pg.solve_environment()

    # ── Final evaluation ──────────────────────────────────────────────────────
    best_model = pg._best_model or pruned_model
    final_acc  = quick_eval(best_model, test_loader, device)
    final_sp   = compute_channel_sparsity(best_model, original_channels)

    print(f"\n{'=' * 70}")
    print(f"DONE | Acc={final_acc:.2f}%  Sp={final_sp:.4f}  "
          f"Threshold={args.acc_threshold:.2f}%  "
          f"Δ={final_acc - args.acc_threshold:+.2f}pp")
    print(f"{'=' * 70}")

    run.log({"final/accuracy":           final_acc,
             "final/sparsity":           final_sp,
             "final/delta_threshold_pp": final_acc - args.acc_threshold})
    run.finish()


if __name__ == '__main__':
    main()