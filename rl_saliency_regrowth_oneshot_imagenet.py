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
from transformers import get_cosine_schedule_with_warmup
from collections import deque
from tqdm import tqdm
from utils.data_loader_tiny_imagenet import data_loader_tiny_imagenet
from utils.model_loader import model_loader
from utils.data_loader import data_loader
from utils.analysis_utils import (
    load_model_name, prune_weights_reparam, count_pruned_params,
    BlockwiseFeatureExtractor, compute_block_ssim,   # ← 新增两个
)

# ─────────────────────────────────────────────────────────
method = "oneshot"
model_name = "vgg16ImageNet"

run = wandb.init(
    project="ICCAD_Regrowth_oneshot_imagenet",
    name=f"{model_name}_regrowth_from99.5_to0.97",
    config={
        "learning_rate": 5e-3,
        "architecture": "VGG16",
        "regrow_step": 0.025,
        "epochs": 400,
    },
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


# ═══════════════════════════════════════════════════════════════
# ★ 新增：SSIM 自动选层（直接从第二个文件移植）
# ═══════════════════════════════════════════════════════════════

class SSIMLayerSelector:
    """
    One-shot SSIM-based layer selector.

    Scans ALL layers with weight_mask in the sparse model,
    computes feature-map SSIM vs. pretrained, and selects
    layers with SSIM < threshold.
    Fallback: single worst layer if nothing qualifies.
    """

    @staticmethod
    def select_layers(sparse_model, pretrained_model,
                      data_loader_ref,
                      threshold: float = 0.0,
                      num_batches: int = 64) -> tuple:
        """
        Returns (selected_layers, ssim_dict).

        selected_layers : list[str]  layers whose feature-SSIM < threshold
        ssim_dict       : dict[str, float]  SSIM for ALL masked layers
        """
        all_masked = [name for name, m in sparse_model.named_modules()
                      if hasattr(m, 'weight_mask') and len(name) > 0]

        block_dict = {'all_layers': all_masked}
        ext_pre  = BlockwiseFeatureExtractor(pretrained_model, block_dict)
        ext_spar = BlockwiseFeatureExtractor(sparse_model,     block_dict)

        with torch.no_grad():
            feats_pre  = ext_pre.extract_block_features(data_loader_ref,
                                                        num_batches=num_batches)
            feats_spar = ext_spar.extract_block_features(data_loader_ref,
                                                         num_batches=num_batches)

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
            flag = "  ← SELECTED" if n in selected else ""
            print(f"    {n}: {s:+.4f}{flag}")

        if not selected:
            worst    = min(ssim_dict, key=ssim_dict.get)
            selected = [worst]
            print(f"  No layer below {threshold:+.3f} → fallback: "
                  f"{worst} ({ssim_dict[worst]:+.4f})")

        print(f"  Selected: {len(selected)}/{len(all_masked)} layers\n")
        return selected, ssim_dict


# ═══════════════════════════════════════════════════════════════
# 以下类与原代码完全一致，仅保留，不做任何修改
# ═══════════════════════════════════════════════════════════════

class SaliencyComputer:
    """
    Computes gradient-based saliency scores for parameters using FairPrune formula.
    (unchanged)
    """

    def __init__(self, model, criterion, device='cuda'):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.accumulated_grads = {}
        self.grad_count = 0

    def reset_accumulated_grads(self):
        self.accumulated_grads = {}
        self.grad_count = 0

    def compute_saliency_scores(self, data_loader, target_layers,
                                max_batches=None, num_classes=10):
        self.model.eval()
        self.reset_accumulated_grads()

        print(f"\nComputing saliency scores (RigL-style)...")
        print(f"  Target layers: {len(target_layers)}")
        print(f"  Max batches: {max_batches}")

        module_dict = dict(self.model.named_modules())

        for layer_name in target_layers:
            module = module_dict.get(layer_name)
            if module is not None and hasattr(module, 'weight'):
                self.accumulated_grads[layer_name] = torch.zeros(
                    module.weight.shape, device=self.device)

        batch_count = 0

        for inputs, labels in tqdm(data_loader, desc="Accumulating gradients"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            grads = torch.autograd.grad(loss, self.model.parameters(),
                                        create_graph=False,allow_unused=True)

            for param, grad in zip(self.model.parameters(), grads):
                if grad is None:
                    continue
                for name, p in self.model.named_parameters():
                    if p is param:
                        for layer_name in target_layers:
                            if name == f"{layer_name}.weight":
                                hessian_approx = grad.pow(2).detach()
                                param_squared  = param.data.pow(2).detach()
                                self.accumulated_grads[layer_name] += (
                                    hessian_approx * param_squared)
                                break
                        break

            batch_count += 1
            self.grad_count += 1

        saliency_dict = {}
        for layer_name in target_layers:
            if layer_name in self.accumulated_grads:
                saliency = self.accumulated_grads[layer_name] / max(self.grad_count, 1)
                saliency_dict[layer_name] = saliency.cpu()
                print(f"  {layer_name}: mean={saliency.mean().item():.6e}, "
                      f"std={saliency.std().item():.6e}, "
                      f"max={saliency.max().item():.6e}")

        return saliency_dict


class RegrowthAgent(nn.Module):
    """LSTM-based controller (unchanged)"""

    def __init__(self, action_dim, hidden_size, num_layers, context_dim, device='cuda'):
        super(RegrowthAgent, self).__init__()
        self.DEVICE = device
        self.num_layers = num_layers
        self.nhid = hidden_size
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.input_dim = action_dim + context_dim

        self.lstm    = nn.LSTMCell(self.input_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, action_dim)
        self.hidden  = self.init_hidden()

    def forward(self, prev_logits, context_vec):
        if prev_logits.dim() == 1:  prev_logits = prev_logits.unsqueeze(0)
        if context_vec.dim() == 1:  context_vec = context_vec.unsqueeze(0)
        lstm_input   = torch.cat([prev_logits, context_vec], dim=-1)
        h_t, c_t     = self.lstm(lstm_input, self.hidden)
        self.hidden   = (h_t, c_t)
        return self.decoder(h_t)

    def init_hidden(self):
        h_t = torch.zeros(1, self.nhid, dtype=torch.float, device=self.DEVICE)
        c_t = torch.zeros(1, self.nhid, dtype=torch.float, device=self.DEVICE)
        return (h_t, c_t)


class SaliencyBasedRegrowth:
    """RigL-inspired saliency regrowth (unchanged)"""

    @staticmethod
    @torch.no_grad()
    def apply_regrowth(model, layer_name, saliency_tensor, num_weights,
                       init_strategy='zero', device='cuda'):
        module_dict = dict(model.named_modules())
        module = module_dict.get(layer_name)

        if module is None or not hasattr(module, 'weight_mask'):
            return 0, []

        current_mask = module.weight_mask
        saliency     = saliency_tensor.to(device)
        pruned_positions = (current_mask == 0)

        if not pruned_positions.any():
            return 0, []

        saliency_masked = saliency.clone()
        saliency_masked[~pruned_positions] = -float('inf')

        flat_saliency = saliency_masked.flatten()
        k = min(num_weights, (flat_saliency > -float('inf')).sum().item())
        if k == 0:
            return 0, []

        _, top_k_flat_indices = torch.topk(flat_saliency, k=k)
        shape = saliency.shape
        regrown_indices = []

        for flat_idx in top_k_flat_indices:
            multi_idx = np.unravel_index(flat_idx.cpu().item(), shape)
            regrown_indices.append(multi_idx)
            current_mask[multi_idx] = 1.0

            wp = getattr(module, 'weight_orig', module.weight)
            if init_strategy == 'zero':
                wp.data[multi_idx] = 0.0
            elif init_strategy == 'kaiming':
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(wp)
                bound = np.sqrt(6.0 / fan_in)
                wp.data[multi_idx] = torch.empty(1).uniform_(-bound, bound).item()
            elif init_strategy == 'xavier':
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(wp)
                bound = np.sqrt(6.0 / (fan_in + fan_out))
                wp.data[multi_idx] = torch.empty(1).uniform_(-bound, bound).item()
            # 'magnitude': keep existing value, do nothing

        return len(regrown_indices), regrown_indices


class RegrowthPolicyGradient:
    """RL Policy Gradient with Saliency-Based Regrowth (unchanged)"""

    def __init__(self, config, model_pretrained, model_99,
                 target_layers, train_loader, test_loader, device):
        self.NUM_EPOCHS         = config['num_epochs']
        self.ALPHA              = config['learning_rate']
        self.BATCH_SIZE         = config['batch_size']
        self.HIDDEN_SIZE        = config['hidden_size']
        self.BETA               = config['entropy_coef']
        self.REWARD_TEMPERATURE = config.get('reward_temperature', 0.01)
        self.DEVICE             = device
        self.ACTION_SPACE       = config['action_space_size']
        self.NUM_STEPS          = len(target_layers)
        self.CONTEXT_DIM        = config.get('context_dim', 3)
        self.BASELINE_DECAY     = config.get('baseline_decay', 0.9)

        self.model_pretrained = model_pretrained.to(device)
        self.model_99         = model_99.to(device)
        self.target_layers    = target_layers
        self.train_loader     = train_loader
        self.test_loader      = test_loader

        self.target_regrow    = config['target_regrow']
        self.layer_capacities = config['layer_capacities']
        self.total_capacity   = max(sum(self.layer_capacities), 1)
        self.init_strategy    = config.get('init_strategy', 'zero')
        self.saliency_max_batches = config.get('saliency_max_batches', 50)

        self.early_stop_patience   = config.get('early_stop_patience', 40)
        self.min_epochs            = config.get('min_epochs', 50)
        self.reward_std_threshold  = config.get('reward_std_threshold', 0.002)
        self.min_entropy_threshold = config.get('min_entropy_threshold', 0.05)
        self.reward_window_size    = config.get('reward_window_size', 20)

        self.acc_baseline      = config.get('acc_baseline', None)
        self.baseline_exceeded = False
        self._best_model_state = None
        self._best_reward_seen = float('-inf')

        self.saliency_computer = SaliencyComputer(
            model=self.model_pretrained,
            criterion=nn.CrossEntropyLoss(),
            device=self.DEVICE
        )

        print("\n" + "=" * 70)
        print("Computing Initial Saliency Scores")
        print("=" * 70)
        self.saliency_dict = self.saliency_computer.compute_saliency_scores(
            data_loader=self.train_loader,
            target_layers=self.target_layers,
            max_batches=self.saliency_max_batches,
            num_classes=10
        )
        print("=" * 70 + "\n")

        self.agent = RegrowthAgent(
            action_dim=self.ACTION_SPACE,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_STEPS,
            context_dim=self.CONTEXT_DIM,
            device=self.DEVICE
        ).to(self.DEVICE)

        self.adam             = optim.Adam(params=self.agent.parameters(), lr=self.ALPHA)
        self.reward_baseline  = None
        self.total_rewards    = deque([], maxlen=100)
        self._model_name      = config.get('model_name')
        self.checkpoint_dir   = config.get('checkpoint_dir', './rl_saliency_checkpoints')
        self.model_sparsity   = config.get('model_sparsity', '0.98')
        self.save_freq        = config.get('save_freq', 1)

        self.use_entropy_schedule = config.get('use_entropy_schedule', True)
        self.start_beta           = config.get('start_beta', 0.4)
        self.end_beta             = config.get('end_beta', 0.004)
        self.decay_fraction       = config.get('decay_fraction', 0.4)

        self.layer_priority = [(lname, idx)
                               for idx, lname in enumerate(self.target_layers)]

        print(f"\nLayer regrowth order:")
        for lname, orig_idx in self.layer_priority:
            print(f"  {orig_idx+1}. {lname}: capacity={int(self.layer_capacities[orig_idx])}")

        print(f"\nEarly stopping config:")
        print(f"  patience={self.early_stop_patience}  min_epochs={self.min_epochs}")
        print(f"  reward_std_thresh={self.reward_std_threshold}  "
              f"reward_window={self.reward_window_size}")
        if self.acc_baseline is not None:
            print(f"  acc_baseline={self.acc_baseline*100:.2f}%")

    # ── 以下方法与原代码完全相同 ──────────────────────────────────────────────

    def get_entropy_coef(self, epoch):
        if not self.use_entropy_schedule:
            return self.BETA
        decay_epochs = self.NUM_EPOCHS * self.decay_fraction
        if epoch < decay_epochs:
            return self.start_beta - (self.start_beta - self.end_beta) * (epoch / decay_epochs)
        return self.end_beta

    def _create_model_copy(self, source_model):
        new_model = model_loader(self._model_name, self.DEVICE)
        prune_weights_reparam(new_model)
        new_model.load_state_dict(source_model.state_dict())
        return new_model

    def solve_environment(self, resume_from=None):
        solve_start_time = time.time()
        print(f"\n{'='*70}\nRL Training with Saliency-Based Regrowth\n{'='*70}")

        best_reward, best_allocation, best_regrow_indices = float('-inf'), None, None
        start_epoch = 0
        reward_window = deque(maxlen=self.reward_window_size)
        best_reward_epoch = 0
        stop_reason = ""

        if resume_from and os.path.exists(resume_from):
            print(f"Resuming from: {resume_from}")
            checkpoint = torch.load(resume_from)
            self.agent.load_state_dict(checkpoint['agent_state_dict'])
            self.adam.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch   = checkpoint['epoch'] + 1
            best_reward   = checkpoint['best_reward']
            best_allocation   = checkpoint['best_allocation']
            best_regrow_indices = checkpoint['best_regrow_indices']
            if 'reward_baseline' in checkpoint:
                self.reward_baseline = checkpoint['reward_baseline']
            print(f"  Resumed from epoch {start_epoch}, best reward: {best_reward:.4f}\n")

        epoch = start_epoch
        while epoch < self.NUM_EPOCHS:
            (ep_weighted_log_prob, ep_logits, reward,
             allocation, sparsity, regrow_indices) = self.play_episode(
                solve_start_time, epoch)

            reward_window.append(reward)

            if reward > best_reward:
                best_reward       = reward
                best_reward_epoch = epoch
                best_allocation   = allocation
                best_regrow_indices = copy.deepcopy(regrow_indices)
                self._save_best_allocation(epoch, best_reward,
                                           best_allocation, best_regrow_indices)

            current_beta = self.get_entropy_coef(epoch)
            loss, entropy = self.calculate_loss(ep_logits, ep_weighted_log_prob,
                                                beta=current_beta)
            self.adam.zero_grad()
            loss.backward()
            self.adam.step()

            epochs_no_improve = epoch - best_reward_epoch
            reward_std = (float(np.std(list(reward_window)))
                          if len(reward_window) > 1 else float('inf'))
            should_stop = False

            if epochs_no_improve >= self.early_stop_patience:
                should_stop = True
                stop_reason = (f"No improvement for {self.early_stop_patience} epochs "
                               f"(last improved at epoch {best_reward_epoch+1})")
            elif (epoch >= self.min_epochs
                  and len(reward_window) >= self.reward_window_size
                  and reward_std < self.reward_std_threshold):
                should_stop = True
                stop_reason = (f"Reward std={reward_std:.5f} < {self.reward_std_threshold}")

            pg_norm = torch.norm(ep_weighted_log_prob).item()
            run.log({
                "epoch": epoch + 1,
                "epoch_loss": loss.item(),
                "epoch_entropy": entropy.item(),
                "epoch_pg_norm": pg_norm,
                "current_beta": current_beta,
                "reward": reward,
                "acc": reward * 100.0,
                "reward_baseline": self.reward_baseline,
                "sparsity": sparsity,
                "early_stop/epochs_no_improve": epochs_no_improve,
                "early_stop/reward_std": reward_std,
            })

            print(f"Epoch {epoch+1:3d}/{self.NUM_EPOCHS} | "
                  f"Reward: {reward:.4f} ({reward*100:.2f}%) | "
                  f"Best: {best_reward:.4f} ({best_reward*100:.2f}%) | "
                  f"Loss: {loss.item():.4f} | "
                  f"Entropy: {entropy.item():.4f} | "
                  f"Beta: {current_beta:.4f} | "
                  f"NoImprove: {epochs_no_improve} | "
                  f"RwdStd({self.reward_window_size}): {reward_std:.5f}")

            if should_stop:
                print(f"\n>>> Early stopping at epoch {epoch+1}: {stop_reason}")
                break

            epoch += 1

        print(f"\n{'='*70}\nTraining Completed!\n"
              f"Best Reward: {best_reward:.4f} ({best_reward*100:.2f}%)\n{'='*70}\n")
        return best_allocation, best_reward, best_regrow_indices

    def play_episode(self, solve_start_time, epoch=0):
        episode_start_time = time.time()
        self.agent.hidden = self.agent.init_hidden()
        prev_logits = torch.zeros(1, self.ACTION_SPACE, device=self.DEVICE)

        ratio_options = (torch.arange(self.ACTION_SPACE, device=self.DEVICE,
                                      dtype=torch.float) / (self.ACTION_SPACE - 1))

        total_budget    = int(self.target_regrow)
        remaining_budget = total_budget
        masked_logits_list, log_prob_list, selected_counts, priority_layer_names = [], [], [], []

        for priority_idx, (layer_name, orig_idx) in enumerate(self.layer_priority):
            layer_capacity   = int(self.layer_capacities[orig_idx])
            layer_position   = (priority_idx / max(self.NUM_STEPS - 1, 1)
                                if self.NUM_STEPS > 1 else 0.0)
            capacity_fraction  = layer_capacity / self.total_capacity
            remaining_fraction = (remaining_budget / total_budget
                                  if total_budget > 0 else 0.0)

            ctx = torch.tensor([layer_position, capacity_fraction, remaining_fraction],
                               dtype=torch.float, device=self.DEVICE).unsqueeze(0)
            logits_layer = self.agent(prev_logits, ctx).squeeze(0)

            effective_max  = min(layer_capacity, remaining_budget)
            counts_options = torch.round(ratio_options * effective_max).to(torch.long)
            feasible = counts_options <= remaining_budget
            if not torch.any(feasible):
                feasible[0] = True

            masked_logits = torch.where(feasible, logits_layer,
                                        torch.full_like(logits_layer, -1e9))
            probs     = F.softmax(masked_logits, dim=0)
            dist      = Categorical(probs=probs)
            action_idx = dist.sample()
            log_prob  = dist.log_prob(action_idx)

            chosen_count  = int(counts_options[action_idx].item())
            chosen_count  = min(chosen_count, layer_capacity, remaining_budget)
            remaining_budget = max(remaining_budget - chosen_count, 0)

            selected_counts.append(chosen_count)
            priority_layer_names.append(layer_name)
            log_prob_list.append(log_prob)
            masked_logits_list.append(masked_logits)
            prev_logits = logits_layer.unsqueeze(0)

        sampling_end_time      = time.time()
        time_from_solve_start  = sampling_end_time - solve_start_time
        print(f"  [Timing] Time from solve_environment start: {time_from_solve_start:.3f}s")

        episode_log_probs = torch.stack(log_prob_list)

        allocation   = {n: int(c) for n, c in zip(priority_layer_names, selected_counts) if c > 0}
        total_alloc  = sum(selected_counts)
        print(f"\n  Sampled allocation:")
        for i, lname in enumerate(priority_layer_names):
            if selected_counts[i] > 0:
                print(f"    {lname}: {selected_counts[i]} "
                      f"(cap={self.layer_capacities[self.layer_priority[i][1]]})")
        print(f"  Total: {total_alloc}/{total_budget} "
              f"({100*total_alloc/total_budget:.1f}%)")

        model_copy = self._create_model_copy(self.model_99)
        model_copy.eval()
        actual_regrown, regrow_indices = {}, {}

        for lname in priority_layer_names:
            if lname in allocation:
                sal = self.saliency_dict.get(lname)
                if sal is not None:
                    actual, indices = SaliencyBasedRegrowth.apply_regrowth(
                        model=model_copy, layer_name=lname,
                        saliency_tensor=sal,
                        num_weights=allocation[lname],
                        init_strategy=self.init_strategy,
                        device=self.DEVICE)
                    actual_regrown[lname]  = actual
                    regrow_indices[lname]  = indices

        pre_finetune_state = copy.deepcopy(model_copy.state_dict())
        self.mini_finetune(model_copy, epochs=50)

        accuracy          = self.evaluate_model(model_copy, full_eval=True)
        sparsity, _, _    = self.calculate_sparsity(model_copy)
        reward            = accuracy / 100.0

        if reward > self._best_reward_seen:
            self._best_reward_seen = reward
            self._best_model_state = copy.deepcopy(model_copy.state_dict())

        if self.acc_baseline is not None and reward > self.acc_baseline:
            model_copy.load_state_dict(pre_finetune_state)
            self._save_baseline_model(epoch, reward, model_copy, allocation)

        if self.reward_baseline is None:
            self.reward_baseline = reward

        baseline           = self.reward_baseline
        temperature        = max(self.REWARD_TEMPERATURE, 1e-6)
        normalized_adv     = float(np.clip(
            (reward - baseline) / temperature, -100.0, 100.0))

        run.log({"acc": accuracy, "reward": reward, "reward_baseline": baseline,
                 "centered_reward": reward - baseline,
                 "normalized_advantage": normalized_adv, "sparsity": sparsity})

        self.reward_baseline = (self.BASELINE_DECAY * self.reward_baseline
                                + (1 - self.BASELINE_DECAY) * reward)

        adv_t    = torch.tensor(normalized_adv, device=self.DEVICE, dtype=torch.float)
        ep_wlp   = torch.sum(episode_log_probs * adv_t).unsqueeze(0)
        ep_logits = (torch.stack(masked_logits_list)
                     if masked_logits_list else None)

        episode_end_time     = time.time()
        total_episode_dur    = episode_end_time - episode_start_time
        print(f"  [Timing] Total episode: {total_episode_dur:.3f}s")
        run.log({"timing/time_from_solve_start": time_from_solve_start,
                 "timing/total_episode": total_episode_dur})

        return ep_wlp, ep_logits, reward, allocation, sparsity, regrow_indices

    def calculate_loss(self, epoch_logits, weighted_log_probs, beta=None):
        if beta is None:
            beta = self.BETA
        policy_loss = -torch.mean(weighted_log_probs)
        if epoch_logits is None or epoch_logits.numel() == 0:
            entropy = torch.tensor(0.0, device=self.DEVICE)
        else:
            p   = F.softmax(epoch_logits, dim=1)
            log_p = F.log_softmax(epoch_logits, dim=1)
            entropy = -torch.mean(torch.sum(p * log_p, dim=1))
        return policy_loss - beta * entropy, entropy

    def calculate_sparsity(self, model):
        total, pruned = 0, 0
        for _, m in model.named_modules():
            if hasattr(m, 'weight_mask'):
                total  += m.weight_mask.numel()
                pruned += (m.weight_mask == 0).sum().item()
        return (100.0 * pruned / total if total > 0 else 0.0), total, pruned

    def mini_finetune(self, model, epochs=50, lr=0.0003):
        model.train()
        optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion  = nn.CrossEntropyLoss()
        best_acc, best_state = 0.0, None

        for _ in range(epochs):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
                optimizer.zero_grad()
                criterion(model(inputs), targets).backward()
                optimizer.step()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in self.test_loader:
                    inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
                    _, pred = model(inputs).max(1)
                    total   += targets.size(0)
                    correct += pred.eq(targets).sum().item()
            acc = 100.0 * correct / total
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(model.state_dict())
            model.train()

        if best_state:
            model.load_state_dict(best_state)
        model.eval()

    def evaluate_model(self, model, full_eval=False):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_loader):
                if not full_eval and i >= 20:
                    break
                inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)
                _, pred = model(inputs).max(1)
                total   += targets.size(0)
                correct += pred.eq(targets).sum().item()
        return 100.0 * correct / total

    def _save_baseline_model(self, epoch, accuracy, model, allocation):
        self.baseline_exceeded = True
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        save_dir = os.path.join(self.checkpoint_dir, f'{self._model_name}/{method}')
        os.makedirs(save_dir, exist_ok=True)
        filename = (f'{self.model_sparsity}/baseline_exceeded_'
                    f'epoch{epoch+1}_acc{accuracy:.4f}.pth')
        torch.save({
            'epoch': epoch, 'accuracy': accuracy,
            'model_state_dict': model.state_dict(),
            'allocation': allocation, 'acc_baseline': self.acc_baseline,
            'timestamp': time.time(),
        }, os.path.join(save_dir, filename))
        print(f"  ✓ Baseline exceeded! acc={accuracy*100:.2f}% > "
              f"{self.acc_baseline*100:.2f}%")
        run.log({"baseline_exceeded_acc": accuracy * 100.0,
                 "baseline_exceeded_epoch": epoch + 1})

    def _save_best_allocation(self, epoch, best_reward, best_allocation, best_regrow_indices):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        best_path = os.path.join(
            self.checkpoint_dir,
            f'{model_name}/{method}/{self.model_sparsity}/best_saliency_allocation.pth')
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        torch.save({
            'epoch': epoch, 'reward': best_reward,
            'accuracy': best_reward * 100.0,
            'allocation': best_allocation,
            'regrow_indices': best_regrow_indices,
            'timestamp': time.time(),
        }, best_path)
        print(f"  ✓ New best! Reward: {best_reward:.4f} "
              f"({best_reward*100:.2f}%) at epoch {epoch+1}")


# ═══════════════════════════════════════════════════════════════
# main  — 唯一显著改动：target_layers 由 SSIM 自动生成
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='RL with Saliency-Based Regrowth (SSIM auto-select)')
    parser.add_argument('--m_name',        type=str,   default='effnetTinyImageNet')
    parser.add_argument('--data_dir',      type=str,   default='./data')
    parser.add_argument('--model_sparsity',type=str,   default='0.99')

    # RL
    parser.add_argument('--num_epochs',        type=int,   default=400)
    parser.add_argument('--learning_rate',     type=float, default=3e-4)
    parser.add_argument('--hidden_size',       type=int,   default=64)
    parser.add_argument('--entropy_coef',      type=float, default=0.5)
    parser.add_argument('--reward_temperature',type=float, default=0.005)
    parser.add_argument('--start_beta',        type=float, default=0.40)
    parser.add_argument('--end_beta',          type=float, default=0.04)
    parser.add_argument('--decay_fraction',    type=float, default=0.4)
    parser.add_argument('--action_space_size', type=int,   default=11)

    # Regrowth
    parser.add_argument('--regrow_step',     type=float, default=0.01)
    parser.add_argument('--init_strategy',   type=str,   default='zero',
                        choices=['zero', 'kaiming', 'xavier', 'magnitude'])
    parser.add_argument('--saliency_max_batches', type=int, default=50)

    # ★ SSIM 自动选层（新增两个参数）
    parser.add_argument('--ssim_threshold',   type=float, default=0.0,
                        help='Layers with feature-SSIM < this value are selected. '
                             'Feature-SSIM can be negative. Default=0.0')
    parser.add_argument('--ssim_num_batches', type=int,   default=64,
                        help='Number of batches used for SSIM computation.')

    # Early stopping
    parser.add_argument('--early_stop_patience',   type=int,   default=50)
    parser.add_argument('--min_epochs',            type=int,   default=50)
    parser.add_argument('--reward_std_threshold',  type=float, default=0.002)
    parser.add_argument('--min_entropy_threshold', type=float, default=0.05)
    parser.add_argument('--reward_window_size',    type=int,   default=20)
    parser.add_argument('--acc_baseline',          type=float, default=0.5501)

    # Dataset
    parser.add_argument('--batch_size',  type=int,   default=128)
    parser.add_argument('--num_workers', type=int,   default=15)
    parser.add_argument('--val_split',   type=float, default=0.1)

    # Checkpoint
    parser.add_argument('--save_dir',  type=str, default='./rl_saliency_checkpoints')
    parser.add_argument('--resume',    type=str, default=None)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--seed',      type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, _, test_loader = data_loader_tiny_imagenet(
        data_dir=args.data_dir,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Models ───────────────────────────────────────────────────────────────
    print("Loading models...")
    model_pretrained = model_loader(args.m_name, device)
    load_model_name(model_pretrained, f'./{args.m_name}/checkpoint', args.m_name)
    model_pretrained.eval()

    model_99 = model_loader(args.m_name, device)
    prune_weights_reparam(model_99)


    checkpoint_99 = torch.load(
        f'./{args.m_name}/ckpt_after_prune_oneshot/pruned_oneshot_mask_{args.model_sparsity}.pth')


    model_99.load_state_dict(checkpoint_99)

    # ★ ── SSIM 自动选层（替换原来手填的 target_layers）───────────────────────
    print("\n" + "=" * 70)
    print("SSIM-based Automatic Layer Selection")
    print("=" * 70)
    target_layers, ssim_scores = SSIMLayerSelector.select_layers(
        sparse_model=model_99,
        pretrained_model=model_pretrained,
        data_loader_ref=test_loader,
        threshold=args.ssim_threshold,
        num_batches=args.ssim_num_batches,
    )
    # 记录到 wandb
    for lname, sc in ssim_scores.items():
        run.log({f"ssim/{lname}": sc})
    print("=" * 70 + "\n")

    # ── Layer capacities ──────────────────────────────────────────────────────
    layer_capacities = []
    for lname in target_layers:
        m = dict(model_99.named_modules()).get(lname)
        cap = int((m.weight_mask == 0).sum().item()) if (
            m is not None and hasattr(m, 'weight_mask')) else 0
        layer_capacities.append(cap)

    # ── Regrowth budget ───────────────────────────────────────────────────────
    total_weights, _, _ = count_pruned_params(model_99)
    target_regrow = int(total_weights * args.regrow_step)
    target_regrow = min(target_regrow, sum(layer_capacities))

    print(f"Regrowth configuration:")
    print(f"  Total weights  : {total_weights}")
    print(f"  Target regrowth: {target_regrow}")
    print(f"  Total capacity : {sum(layer_capacities)}")
    print(f"  Init strategy  : {args.init_strategy}")

    # ── Config ────────────────────────────────────────────────────────────────
    config = {
        'num_epochs':           args.num_epochs,
        'batch_size':           args.batch_size,
        'learning_rate':        args.learning_rate,
        'hidden_size':          args.hidden_size,
        'entropy_coef':         args.entropy_coef,
        'action_space_size':    args.action_space_size,
        'target_regrow':        target_regrow,
        'layer_capacities':     layer_capacities,
        'model_name':           args.m_name,
        'reward_temperature':   args.reward_temperature,
        'checkpoint_dir':       args.save_dir,
        'save_freq':            args.save_freq,
        'start_beta':           args.start_beta,
        'end_beta':             args.end_beta,
        'decay_fraction':       args.decay_fraction,
        'init_strategy':        args.init_strategy,
        'saliency_max_batches': args.saliency_max_batches,
        'early_stop_patience':  args.early_stop_patience,
        'min_epochs':           args.min_epochs,
        'reward_std_threshold': args.reward_std_threshold,
        'min_entropy_threshold':args.min_entropy_threshold,
        'reward_window_size':   args.reward_window_size,
        'acc_baseline':         args.acc_baseline,
        'model_sparsity':       args.model_sparsity,
    }

    print(f"\n{'='*70}\ntarget sparsity: {args.model_sparsity}\n{'='*70}")

    pg = RegrowthPolicyGradient(
        config=config,
        model_pretrained=model_pretrained,
        model_99=model_99,
        target_layers=target_layers,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
    )

    before_accuracy       = pg.evaluate_model(model_99, full_eval=True)
    before_sparsity, _, _ = pg.calculate_sparsity(model_99)
    print(f"\nBefore regrowth:")
    print(f"  Accuracy: {before_accuracy:.2f}%")
    print(f"  Sparsity: {before_sparsity:.2f}%")

    best_allocation, best_reward, best_regrow_indices = pg.solve_environment(
        resume_from=args.resume)

    print(f"\n{'='*70}\nFINAL RESULTS\n{'='*70}")
    print(f"Before regrowth   : {before_accuracy:.2f}%")
    print(f"Best RL reward    : {best_reward*100:.2f}%")
    if args.acc_baseline is not None:
        print(f"Baseline threshold: {args.acc_baseline*100:.2f}%")
    print(f"{'='*70}\n")

    run.finish()


if __name__ == '__main__':
    main()