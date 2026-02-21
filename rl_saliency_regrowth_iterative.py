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
from utils.analysis_utils import (load_model_name, prune_weights_reparam, count_pruned_params)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Saliency: computed ONCE from the dense pretrained model, shared across iters
# ─────────────────────────────────────────────────────────────────────────────
class SaliencyComputer:
    """
    Saliency(θ) = (∂L/∂θ)² · θ²  (FairPrune / OBD approximation)

    Call compute_saliency_scores() once before the iterative loop.
    The returned saliency_dict is then passed into every RegrowthPolicyGradient
    instance unchanged.
    """

    def __init__(self, model, criterion, device='cuda'):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.accumulated_grads = {}
        self.grad_count = 0

    def reset(self):
        self.accumulated_grads = {}
        self.grad_count = 0

    def compute_saliency_scores(self, data_loader, target_layers, num_classes=10):
        self.model.eval()
        self.reset()
        print(f"\nComputing saliency scores (pretrained model, one-time)...")

        module_dict = dict(self.model.named_modules())
        for layer_name in target_layers:
            m = module_dict.get(layer_name)
            if m is not None and hasattr(m, 'weight'):
                self.accumulated_grads[layer_name] = torch.zeros(m.weight.shape, device=self.device)

        for inputs, labels in tqdm(data_loader, desc="  Gradient accumulation"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            loss = self.criterion(self.model(inputs), labels)
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)

            for param, grad in zip(self.model.parameters(), grads):
                if grad is None:
                    continue
                for name, p in self.model.named_parameters():
                    if p is param:
                        for layer_name in target_layers:
                            if name == f"{layer_name}.weight":
                                self.accumulated_grads[layer_name] += (
                                        grad.pow(2).detach() * param.data.pow(2).detach())
                                break
                        break
            self.grad_count += 1

        saliency_dict = {}
        for layer_name in target_layers:
            if layer_name in self.accumulated_grads:
                sal = self.accumulated_grads[layer_name] / max(self.grad_count, 1)
                saliency_dict[layer_name] = sal.cpu()
                print(f"  {layer_name}: mean={sal.mean():.3e}  max={sal.max():.3e}")

        print("Saliency computation done.\n")
        return saliency_dict


# ─────────────────────────────────────────────────────────────────────────────
# LSTM controller
# ─────────────────────────────────────────────────────────────────────────────
class RegrowthAgent(nn.Module):
    def __init__(self, action_dim, hidden_size, context_dim, device='cuda'):
        super().__init__()
        self.DEVICE = device
        self.nhid = hidden_size
        self.action_dim = action_dim
        self.lstm = nn.LSTMCell(action_dim + context_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, action_dim)
        self.hidden = self.init_hidden()

    def forward(self, prev_logits, context_vec):
        if prev_logits.dim() == 1: prev_logits = prev_logits.unsqueeze(0)
        if context_vec.dim() == 1: context_vec = context_vec.unsqueeze(0)
        h_t, c_t = self.lstm(torch.cat([prev_logits, context_vec], dim=-1), self.hidden)
        self.hidden = (h_t, c_t)
        return self.decoder(h_t)

    def init_hidden(self):
        return (torch.zeros(1, self.nhid, device=self.DEVICE),
                torch.zeros(1, self.nhid, device=self.DEVICE))


# ─────────────────────────────────────────────────────────────────────────────
# Saliency-based regrowth (top-K selection on pruned positions)
# ─────────────────────────────────────────────────────────────────────────────
class SaliencyBasedRegrowth:
    @staticmethod
    @torch.no_grad()
    def apply_regrowth(model, layer_name, saliency_tensor, num_weights,
                       init_strategy='zero', device='cuda'):
        module = dict(model.named_modules()).get(layer_name)
        if module is None or not hasattr(module, 'weight_mask'):
            return 0, []

        mask = module.weight_mask
        sal = saliency_tensor.to(device)

        pruned = (mask == 0)
        if not pruned.any():
            return 0, []

        sal_masked = sal.clone()
        sal_masked[~pruned] = -float('inf')
        flat = sal_masked.flatten()
        k = min(num_weights, (flat > -float('inf')).sum().item())
        if k == 0:
            return 0, []

        _, top_k = torch.topk(flat, k=k)
        shape = sal.shape
        regrown = []

        for fi in top_k:
            idx = np.unravel_index(fi.cpu().item(), shape)
            regrown.append(idx)
            mask[idx] = 1.0

            wp = module.weight_orig if hasattr(module, 'weight_orig') else module.weight
            if init_strategy == 'zero':
                wp.data[idx] = 0.0
            elif init_strategy == 'kaiming':
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(wp)
                bound = np.sqrt(6.0 / fan_in)
                wp.data[idx] = torch.empty(1).uniform_(-bound, bound).item()
            elif init_strategy == 'xavier':
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(wp)
                bound = np.sqrt(6.0 / (fan_in + fan_out))
                wp.data[idx] = torch.empty(1).uniform_(-bound, bound).item()
            # 'magnitude': keep existing value

        return len(regrown), regrown


# ─────────────────────────────────────────────────────────────────────────────
# RL Policy Gradient (one instance per iteration)
# ─────────────────────────────────────────────────────────────────────────────
class RegrowthPolicyGradient:
    """
    saliency_dict is passed in from outside (computed once from pretrained model)
    and never recomputed here.
    """

    def __init__(self, config, model_sparse, saliency_dict,
                 target_layers, train_loader, test_loader, device, wandb_run=None):

        self.NUM_EPOCHS = config['num_epochs']
        self.ALPHA = config['learning_rate']
        self.HIDDEN_SIZE = config['hidden_size']
        self.BETA = config['entropy_coef']
        self.REWARD_TEMPERATURE = config.get('reward_temperature', 0.01)
        self.DEVICE = device
        self.ACTION_SPACE = config['action_space_size']
        self.NUM_STEPS = len(target_layers)
        self.CONTEXT_DIM = config.get('context_dim', 3)
        self.BASELINE_DECAY = config.get('baseline_decay', 0.9)

        self.model_sparse = model_sparse.to(device)
        self.target_layers = target_layers
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.target_regrow = config['target_regrow']
        self.layer_capacities = config['layer_capacities']
        self.total_capacity = max(sum(self.layer_capacities), 1)
        self.init_strategy = config.get('init_strategy', 'zero')

        self.early_stop_patience = config.get('early_stop_patience', 40)
        self.min_epochs = config.get('min_epochs', 50)
        self.reward_std_threshold = config.get('reward_std_threshold', 0.002)
        self.reward_window_size = config.get('reward_window_size', 20)

        self._best_model_state = None
        self._best_reward_seen = float('-inf')

        self.run = wandb_run
        self.model_name = config.get('model_name')
        self.method = config.get('method', 'iterative')
        self.checkpoint_dir = config.get('checkpoint_dir', './rl_saliency_checkpoints')
        self.model_sparsity = config.get('model_sparsity', '0.98')

        self.use_entropy_schedule = config.get('use_entropy_schedule', True)
        self.start_beta = config.get('start_beta', 0.4)
        self.end_beta = config.get('end_beta', 0.004)
        self.decay_fraction = config.get('decay_fraction', 0.4)

        # ── Pre-computed saliency (fixed, from pretrained) ────────────────────
        self.saliency_dict = saliency_dict
        print(f"  Saliency dict loaded for {len(self.saliency_dict)} layers (pretrained, no recompute).")

        self.agent = RegrowthAgent(
            action_dim=self.ACTION_SPACE,
            hidden_size=self.HIDDEN_SIZE,
            context_dim=self.CONTEXT_DIM,
            device=self.DEVICE,
        ).to(self.DEVICE)

        self.adam = optim.Adam(self.agent.parameters(), lr=self.ALPHA)
        self.reward_baseline = None
        self.layer_priority = [(name, idx) for idx, name in enumerate(target_layers)]

        print(f"  Layer order & capacities:")
        for name, idx in self.layer_priority:
            print(f"    {idx + 1}. {name}: cap={self.layer_capacities[idx]}")

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
                total += m.weight_mask.numel()
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
                total += y.size(0)
                correct += pred.eq(y).sum().item()
        return 100.0 * correct / total

    def mini_finetune(self, model, epochs=50, lr=0.0003):
        model.train()
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
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
                    total += y.size(0)
                    correct += pred.eq(y).sum().item()
            acc = 100.0 * correct / total
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(model.state_dict())
            model.train()

        if best_state:
            model.load_state_dict(best_state)
        model.eval()

    def calculate_loss(self, logits, weighted_log_probs, beta):
        policy_loss = -torch.mean(weighted_log_probs)
        p = F.softmax(logits, dim=1)
        ent = -torch.mean(torch.sum(p * F.log_softmax(logits, dim=1), dim=1))
        return policy_loss - beta * ent, ent

    # ── Main RL loop ─────────────────────────────────────────────────────────
    def solve_environment(self, resume_from=None):
        t0 = time.time()
        best_reward, best_allocation, best_regrow = float('-inf'), None, None
        best_reward_epoch, start_epoch = 0, 0
        reward_window = deque(maxlen=self.reward_window_size)
        stop_reason = ""

        if resume_from and os.path.exists(resume_from):
            ckpt = torch.load(resume_from)
            self.agent.load_state_dict(ckpt['agent_state_dict'])
            self.adam.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_reward = ckpt['best_reward']
            best_allocation = ckpt['best_allocation']
            best_regrow = ckpt['best_regrow_indices']
            if 'reward_baseline' in ckpt:
                self.reward_baseline = ckpt['reward_baseline']
            print(f"Resumed from epoch {start_epoch}, best={best_reward:.4f}")

        for epoch in range(start_epoch, self.NUM_EPOCHS):
            ep_wlp, ep_logits, reward, allocation, sparsity, regrow_indices = \
                self.play_episode(t0, epoch)

            reward_window.append(reward)

            if reward > best_reward:
                best_reward, best_reward_epoch = reward, epoch
                best_allocation = allocation
                best_regrow = copy.deepcopy(regrow_indices)
                self._save_best_allocation(epoch, best_reward, best_allocation, best_regrow)

            beta = self.get_entropy_coef(epoch)
            loss, ent = self.calculate_loss(ep_logits, ep_wlp, beta)
            self.adam.zero_grad()
            loss.backward()
            self.adam.step()

            no_imp = epoch - best_reward_epoch
            rwd_std = float(np.std(list(reward_window))) if len(reward_window) > 1 else float('inf')

            pfx = f"iter_{self.model_sparsity}"
            if self.run:
                self.run.log({
                    f"{pfx}/epoch": epoch + 1,
                    f"{pfx}/loss": loss.item(),
                    f"{pfx}/entropy": ent.item(),
                    f"{pfx}/beta": beta,
                    f"{pfx}/reward": reward,
                    f"{pfx}/acc": reward * 100.0,
                    f"{pfx}/sparsity": sparsity,
                    f"{pfx}/no_improve": no_imp,
                    f"{pfx}/rwd_std": rwd_std,
                })

            print(f"[{self.model_sparsity}] Ep {epoch + 1:3d}/{self.NUM_EPOCHS} | "
                  f"Rwd={reward:.4f} ({reward * 100:.2f}%) | "
                  f"Best={best_reward:.4f} | "
                  f"Loss={loss.item():.4f} | Ent={ent.item():.4f} | "
                  f"NoImp={no_imp} | Std={rwd_std:.5f}")

            # Early stop
            if no_imp >= self.early_stop_patience:
                stop_reason = f"No improvement for {self.early_stop_patience} epochs"
                print(f"\n>>> Early stop: {stop_reason}")
                break
            if (epoch >= self.min_epochs and
                    len(reward_window) >= self.reward_window_size and
                    rwd_std < self.reward_std_threshold):
                stop_reason = f"Reward std {rwd_std:.5f} < threshold"
                print(f"\n>>> Early stop: {stop_reason}")
                break

        print(f"\nBest: {best_reward:.4f} ({best_reward * 100:.2f}%)"
              + (f"  [{stop_reason}]" if stop_reason else ""))
        return best_allocation, best_reward, best_regrow

    # ── Episode ───────────────────────────────────────────────────────────────
    def play_episode(self, solve_start, epoch):
        t0 = time.time()
        self.agent.hidden = self.agent.init_hidden()

        prev_logits = torch.zeros(1, self.ACTION_SPACE, device=self.DEVICE)
        ratio_opts = torch.arange(self.ACTION_SPACE, device=self.DEVICE, dtype=torch.float) / (self.ACTION_SPACE - 1)

        total_budget = int(self.target_regrow)
        remaining = total_budget
        log_probs, masked_logits_list, selected_counts, priority_names = [], [], [], []

        for p_idx, (layer_name, orig_idx) in enumerate(self.layer_priority):
            cap = int(self.layer_capacities[orig_idx])
            ctx = torch.tensor([
                p_idx / max(self.NUM_STEPS - 1, 1) if self.NUM_STEPS > 1 else 0.0,
                cap / self.total_capacity if self.total_capacity > 0 else 0.0,
                remaining / total_budget if total_budget > 0 else 0.0,
            ], dtype=torch.float, device=self.DEVICE).unsqueeze(0)

            logits = self.agent(prev_logits, ctx).squeeze(0)

            eff_max = min(cap, remaining)
            count_opts = torch.round(ratio_opts * eff_max).to(torch.long)
            feasible = count_opts <= remaining
            if not feasible.any(): feasible[0] = True

            masked = torch.where(feasible, logits, torch.full_like(logits, -1e9))
            dist = Categorical(probs=F.softmax(masked, dim=0))
            action = dist.sample()

            chosen = int(count_opts[action].item())
            chosen = min(chosen, cap, remaining)
            remaining = max(remaining - chosen, 0)

            selected_counts.append(chosen)
            priority_names.append(layer_name)
            log_probs.append(dist.log_prob(action))
            masked_logits_list.append(masked)
            prev_logits = logits.unsqueeze(0)

        ep_log_probs = torch.stack(log_probs)
        allocation = {n: int(c) for n, c in zip(priority_names, selected_counts) if c > 0}
        total_alloc = sum(selected_counts)
        print(f"  Allocated: {total_alloc}/{total_budget} | {allocation}")

        # Apply regrowth (saliency from pretrained, fixed)
        model_copy = self._create_model_copy(self.model_sparse)
        regrow_indices = {}
        for layer_name, num_w in allocation.items():
            sal = self.saliency_dict.get(layer_name)
            if sal is not None:
                _, indices = SaliencyBasedRegrowth.apply_regrowth(
                    model=model_copy, layer_name=layer_name,
                    saliency_tensor=sal, num_weights=num_w,
                    init_strategy=self.init_strategy, device=self.DEVICE)
                regrow_indices[layer_name] = indices

        self.mini_finetune(model_copy, epochs=40)
        accuracy = self.evaluate_model(model_copy, full_eval=True)
        sparsity, _, _ = self.calculate_sparsity(model_copy)
        reward = accuracy / 100.0

        if reward > self._best_reward_seen:
            self._best_reward_seen = reward
            self._best_model_state = copy.deepcopy(model_copy.state_dict())
            self._save_best_model(epoch, reward, model_copy, allocation)

        if self.reward_baseline is None:
            self.reward_baseline = reward
        adv = float(np.clip(
            (reward - self.reward_baseline) / max(self.REWARD_TEMPERATURE, 1e-6),
            -100.0, 100.0))
        self.reward_baseline = (self.BASELINE_DECAY * self.reward_baseline +
                                (1 - self.BASELINE_DECAY) * reward)

        adv_t = torch.tensor(adv, device=self.DEVICE, dtype=torch.float)
        ep_wlp = torch.sum(ep_log_probs * adv_t).unsqueeze(0)
        ep_logits_stacked = torch.stack(masked_logits_list)

        print(f"  [episode {time.time() - t0:.1f}s]")
        return ep_wlp, ep_logits_stacked, reward, allocation, sparsity, regrow_indices

    def _save_best_model(self, epoch, reward, model, allocation):
        d = os.path.join(self.checkpoint_dir, f'{self.model_name}/{self.method}/{self.model_sparsity}')
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f'best_model_epoch{epoch + 1}_acc{reward * 100:.2f}.pth')
        torch.save({
            'epoch': epoch,
            'accuracy': reward * 100,
            'model_state_dict': model.state_dict(),
            'allocation': allocation,
            'timestamp': time.time(),
        }, p)
        print(f"  New best {reward * 100:.2f}% → saved: {p}")
        if self.run:
            self.run.log({"best_acc": reward * 100, "best_epoch": epoch + 1})

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_best_allocation(self, epoch, reward, allocation, regrow_indices):
        d = os.path.join(self.checkpoint_dir, f'{self.model_name}/{self.method}/{self.model_sparsity}')
        os.makedirs(d, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'reward': reward,
            'allocation': allocation,
            'regrow_indices': regrow_indices,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.adam.state_dict(),
            'reward_baseline': self.reward_baseline,
            'timestamp': time.time(),
        }, os.path.join(d, 'best_allocation.pth'))
        print(f"  New best {reward * 100:.2f}% @ epoch {epoch + 1}")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
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
            total += y.size(0);
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total


def get_sparsity(model):
    total, pruned = 0, 0
    for _, m in model.named_modules():
        if hasattr(m, 'weight_mask'):
            total += m.weight_mask.numel()
            pruned += (m.weight_mask == 0).sum().item()
    return 100.0 * pruned / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_name', type=str, default='vgg16')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--method', type=str, default="iterative")

    # Sparsity schedule
    parser.add_argument('--start_sparsity', type=float, default=0.995)
    parser.add_argument('--target_sparsity', type=float, default=0.99)
    parser.add_argument('--regrow_step', type=float, default=0.0025,
                        help='Fraction of total weights to regrow per iteration')
    # RL
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--entropy_coef', type=float, default=0.5)
    parser.add_argument('--reward_temperature', type=float, default=0.005)
    parser.add_argument('--start_beta', type=float, default=0.40)
    parser.add_argument('--end_beta', type=float, default=0.04)
    parser.add_argument('--decay_fraction', type=float, default=0.4)
    parser.add_argument('--action_space_size', type=int, default=11)
    # Regrowth
    parser.add_argument('--init_strategy', type=str, default='zero',
                        choices=['zero', 'kaiming', 'xavier', 'magnitude'])
    # Early stopping
    parser.add_argument('--early_stop_patience', type=int, default=40)
    parser.add_argument('--min_epochs', type=int, default=40)
    parser.add_argument('--reward_std_threshold', type=float, default=0.002)
    parser.add_argument('--reward_window_size', type=int, default=20)
    # Misc
    parser.add_argument('--save_dir', type=str, default='./rl_saliency_checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume_iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    train_loader, _, test_loader = data_loader(data_dir=args.data_dir)

    # Architecture config
    if args.m_name == 'resnet20':
        initial_ckpt = (f'./{args.m_name}/ckpt_after_prune_0.3_epoch_finetune_40/'
                        'pruned_finetuned_mask_0.9953.pth')
        target_layers = ["layer3.0.conv2", "layer3.1.conv1",
                         "layer3.1.conv2", "layer3.2.conv1"]
    elif args.m_name == 'vgg16':
        initial_ckpt = (f'./{args.m_name}/ckpt_after_prune_0.3_epoch_finetune_40/'
                        'pruned_finetuned_mask_0.9953.pth')
        target_layers = ["features.10", "features.20", "features.24"]
    elif args.m_name == 'alexnet':
        initial_ckpt = (f'./{args.m_name}/ckpt_after_prune_0.3_epoch_finetune_40/'
                        f'pruned_finetuned_mask_0.9953.pth')
        target_layers = ['features.3', 'features.6', 'features.8',
                         'features.10', 'classifier.1']
    elif args.m_name == 'densenet':
        initial_ckpt = (f'./{args.m_name}/ckpt_after_prune_0.3_epoch_finetune_40/'
                        f'pruned_finetuned_mask_0.9953.pth')
        target_layers = ['features.3', 'features.6', 'features.8',
                         'features.10', 'classifier.1']
    else:
        raise ValueError(f"Unknown model: {args.m_name}")

    # ── Load pretrained dense model ───────────────────────────────────────────
    print("Loading pretrained (dense) model...")
    model_pretrained = model_loader(args.m_name, device)
    load_model_name(model_pretrained, f'./{args.m_name}/checkpoint', args.m_name)
    model_pretrained.eval()

    # ── Compute saliency ONCE ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ONE-TIME saliency computation from pretrained model")
    print("=" * 70)
    saliency_dict = SaliencyComputer(
        model=model_pretrained,
        criterion=nn.CrossEntropyLoss(),
        device=device,
    ).compute_saliency_scores(
        data_loader=train_loader,
        target_layers=target_layers,
    )
    print("=" * 70 + "\n")

    # ── Determine total weights & iteration count ─────────────────────────────
    _ref = model_loader(args.m_name, device)
    prune_weights_reparam(_ref)
    _ref.load_state_dict(torch.load(initial_ckpt))
    total_weights, _, _ = count_pruned_params(_ref)
    del _ref

    regrow_per_iter = int(total_weights * args.regrow_step)
    # TODO
    num_iters = max(1, math.ceil((args.start_sparsity - args.target_sparsity) / args.regrow_step))

    print(f"\n{'=' * 70}")
    print(f"Iterative Regrowth Plan")
    print(f"  {args.start_sparsity * 100:.1f}% → {args.target_sparsity * 100:.1f}%  "
          f"({num_iters} iterations × {regrow_per_iter} weights/iter)")
    print(f"{'=' * 70}\n")

    # ── wandb (one run, all iterations) ──────────────────────────────────────
    run = wandb.init(
        project="ICCAD_saliency_iterative",
        name=f"{args.m_name}_{args.start_sparsity:.3f}_to_{args.target_sparsity:.3f}",
        config=vars(args) | {"num_iters": num_iters},
    )

    # ── Load starting sparse model ────────────────────────────────────────────
    current_model = model_loader(args.m_name, device)
    prune_weights_reparam(current_model)

    if args.resume_iter > 0:
        ckpt_path = os.path.join(args.save_dir,
                                 f'{args.m_name}/{args.method}/iter_{args.resume_iter - 1}/best_grown_model.pth')
        assert os.path.exists(ckpt_path), f"Resume checkpoint not found: {ckpt_path}"
        current_model.load_state_dict(torch.load(ckpt_path))
        print(f"Resumed from iter {args.resume_iter}: {ckpt_path}")
    else:
        current_model.load_state_dict(torch.load(initial_ckpt))

    acc0 = quick_eval(current_model, test_loader, device)
    sp0 = get_sparsity(current_model)
    print(f"Starting → Acc: {acc0:.2f}%  Sparsity: {sp0:.2f}%\n")
    run.log({"iter_summary/iteration": 0,
             "iter_summary/accuracy": acc0,
             "iter_summary/sparsity": sp0})

    # ── Iterative loop ────────────────────────────────────────────────────────
    for iter_idx in range(args.resume_iter, num_iters):
        cur_sp = get_sparsity(current_model)
        print(f"\n{'#' * 70}")
        print(f"  ITERATION {iter_idx + 1}/{num_iters}  |  "
              f"Sparsity: {cur_sp:.2f}% → ~{cur_sp - args.regrow_step * 100:.2f}%")
        print(f"{'#' * 70}\n")

        layer_capacities = get_layer_capacities(current_model, target_layers)
        # TODO
        remaining = int(total_weights * (cur_sp / 100 - args.target_sparsity))
        target_regrow = min(regrow_per_iter, remaining, sum(layer_capacities))

        if target_regrow == 0:
            print("  All pruned weights already restored. Stopping.")
            break

        sp_label = f"iter{iter_idx}_sp{(cur_sp / 100 - args.regrow_step):.4f}"

        config = {
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'hidden_size': args.hidden_size,
            'entropy_coef': args.entropy_coef,
            'action_space_size': args.action_space_size,
            'target_regrow': target_regrow,
            'layer_capacities': layer_capacities,
            'model_name': args.m_name,
            'reward_temperature': args.reward_temperature,
            'checkpoint_dir': args.save_dir,
            'start_beta': args.start_beta,
            'end_beta': args.end_beta,
            'decay_fraction': args.decay_fraction,
            'init_strategy': args.init_strategy,
            'early_stop_patience': args.early_stop_patience,
            'min_epochs': args.min_epochs,
            'reward_std_threshold': args.reward_std_threshold,
            'reward_window_size': args.reward_window_size,
            'model_sparsity': sp_label,
            'method': args.method,
        }

        pg = RegrowthPolicyGradient(
            config=config,
            model_sparse=current_model,  # this iteration's starting model
            saliency_dict=saliency_dict,  # ← fixed, pretrained, no recompute
            target_layers=target_layers,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            wandb_run=run,
        )

        resume_path = args.resume if (iter_idx == args.resume_iter and args.resume) else None
        best_allocation, best_reward, _ = pg.solve_environment(resume_from=resume_path)

        best_model = pg.get_best_model() or current_model
        iter_acc = quick_eval(best_model, test_loader, device)
        iter_sp = get_sparsity(best_model)

        print(f"\n  [Iter {iter_idx + 1}] → Acc: {iter_acc:.2f}%  Sparsity: {iter_sp:.2f}%")
        run.log({"iter_summary/iteration": iter_idx + 1,
                 "iter_summary/best_reward": best_reward,
                 "iter_summary/accuracy": iter_acc,
                 "iter_summary/sparsity": iter_sp})

        # Save best model for this iteration
        save_dir = os.path.join(args.save_dir, f'{args.m_name}/{args.method}/iter_{iter_idx}')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'best_grown_model.pth')
        torch.save(best_model.state_dict(), save_path)
        print(f"  ✓ Saved → {save_path}")

        # Next iteration starts from this model
        current_model = best_model

        if iter_sp <= args.target_sparsity * 100 + 0.1:
            print(f"  Target sparsity {args.target_sparsity * 100:.1f}% reached. Done.");
            break

    # ── Final ─────────────────────────────────────────────────────────────────
    final_acc = quick_eval(current_model, test_loader, device)
    final_sp = get_sparsity(current_model)
    print(f"\n{'=' * 70}")
    print(f"DONE  |  Acc: {final_acc:.2f}%  Sparsity: {final_sp:.2f}%")
    print(f"{'=' * 70}")

    run.log({"final/accuracy": final_acc, "final/sparsity": final_sp})
    run.finish()


if __name__ == '__main__':
    main()
