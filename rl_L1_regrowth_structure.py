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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# 通道级 Saliency（L1 norm，替换原来的梯度×权重²）
# ─────────────────────────────────────────────────────────────────────────────
class ChannelSaliencyComputer:
    """
    结构化剪枝的 saliency = 通道 L1 norm
    从 dense 模型计算一次，后续不再重算
    返回 {layer_name: [ch0_importance, ch1_importance, ...]}
    """
    def __init__(self, dense_model, device='cuda'):
        self.model  = dense_model
        self.device = device

    def compute_saliency_scores(self, target_layers):
        print("\nComputing channel saliency (L1 norm from dense model)...")
        saliency_dict = {}

        module_dict = dict(self.model.named_modules())
        for layer_name in target_layers:
            m = module_dict.get(layer_name)
            if m is None or not isinstance(m, nn.Conv2d):
                continue
            # L1 norm per output channel: [C_out]
            importance = m.weight.data.abs().sum(dim=[1, 2, 3])  # [C_out]
            saliency_dict[layer_name] = importance.cpu()
            print(f"  {layer_name}: {m.out_channels} ch, "
                  f"mean={importance.mean():.3e}  max={importance.max():.3e}")

        print("Saliency done.\n")
        return saliency_dict


# ─────────────────────────────────────────────────────────────────────────────
# 通道级 Regrowth（替换原来的 weight mask 操作）
# ─────────────────────────────────────────────────────────────────────────────
class ChannelRegrowth:
    """
    从 dense 模型按 config 重建子网
    saliency 决定被剪掉的通道里哪些最值得加回来
    """

    @staticmethod
    def get_pruned_channel_indices(dense_model, pruned_model):
        """
        对比 dense 和 pruned 权重，找出每层被剪掉的通道索引
        返回 {layer_name: [pruned_ch_idx, ...]}
        """
        pruned_indices = {}
        for name, p_mod in pruned_model.named_modules():
            if not isinstance(p_mod, nn.Conv2d):
                continue
            d_mod = dict(dense_model.named_modules()).get(name)
            if d_mod is None:
                continue

            d_w = d_mod.weight.data   # [C_dense, ...]
            p_w = p_mod.weight.data   # [C_pruned, ...]

            # 匹配保留了哪些通道
            kept = []
            for pw in p_w:
                diffs = [(pw - d_w[j]).abs().sum().item()
                         for j in range(d_w.shape[0])]
                kept.append(int(torch.tensor(diffs).argmin()))

            pruned_indices[name] = [i for i in range(d_mod.out_channels)
                                    if i not in kept]
        return pruned_indices

    @staticmethod
    def select_channels_by_saliency(saliency, pruned_ch_indices, n_restore):
        """
        从被剪掉的通道里，按 saliency 选 top-n 个加回来
        saliency: [C_dense] 全通道重要性
        pruned_ch_indices: 被剪掉的通道索引列表
        返回: 要加回来的通道索引列表
        """
        if len(pruned_ch_indices) == 0 or n_restore == 0:
            return []
        n = min(n_restore, len(pruned_ch_indices))
        sal_candidates = saliency[pruned_ch_indices]   # [n_pruned]
        topk = sal_candidates.topk(n).indices          # top-n 在 candidates 中的位置
        return [pruned_ch_indices[i] for i in topk.tolist()]

    @staticmethod
    def apply_config(dense_model, config, original_channels, example_inputs):
        """
        按 config 从 dense 重建结构
        config: {layer_name: target_ch}
        """
        model = copy.deepcopy(dense_model)
        ch_sparsity_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and name in config:
                sparsity = 1 - config[name] / original_channels[name]
                if sparsity > 0:
                    ch_sparsity_dict[module] = sparsity

        if ch_sparsity_dict:
            pruner = tp.pruner.MagnitudePruner(
                model, example_inputs,
                importance=tp.importance.MagnitudeImportance(p=1),
                iterative_steps=1,
                ch_sparsity=0,
                ch_sparsity_dict=ch_sparsity_dict,
            )
            pruner.step()
        return model


# ─────────────────────────────────────────────────────────────────────────────
# LSTM controller（不变）
# ─────────────────────────────────────────────────────────────────────────────
class RegrowthAgent(nn.Module):
    def __init__(self, action_dim, hidden_size, context_dim, device='cuda'):
        super().__init__()
        self.DEVICE  = device
        self.nhid    = hidden_size
        self.action_dim = action_dim
        self.lstm    = nn.LSTMCell(action_dim + context_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, action_dim)
        self.hidden  = self.init_hidden()

    def forward(self, prev_logits, context_vec):
        if prev_logits.dim()  == 1: prev_logits  = prev_logits.unsqueeze(0)
        if context_vec.dim()  == 1: context_vec  = context_vec.unsqueeze(0)
        h_t, c_t = self.lstm(
            torch.cat([prev_logits, context_vec], dim=-1), self.hidden)
        self.hidden = (h_t, c_t)
        return self.decoder(h_t)

    def init_hidden(self):
        return (torch.zeros(1, self.nhid, device=self.DEVICE),
                torch.zeros(1, self.nhid, device=self.DEVICE))


# ─────────────────────────────────────────────────────────────────────────────
# RL Policy Gradient（结构化版本）
# ─────────────────────────────────────────────────────────────────────────────
class StructuredRegrowthPolicyGradient:

    def __init__(self, config, model_sparse, dense_model,
                 saliency_dict, pruned_ch_indices,
                 original_channels, example_inputs,
                 target_layers, train_loader, test_loader,
                 device, wandb_run=None):

        self.NUM_EPOCHS        = config['num_epochs']
        self.ALPHA             = config['learning_rate']
        self.HIDDEN_SIZE       = config['hidden_size']
        self.BETA              = config['entropy_coef']
        self.REWARD_TEMPERATURE= config.get('reward_temperature', 0.01)
        self.DEVICE            = device
        self.ACTION_SPACE      = config['action_space_size']
        self.NUM_STEPS         = len(target_layers)
        self.CONTEXT_DIM       = config.get('context_dim', 3)
        self.BASELINE_DECAY    = config.get('baseline_decay', 0.9)

        self.model_sparse      = model_sparse
        self.dense_model       = dense_model
        self.target_layers     = target_layers
        self.train_loader      = train_loader
        self.test_loader       = test_loader

        self.target_regrow     = config['target_regrow']    # 本轮要加回多少通道
        self.layer_capacities  = config['layer_capacities'] # 每层可加回的通道数
        self.total_capacity    = max(sum(self.layer_capacities), 1)

        self.saliency_dict     = saliency_dict
        self.pruned_ch_indices = pruned_ch_indices          # 每层被剪掉的通道索引
        self.original_channels = original_channels
        self.example_inputs    = example_inputs

        # 当前模型的 config（每层通道数）
        self.current_config = {
            name: m.out_channels
            for name, m in model_sparse.named_modules()
            if isinstance(m, nn.Conv2d)
        }

        self.early_stop_patience   = config.get('early_stop_patience', 40)
        self.min_epochs            = config.get('min_epochs', 50)
        self.reward_std_threshold  = config.get('reward_std_threshold', 0.002)
        self.reward_window_size    = config.get('reward_window_size', 20)

        self._best_model        = None
        self._best_reward_seen  = float('-inf')

        self.run           = wandb_run
        self.model_name    = config.get('model_name')
        self.method        = config.get('method', 'structured_iterative')
        self.checkpoint_dir= config.get('checkpoint_dir', './structured_rl_ckpts')
        self.model_sparsity= config.get('model_sparsity', '0.5')

        self.use_entropy_schedule = config.get('use_entropy_schedule', True)
        self.start_beta    = config.get('start_beta', 0.4)
        self.end_beta      = config.get('end_beta', 0.004)
        self.decay_fraction= config.get('decay_fraction', 0.4)

        self.agent = RegrowthAgent(
            action_dim=self.ACTION_SPACE,
            hidden_size=self.HIDDEN_SIZE,
            context_dim=self.CONTEXT_DIM,
            device=self.DEVICE,
        ).to(self.DEVICE)

        self.adam             = optim.Adam(self.agent.parameters(), lr=self.ALPHA)
        self.reward_baseline  = None
        self.layer_priority   = [(name, idx) for idx, name in enumerate(target_layers)]

        print(f"  Layer capacities (pruned channels available to restore):")
        for name, idx in self.layer_priority:
            print(f"    {name}: cap={self.layer_capacities[idx]}")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def get_entropy_coef(self, epoch):
        if not self.use_entropy_schedule:
            return self.BETA
        de = self.NUM_EPOCHS * self.decay_fraction
        if epoch < de:
            return self.start_beta - (self.start_beta - self.end_beta) * (epoch / de)
        return self.end_beta

    def compute_channel_sparsity(self, model):
        total, remaining = 0, 0
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and name in self.original_channels:
                total     += self.original_channels[name]
                remaining += m.out_channels
        return 1 - remaining / total if total > 0 else 0

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

    def mini_finetune(self, model, epochs=5, lr=0.0003):
        """结构化剪枝 finetune：epoch 数少，因为只加了几个通道"""
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

    def calculate_loss(self, logits, weighted_log_probs, beta):
        policy_loss = -torch.mean(weighted_log_probs)
        p   = F.softmax(logits, dim=1)
        ent = -torch.mean(torch.sum(p * F.log_softmax(logits, dim=1), dim=1))
        return policy_loss - beta * ent, ent

    # ── Main RL loop ─────────────────────────────────────────────────────────
    def solve_environment(self):
        best_reward, best_config = float('-inf'), None
        best_reward_epoch = 0
        reward_window = deque(maxlen=self.reward_window_size)
        stop_reason = ""

        for epoch in range(self.NUM_EPOCHS):
            ep_wlp, ep_logits, reward, new_config, sparsity = self.play_episode(epoch)

            reward_window.append(reward)

            if reward > best_reward:
                best_reward       = reward
                best_reward_epoch = epoch
                best_config       = new_config
                self._save_best(epoch, reward, new_config)

            beta = self.get_entropy_coef(epoch)
            loss, ent = self.calculate_loss(ep_logits, ep_wlp, beta)
            self.adam.zero_grad()
            loss.backward()
            self.adam.step()

            no_imp  = epoch - best_reward_epoch
            rwd_std = float(np.std(list(reward_window))) if len(reward_window) > 1 else float('inf')

            pfx = f"iter_{self.model_sparsity}"
            if self.run:
                self.run.log({
                    f"{pfx}/epoch":      epoch + 1,
                    f"{pfx}/loss":       loss.item(),
                    f"{pfx}/entropy":    ent.item(),
                    f"{pfx}/reward":     reward,
                    f"{pfx}/acc":        reward * 100.0,
                    f"{pfx}/sparsity":   sparsity,
                    f"{pfx}/no_improve": no_imp,
                })

            print(f"[{self.model_sparsity}] Ep {epoch+1:3d}/{self.NUM_EPOCHS} | "
                  f"Rwd={reward:.4f} ({reward*100:.2f}%) | "
                  f"Best={best_reward:.4f} | Loss={loss.item():.4f} | "
                  f"NoImp={no_imp} | Std={rwd_std:.5f}")

            if no_imp >= self.early_stop_patience:
                stop_reason = f"No improvement for {self.early_stop_patience} epochs"
                break
            if (epoch >= self.min_epochs and
                    len(reward_window) >= self.reward_window_size and
                    rwd_std < self.reward_std_threshold):
                stop_reason = f"Reward std < threshold"
                break

        print(f"\nBest: {best_reward*100:.2f}%"
              + (f"  [{stop_reason}]" if stop_reason else ""))
        return best_config, best_reward

    # ── Episode ───────────────────────────────────────────────────────────────
    def play_episode(self, epoch):
        self.agent.hidden = self.agent.init_hidden()

        prev_logits = torch.zeros(1, self.ACTION_SPACE, device=self.DEVICE)
        ratio_opts  = torch.arange(
            self.ACTION_SPACE, device=self.DEVICE, dtype=torch.float
        ) / (self.ACTION_SPACE - 1)

        total_budget   = int(self.target_regrow)
        remaining      = total_budget
        log_probs, masked_logits_list = [], []
        selected_counts, priority_names = [], []

        for p_idx, (layer_name, orig_idx) in enumerate(self.layer_priority):
            cap = int(self.layer_capacities[orig_idx])  # 该层可加回的通道数
            ctx = torch.tensor([
                p_idx / max(self.NUM_STEPS - 1, 1),
                cap   / self.total_capacity if self.total_capacity > 0 else 0.0,
                remaining / total_budget    if total_budget > 0 else 0.0,
            ], dtype=torch.float, device=self.DEVICE).unsqueeze(0)

            logits   = self.agent(prev_logits, ctx).squeeze(0)
            eff_max  = min(cap, remaining)
            count_opts = torch.round(ratio_opts * eff_max).to(torch.long)
            feasible = count_opts <= remaining
            if not feasible.any(): feasible[0] = True

            masked = torch.where(feasible, logits, torch.full_like(logits, -1e9))
            dist   = Categorical(probs=F.softmax(masked, dim=0))
            action = dist.sample()

            chosen    = int(count_opts[action].item())
            chosen    = min(chosen, cap, remaining)
            remaining = max(remaining - chosen, 0)

            selected_counts.append(chosen)
            priority_names.append(layer_name)
            log_probs.append(dist.log_prob(action))
            masked_logits_list.append(masked)
            prev_logits = logits.unsqueeze(0)

        ep_log_probs = torch.stack(log_probs)

        # ── 关键改动：用通道 allocation 重建模型 ─────────────────────────────
        allocation = {n: int(c) for n, c in zip(priority_names, selected_counts) if c > 0}
        print(f"  Allocated channels: {allocation}")

        # 构建新 config：在当前 config 基础上，按 saliency 加回 allocation 指定的通道数
        new_config = copy.copy(self.current_config)
        for layer_name, n_restore in allocation.items():
            sal = self.saliency_dict.get(layer_name)
            if sal is None:
                continue
            # L1 saliency 选通道
            channels_to_add = ChannelRegrowth.select_channels_by_saliency(
                saliency=sal,
                pruned_ch_indices=self.pruned_ch_indices.get(layer_name, []),
                n_restore=n_restore,
            )
            new_config[layer_name] = self.current_config[layer_name] + len(channels_to_add)

        # apply_config 从 dense 重建
        new_model = ChannelRegrowth.apply_config(
            self.dense_model, new_config,
            self.original_channels, self.example_inputs
        )
        new_model = new_model.to(self.DEVICE)

        # finetune
        self.mini_finetune(new_model, epochs=5)
        accuracy  = self.evaluate_model(new_model, full_eval=True)
        sparsity  = self.compute_channel_sparsity(new_model)
        reward    = accuracy / 100.0

        if reward > self._best_reward_seen:
            self._best_reward_seen = reward
            self._best_model       = copy.deepcopy(new_model)

        # advantage
        if self.reward_baseline is None:
            self.reward_baseline = reward
        adv = float(np.clip(
            (reward - self.reward_baseline) / max(self.REWARD_TEMPERATURE, 1e-6),
            -100.0, 100.0))
        self.reward_baseline = (self.BASELINE_DECAY * self.reward_baseline +
                                (1 - self.BASELINE_DECAY) * reward)

        adv_t  = torch.tensor(adv, device=self.DEVICE, dtype=torch.float)
        ep_wlp = torch.sum(ep_log_probs * adv_t).unsqueeze(0)
        ep_logits_stacked = torch.stack(masked_logits_list)

        return ep_wlp, ep_logits_stacked, reward, new_config, sparsity

    def _save_best(self, epoch, reward, config):
        d = os.path.join(self.checkpoint_dir,
                         f'{self.model_name}/{self.method}/{self.model_sparsity}')
        os.makedirs(d, exist_ok=True)
        torch.save({
            'epoch':   epoch,
            'accuracy': reward * 100,
            'config':   config,        # 结构化保存 config，不保存 state_dict
            'timestamp': time.time(),
        }, os.path.join(d, 'best_allocation.pth'))
        # 同时保存整个 model（结构化必须存整个 model）
        if self._best_model is not None:
            torch.save(self._best_model,
                       os.path.join(d, f'best_model_acc{reward*100:.2f}.pth'))
        print(f"  New best {reward*100:.2f}% saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def get_layer_capacities_structured(dense_model, pruned_model, target_layers):
    """每层还有多少通道可以加回来"""
    caps = []
    d_modules = dict(dense_model.named_modules())
    p_modules = dict(pruned_model.named_modules())
    for name in target_layers:
        d_m = d_modules.get(name)
        p_m = p_modules.get(name)
        if d_m and p_m and isinstance(d_m, nn.Conv2d):
            caps.append(d_m.out_channels - p_m.out_channels)
        else:
            caps.append(0)
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_name',       type=str,   default='vgg16')
    parser.add_argument('--data_dir',     type=str,   default='./data')
    parser.add_argument('--start_sp',     type=float, default=0.7,
                        help='起始 channel sparsity（最稀疏模型）')
    parser.add_argument('--target_sp',    type=float, default=0.3,
                        help='目标 channel sparsity（恢复后）')
    parser.add_argument('--restore_step', type=int,   default=4,
                        help='每轮总共加回多少通道')
    parser.add_argument('--num_iters',    type=int,   default=20)
    # RL
    parser.add_argument('--num_epochs',       type=int,   default=200)
    parser.add_argument('--learning_rate',    type=float, default=3e-4)
    parser.add_argument('--hidden_size',      type=int,   default=64)
    parser.add_argument('--entropy_coef',     type=float, default=0.5)
    parser.add_argument('--reward_temperature', type=float, default=0.005)
    parser.add_argument('--start_beta',       type=float, default=0.40)
    parser.add_argument('--end_beta',         type=float, default=0.04)
    parser.add_argument('--decay_fraction',   type=float, default=0.4)
    parser.add_argument('--action_space_size',type=int,   default=11)
    parser.add_argument('--early_stop_patience', type=int, default=40)
    parser.add_argument('--min_epochs',       type=int,   default=50)
    parser.add_argument('--reward_std_threshold', type=float, default=0.002)
    parser.add_argument('--reward_window_size',   type=int,   default=20)
    parser.add_argument('--save_dir',  type=str, default='./structured_rl_ckpts')
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, _, test_loader = data_loader(data_dir=args.data_dir)

    # example_inputs 用真实数据
    example_inputs, _ = next(iter(train_loader))
    example_inputs = example_inputs[:1].to(device)

    # target layers（Conv2d 层名）
    if args.m_name == 'vgg16':
        initial_ckpt = (f'./{args.m_name}/ckpt_after_prune_structured_oneshot/'
                        f'pruned_structured_l1_sp{args.start_sp:.4f}_it1.pth')
        target_layers = [
            'features.0',  'features.2',  'features.5',  'features.7',
            'features.10', 'features.12', 'features.14', 'features.17',
            'features.19', 'features.21', 'features.24', 'features.26',
            'features.28',
        ]
    elif args.m_name == 'resnet20':
        initial_ckpt = (f'./{args.m_name}/ckpt_after_prune_structured_oneshot/'
                        f'pruned_structured_l1_sp{args.start_sp:.4f}_it1.pth')
        target_layers = [
            'layer1.0.conv1', 'layer1.0.conv2',
            'layer2.0.conv1', 'layer2.0.conv2',
            'layer3.0.conv1', 'layer3.0.conv2',
        ]
    else:
        raise ValueError(f"Unknown model: {args.m_name}")

    # ── 加载 dense model（原始预训练）
    dense_model = model_loader(args.m_name, device)
    cktp = f'pretrain_{args.m_name}_ckpt.pth'
    checkpoint = torch.load(f'./{args.m_name}/checkpoint/{cktp}')
    dense_model.load_state_dict(checkpoint['net'])
    dense_model.eval()

    # original channels
    original_channels = {
        name: m.out_channels
        for name, m in dense_model.named_modules()
        if isinstance(m, nn.Conv2d)
    }

    # ── 计算 channel saliency（一次，从 dense）
    saliency_dict = ChannelSaliencyComputer(
        dense_model=dense_model, device=device
    ).compute_saliency_scores(target_layers)

    # ── 加载起始稀疏模型
    # ⚠️ 结构化剪枝保存的是整个 model，不是 state_dict
    current_model = torch.load(initial_ckpt, map_location=device)
    current_model.eval()

    acc0 = quick_eval(current_model, test_loader, device)
    print(f"起点模型精度: {acc0:.2f}%\n")

    run = wandb.init(
        project="structured_rl_regrowth",
        name=f"{args.m_name}_sp{args.start_sp:.2f}_to_{args.target_sp:.2f}",
        config=vars(args),
    )
    run.log({"iter_summary/iteration": 0, "iter_summary/accuracy": acc0})

    # ── 迭代 regrowth 主循环
    for iter_idx in range(args.num_iters):
        # 计算 pruned_ch_indices（每轮更新，因为模型在变）
        pruned_ch_indices = ChannelRegrowth.get_pruned_channel_indices(
            dense_model, current_model)

        # 计算每层 capacity（还能加回多少通道）
        layer_capacities = get_layer_capacities_structured(
            dense_model, current_model, target_layers)

        total_cap = sum(layer_capacities)
        if total_cap == 0:
            print("所有通道已恢复，停止。")
            break

        target_regrow = min(args.restore_step, total_cap)

        sp_label = f"iter{iter_idx}"
        config = {
            'num_epochs':          args.num_epochs,
            'learning_rate':       args.learning_rate,
            'hidden_size':         args.hidden_size,
            'entropy_coef':        args.entropy_coef,
            'action_space_size':   args.action_space_size,
            'target_regrow':       target_regrow,
            'layer_capacities':    layer_capacities,
            'model_name':          args.m_name,
            'reward_temperature':  args.reward_temperature,
            'checkpoint_dir':      args.save_dir,
            'start_beta':          args.start_beta,
            'end_beta':            args.end_beta,
            'decay_fraction':      args.decay_fraction,
            'early_stop_patience': args.early_stop_patience,
            'min_epochs':          args.min_epochs,
            'reward_std_threshold':args.reward_std_threshold,
            'reward_window_size':  args.reward_window_size,
            'model_sparsity':      sp_label,
            'method':              'structured_iterative',
        }

        print(f"\n{'#'*60}")
        print(f"  ITER {iter_idx+1}/{args.num_iters} | "
              f"target_regrow={target_regrow} channels | "
              f"total_cap={total_cap}")
        print(f"{'#'*60}")

        pg = StructuredRegrowthPolicyGradient(
            config=config,
            model_sparse=current_model,
            dense_model=dense_model,
            saliency_dict=saliency_dict,
            pruned_ch_indices=pruned_ch_indices,
            original_channels=original_channels,
            example_inputs=example_inputs,
            target_layers=target_layers,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            wandb_run=run,
        )

        best_config, best_reward = pg.solve_environment()

        # 用 best_config 重建最终模型
        best_model = pg._best_model or current_model
        iter_acc   = quick_eval(best_model, test_loader, device)
        print(f"\n  [Iter {iter_idx+1}] → Acc: {iter_acc:.2f}%")
        run.log({"iter_summary/iteration": iter_idx+1,
                 "iter_summary/accuracy":  iter_acc})

        # 保存（整个 model）
        save_dir = os.path.join(args.save_dir,
                                f'{args.m_name}/structured_iterative/iter_{iter_idx}')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_model, os.path.join(save_dir, 'best_model.pth'))

        current_model = best_model

    final_acc = quick_eval(current_model, test_loader, device)
    print(f"\n{'='*60}")
    print(f"DONE | Final Acc: {final_acc:.2f}%")
    run.finish()


if __name__ == '__main__':
    main()