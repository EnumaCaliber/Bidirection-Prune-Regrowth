"""
Baseline 2: Greedy SSIM allocation + Saliency weight selection
- Budget allocation : sort layers by SSIM ascending, fill greedily (lowest SSIM first)
- Weight selection  : top-k highest saliency among pruned (mask=0) weights
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import copy
import os
import re
import random
import wandb
from pathlib import Path
from tqdm import tqdm

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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_baseline_from_folder(model_name, model_dir, device, test_loader):
    pth_files = sorted(Path(model_dir).glob("*.pth"))
    if not pth_files:
        raise ValueError(f"No .pth files in {model_dir}")
    print("=" * 60)
    baseline_dict = {}
    for f in pth_files:
        m = re.search(r'(\d+\.\d+)', f.name)
        if not m:
            continue
        sparsity = float(m.group(1))
        try:
            sd = torch.load(f, map_location=device)
            merged = {}
            for k, v in sd.items():
                if k.endswith('_orig'):
                    base = k[:-5]
                    merged[base] = v * sd[base + '_mask']
                elif not k.endswith('_mask'):
                    merged[k] = v
            model = model_loader(model_name, device)
            model.load_state_dict(merged)
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    _, pred = model(x).max(1)
                    total += y.size(0)
                    correct += pred.eq(y).sum().item()
            acc = 100.0 * correct / total
            baseline_dict[sparsity] = acc
            print(f"  {f.name}: sp={sparsity:.4f} acc={acc:.2f}%")
        except Exception as e:
            print(f"  Error {f.name}: {e}")
    baseline_dict = dict(sorted(baseline_dict.items()))
    if not baseline_dict:
        raise ValueError(f"No valid checkpoints in {model_dir}")
    print(f"Baseline built: {len(baseline_dict)} pts")
    print("=" * 60 + "\n")
    return baseline_dict


class BaselineInterpolator:
    def __init__(self, table):
        self.points = {float(k): float(v) for k, v in table.items()}

    def get_baseline_acc(self, sparsity):
        pts = sorted(self.points.items())
        if sparsity <= pts[0][0]: return pts[0][1]
        if sparsity >= pts[-1][0]: return pts[-1][1]
        for i in range(len(pts) - 1):
            s1, a1 = pts[i]; s2, a2 = pts[i + 1]
            if s1 <= sparsity <= s2:
                return a1 + (sparsity - s1) / (s2 - s1 + 1e-12) * (a2 - a1)
        return pts[-1][1]


def get_ssim_scores(sparse_model, pretrained_model, loader, threshold, num_batches):
    all_masked = [n for n, m in sparse_model.named_modules()
                  if hasattr(m, 'weight_mask') and n]
    block_dict = {'all': all_masked}
    ext_pre  = BlockwiseFeatureExtractor(pretrained_model, block_dict)
    ext_spar = BlockwiseFeatureExtractor(sparse_model,     block_dict)
    with torch.no_grad():
        fp = ext_pre.extract_block_features(loader,  num_batches=num_batches)
        fs = ext_spar.extract_block_features(loader, num_batches=num_batches)
    block_ssim = compute_block_ssim(fp, fs).get('all', {})
    ssim_dict  = {l: float(block_ssim.get(l, 0.5)) for l in all_masked}
    selected   = [l for l in all_masked if ssim_dict[l] < threshold]
    if not selected:
        worst    = min(ssim_dict, key=ssim_dict.get)
        selected = [worst]
        print(f"  Fallback to worst SSIM layer: {worst} ({ssim_dict[worst]:.4f})")
    print(f"  SSIM search space: {len(selected)}/{len(all_masked)} layers")
    return selected, ssim_dict


def compute_saliency(model, train_loader, target_layers, device):
    model.eval()
    module_dict = dict(model.named_modules())
    accumulated = {l: torch.zeros_like(module_dict[l].weight, device=device)
                   for l in target_layers
                   if l in module_dict and hasattr(module_dict[l], 'weight')}
    crit  = nn.CrossEntropyLoss()
    count = 0
    for x, y in tqdm(train_loader, desc="Saliency"):
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        crit(model(x), y).backward()
        for l in target_layers:
            m = module_dict.get(l)
            if m is not None and m.weight.grad is not None:
                accumulated[l] += m.weight.grad.pow(2) * m.weight.data.pow(2)
        count += 1
    model.zero_grad()
    return {l: (v / max(count, 1)).cpu() for l, v in accumulated.items()}


# ── Greedy allocation ─────────────────────────────────────────────────────────
def allocate_greedy(target_layers, ssim_dict, layer_capacities, total_budget):
    order = sorted(range(len(target_layers)),
                   key=lambda i: ssim_dict.get(target_layers[i], 0.5))
    allocation, remaining = {}, total_budget
    for i in order:
        if remaining <= 0:
            break
        give = min(layer_capacities[i], remaining)
        if give > 0:
            allocation[target_layers[i]] = give
            remaining -= give
    return allocation


# ── Saliency weight selection ─────────────────────────────────────────────────
@torch.no_grad()
def apply_saliency_regrowth(model, layer_name, saliency, num_weights, device):
    module = dict(model.named_modules()).get(layer_name)
    if module is None or not hasattr(module, 'weight_mask'):
        return 0
    mask   = module.weight_mask
    sal    = saliency.to(device)
    pruned = (mask == 0)
    if not pruned.any():
        return 0
    sal_m         = sal.clone()
    sal_m[~pruned] = -float('inf')
    k             = min(num_weights, (sal_m.flatten() > -float('inf')).sum().item())
    if k == 0:
        return 0
    _, top_k = torch.topk(sal_m.flatten(), k=k)
    wp = getattr(module, 'weight_orig', module.weight)
    for fi in top_k:
        idx = tuple(np.unravel_index(fi.cpu().item(), sal.shape))
        mask[idx] = 1.0
        wp.data[idx] = 0.0
    return k


# ── Shared utilities ──────────────────────────────────────────────────────────
def mini_finetune(model, train_loader, test_loader, device, epochs, lr=3e-4):
    model.train()
    opt  = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, None
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        acc = quick_eval(model, test_loader, device)
        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(model.state_dict())
        model.train()
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return best_acc


def quick_eval(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            _, pred = model(x).max(1)
            total += y.size(0); correct += pred.eq(y).sum().item()
    return 100.0 * correct / total


def get_sparsity(model):
    total = pruned = 0
    for _, m in model.named_modules():
        if hasattr(m, 'weight_mask'):
            total  += m.weight_mask.numel()
            pruned += (m.weight_mask == 0).sum().item()
    return 100.0 * pruned / total if total > 0 else 0.0


def get_layer_capacities(model, target_layers):
    return [int((dict(model.named_modules())[n].weight_mask == 0).sum().item())
            if hasattr(dict(model.named_modules()).get(n, object()), 'weight_mask') else 0
            for n in target_layers]


def create_copy(src, model_name, device):
    m = model_loader(model_name, device)
    prune_weights_reparam(m)
    m.load_state_dict(src.state_dict())
    return m


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_name',          type=str,   default='vgg16')
    parser.add_argument('--data_dir',        type=str,   default='./data')
    parser.add_argument('--baseline_dir',    type=str,   default='./vgg16/ckpt_after_prune_0.3_epoch_finetune_40')
    parser.add_argument('--initial_ckpt',    type=str,   default='./vgg16/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.9953.pth')
    parser.add_argument('--target_sparsity', type=float, default=0.97)
    parser.add_argument('--num_iters',       type=int,   default=5)
    parser.add_argument('--budget_frac',     type=float, default=0.003)
    parser.add_argument('--finetune_epochs', type=int,   default=40)
    parser.add_argument('--ssim_threshold',  type=float, default=0.3)
    parser.add_argument('--ssim_num_batches',type=int,   default=128)
    parser.add_argument('--save_dir',        type=str,   default='./baselines/greedy_saliency')
    parser.add_argument('--seed',            type=int,   default=42)
    parser.add_argument('--no_wandb',        action='store_true', default=False)
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[greedy_saliency] Device={device}  seed={args.seed}")

    train_loader, _, test_loader = data_loader(data_dir=args.data_dir)
    baseline_interp = BaselineInterpolator(
        load_baseline_from_folder(args.m_name, args.baseline_dir, device, test_loader))

    _ref = model_loader(args.m_name, device)
    prune_weights_reparam(_ref)
    _ref.load_state_dict(torch.load(args.initial_ckpt))
    total_weights, _, _ = count_pruned_params(_ref)
    del _ref

    run = None if args.no_wandb else wandb.init(
        project="regrowth_baselines",
        name=f"greedy_saliency_{args.m_name}_s{args.seed}",
        config=vars(args))

    print("Loading pretrained model...")
    model_pretrained = model_loader(args.m_name, device)
    load_model_name(model_pretrained, f'./{args.m_name}/checkpoint', args.m_name)
    model_pretrained.eval()

    print("Computing saliency (one-time, pretrained model)...")
    _init = model_loader(args.m_name, device)
    prune_weights_reparam(_init)
    _init.load_state_dict(torch.load(args.initial_ckpt))
    all_masked = [n for n, m in _init.named_modules() if hasattr(m, 'weight_mask') and n]
    del _init
    saliency_dict = compute_saliency(model_pretrained, train_loader, all_masked, device)
    print(f"Saliency computed for {len(saliency_dict)} layers.\n")

    current_model = model_loader(args.m_name, device)
    prune_weights_reparam(current_model)
    current_model.load_state_dict(torch.load(args.initial_ckpt))
    print(f"Start → Acc: {quick_eval(current_model, test_loader, device):.2f}%  "
          f"Sp: {get_sparsity(current_model):.2f}%\n")

    for it in range(args.num_iters):
        cur_sp  = get_sparsity(current_model)
        cur_acc = quick_eval(current_model, test_loader, device)
        bline   = baseline_interp.get_baseline_acc(cur_sp / 100.0)
        print(f"\n{'#'*60}\n  ITER {it+1}/{args.num_iters}  sp={cur_sp:.2f}%  "
              f"acc={cur_acc:.2f}%  baseline={bline:.2f}%\n{'#'*60}")

        target_layers, ssim_dict = get_ssim_scores(
            current_model, model_pretrained, test_loader,
            args.ssim_threshold, args.ssim_num_batches)
        layer_caps = get_layer_capacities(current_model, target_layers)
        if sum(layer_caps) == 0:
            print("No pruned weights remain."); break

        total_budget = min(int(total_weights * args.budget_frac), sum(layer_caps))
        allocation   = allocate_greedy(target_layers, ssim_dict, layer_caps, total_budget)
        print(f"  Budget={total_budget}  Allocation={allocation}")

        model_copy = create_copy(current_model, args.m_name, device)
        regrown = sum(
            apply_saliency_regrowth(model_copy, l, saliency_dict[l], n, device)
            for l, n in allocation.items() if l in saliency_dict)

        mini_finetune(model_copy, train_loader, test_loader, device, args.finetune_epochs)
        iter_acc = quick_eval(model_copy, test_loader, device)
        iter_sp  = get_sparsity(model_copy)
        delta    = iter_acc - baseline_interp.get_baseline_acc(iter_sp / 100.0)
        print(f"  → acc={iter_acc:.2f}%  sp={iter_sp:.2f}%  Δbaseline={delta:+.2f}pp  regrown={regrown}")

        save_dir = os.path.join(args.save_dir, args.m_name, f'iter_{it}')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model_copy.state_dict(), os.path.join(save_dir, 'model.pth'))

        if run:
            run.log({"iter": it+1, "accuracy": iter_acc, "sparsity": iter_sp,
                     "delta_baseline_pp": delta, "regrown": regrown})

        current_model = model_copy
        if iter_sp <= args.target_sparsity * 100 + 0.1:
            print("  Target sparsity reached."); break

    final_acc = quick_eval(current_model, test_loader, device)
    final_sp  = get_sparsity(current_model)
    print(f"\nDONE | Acc={final_acc:.2f}%  Sp={final_sp:.2f}%  "
          f"Δbaseline={final_acc - baseline_interp.get_baseline_acc(final_sp/100):.2f}pp")
    if run:
        run.log({"final/accuracy": final_acc, "final/sparsity": final_sp})
        run.finish()


if __name__ == '__main__':
    main()
