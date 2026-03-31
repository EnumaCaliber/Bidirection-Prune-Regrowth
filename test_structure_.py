"""
eval_structured.py
──────────────────
测试保存的结构化剪枝模型的稀疏度和准确率

用法:
    # 测单个模型
    python eval_structured.py --ckpt ./vgg16/ckpt_structured_iterative/step10_sp0.500.pth

    # 测一个目录下所有 .pth 模型
    python eval_structured.py --dir ./vgg16/ckpt_structured_iterative/

    # 指定模型名（用于加载 dense 计算稀疏度）
    python eval_structured.py --dir ./vgg16/ --m_name vgg16
"""

import torch
import torch.nn as nn
import os
import argparse
import glob

from utils.model_loader import model_loader
from utils.data_loader import data_loader


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_original_channels(dense_model):
    return {name: m.out_channels
            for name, m in dense_model.named_modules()
            if isinstance(m, nn.Conv2d)}


def compute_channel_sparsity(model, original_channels):
    """channel sparsity = 被删通道数 / 原始总通道数"""
    total, remaining = 0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and name in original_channels:
            total     += original_channels[name]
            remaining += m.out_channels
    return (1 - remaining / total) if total > 0 else 0.0


def compute_param_sparsity(model, original_channels):
    """parameter sparsity = 被删参数 / 原始总参数（Conv2d only）"""
    total, remaining = 0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and name in original_channels:
            # 原始参数量（用原始通道数估算）
            orig_out = original_channels[name]
            cur_out  = m.out_channels
            cur_in   = m.in_channels
            kH, kW   = m.kernel_size if isinstance(m.kernel_size, tuple) \
                        else (m.kernel_size, m.kernel_size)
            # 原始 in_channels 无法精确知道（结构化剪枝级联影响），用参数比例估算
            remaining += cur_out * cur_in * kH * kW
            # 原始参数量用当前层 out_ch 占比反推
            total     += orig_out * cur_in * kH * kW
    return (1 - remaining / total) if total > 0 else 0.0


def count_params(model):
    """统计模型总参数量"""
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            _, pred = model(x).max(1)
            total   += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total


def load_model(ckpt_path, device):
    """
    支持两种保存格式：
    1. torch.save(model, path)          → 整个 model 对象（结构化剪枝常用）
    2. torch.save({'net': state_dict})  → state_dict（需配合 model_loader）
    """
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        return obj
    elif isinstance(obj, dict):
        # state_dict 格式，需要外部指定模型
        return obj
    else:
        raise ValueError(f"Unknown checkpoint format: {type(obj)}")


def print_layer_summary(model, original_channels):
    """打印每层通道变化"""
    print(f"\n  {'Layer':<30} {'Dense':>8} {'Pruned':>8} {'Remain%':>8}")
    print(f"  {'-'*58}")
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and name in original_channels:
            orig = original_channels[name]
            cur  = m.out_channels
            pct  = 100.0 * cur / orig
            marker = " ←" if cur < orig else ""
            print(f"  {name:<30} {orig:>8} {cur:>8} {pct:>7.1f}%{marker}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def eval_single(ckpt_path, dense_model, original_channels,
                test_loader, device, verbose=False):
    """评估单个 checkpoint"""
    try:
        model = load_model(ckpt_path, device)
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return None

    if not isinstance(model, nn.Module):
        print(f"  ✗ 不是 nn.Module，跳过")
        return None

    model = model.to(device)
    model.eval()

    ch_sp     = compute_channel_sparsity(model, original_channels)
    total_p, _ = count_params(model)
    acc       = evaluate(model, test_loader, device)

    result = {
        'path':            ckpt_path,
        'acc':             acc,
        'ch_sparsity':     ch_sp,
        'params_M':        total_p / 1e6,
    }

    if verbose:
        print_layer_summary(model, original_channels)

    return result


def main():
    parser = argparse.ArgumentParser(description='Eval structured pruned models')
    parser.add_argument('--ckpt',    type=str,
                        default="./structured_rl_ckpts/effnet/structured_iterative/iter0_sp0.8715/best_model_rwd+2.67pp.pth",
                        help='单个 checkpoint 路径')
    parser.add_argument('--dir',     type=str, default=None,
                        help='目录，自动搜索所有 .pth 文件')
    parser.add_argument('--m_name',  type=str, default='effnet',
                        help='模型名，用于加载 dense 计算稀疏度')
    parser.add_argument('--data_dir',type=str, default='./data')
    parser.add_argument('--verbose', action='store_true',
                        help='打印每层通道详情')
    parser.add_argument('--sort',    type=str, default='sparsity',
                        choices=['sparsity', 'acc', 'name'],
                        help='结果排序方式')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 加载数据
    _, _, test_loader = data_loader(data_dir=args.data_dir)

    # 加载 dense 模型（用于计算 original_channels）
    dense_model = model_loader(args.m_name, device)
    cktp = f'pretrain_{args.m_name}_ckpt.pth'
    checkpoint = torch.load(f'./{args.m_name}/checkpoint/{cktp}',
                            weights_only=False)
    dense_model.load_state_dict(checkpoint['net'])
    dense_model.eval()
    original_channels = get_original_channels(dense_model)

    dense_params, _ = count_params(dense_model)
    dense_acc        = evaluate(dense_model, test_loader, device)
    print(f"\nDense model  | Acc={dense_acc:.2f}%  Params={dense_params/1e6:.2f}M\n")

    # 收集 checkpoint 路径
    ckpt_paths = []
    if args.ckpt:
        ckpt_paths = [args.ckpt]
    elif args.dir:
        ckpt_paths = sorted(glob.glob(os.path.join(args.dir, '**/*.pth'),
                                      recursive=True))
        ckpt_paths += sorted(glob.glob(os.path.join(args.dir, '*.pth')))
        ckpt_paths = sorted(set(ckpt_paths))
    else:
        print("请指定 --ckpt 或 --dir")
        return

    if not ckpt_paths:
        print("没有找到 .pth 文件")
        return

    print(f"找到 {len(ckpt_paths)} 个 checkpoint\n")

    # 逐个评估
    results = []
    for i, path in enumerate(ckpt_paths):
        fname = os.path.relpath(path)
        print(f"[{i+1}/{len(ckpt_paths)}] {fname}")
        r = eval_single(path, dense_model, original_channels,
                        test_loader, device, verbose=args.verbose)
        if r:
            results.append(r)
            print(f"  Acc={r['acc']:.2f}%  "
                  f"ChannelSparsity={r['ch_sparsity']:.4f} ({r['ch_sparsity']*100:.4f}%)  "
                  f"Params={r['params_M']:.2f}M")

    if not results:
        print("没有有效结果")
        return

    # 排序
    if args.sort == 'sparsity':
        results.sort(key=lambda x: x['ch_sparsity'])
    elif args.sort == 'acc':
        results.sort(key=lambda x: x['acc'], reverse=True)
    elif args.sort == 'name':
        results.sort(key=lambda x: x['path'])

    # 汇总表格
    print(f"\n{'='*75}")
    print(f"{'Checkpoint':<40} {'ChSp':>8} {'Acc':>8} {'Params':>8}")
    print(f"{'-'*75}")
    for r in results:
        fname = os.path.basename(r['path'])[:38]
        print(f"{fname:<40} {r['ch_sparsity']*100:>7.1f}% "
              f"{r['acc']:>7.2f}% "
              f"{r['params_M']:>6.2f}M")
    print(f"{'='*75}")
    print(f"{'Dense (baseline)':<40} {'0.0':>8} {dense_acc:>7.2f}% "
          f"{dense_params/1e6:>6.2f}M")

    # 最优结果
    best = max(results, key=lambda x: x['acc'])
    print(f"\n最高精度: {best['acc']:.2f}%  "
          f"ChSparsity={best['ch_sparsity']*100:.1f}%  "
          f"→ {os.path.basename(best['path'])}")


if __name__ == '__main__':
    main()