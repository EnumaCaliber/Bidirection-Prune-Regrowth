import torch
import torch.nn as nn
from utils.model_loader import model_loader
from utils.data_loader import data_loader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--m_name',     type=str, default='effnet')
parser.add_argument('--model_path', type=str, required=True,
                    help='_save_best 保存的 best_epXX_rwdXX.pth')
parser.add_argument('--orig_ckpt',  type=str,
                    default='./effnet/checkpoint/pretrain_effnet_ckpt.pth')
parser.add_argument('--data_dir',   type=str, default='./data')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_, _, test_loader = data_loader(data_dir=args.data_dir)

# ── 原始 dense 通道数（用于算稀疏度）─────────────────────────────────────────
dense_model = model_loader(args.m_name, device)
ckpt = torch.load(args.orig_ckpt, map_location=device, weights_only=False)
dense_model.load_state_dict(ckpt['net'])
original_channels = {n: m.out_channels for n, m in dense_model.named_modules()
                     if isinstance(m, nn.Conv2d)}

# ── 直接加载完整模型（_save_best 用 torch.save(model, p) 存的）──────────────
model = torch.load(args.model_path, map_location=device, weights_only=False)
model = model.to(device)
model.eval()

# ── 精度 ──────────────────────────────────────────────────────────────────────
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        _, pred = model(x).max(1)
        total   += y.size(0)
        correct += pred.eq(y).sum().item()
acc = 100.0 * correct / total

# ── 稀疏度 ────────────────────────────────────────────────────────────────────
total_ch, remaining_ch = 0, 0
for name, m in model.named_modules():
    if isinstance(m, nn.Conv2d) and name in original_channels:
        total_ch     += original_channels[name]
        remaining_ch += m.out_channels
channel_sparsity = 1 - remaining_ch / total_ch

# ── 输出 ──────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  Accuracy        : {acc:.2f}%")
print(f"  Channel Sparsity: {channel_sparsity:.4f} ({channel_sparsity*100:.2f}%)")
print(f"  Remaining ch    : {remaining_ch} / {total_ch}")
print(f"{'=' * 60}")
print(f"\n  Per-layer channel details:")
for name, m in model.named_modules():
    if isinstance(m, nn.Conv2d) and name in original_channels:
        orig = original_channels[name]
        curr = m.out_channels
        sp   = 1 - curr / orig
        print(f"    {name:30s}: {orig} → {curr}  (pruned {orig-curr:3d}, sp={sp:.2f})")