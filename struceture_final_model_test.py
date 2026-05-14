import torch
import torch.nn as nn
from utils.data_loader import data_loader
from utils.model_loader import model_loader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--m_name',     type=str, default='effnet')
parser.add_argument('--model_path', type=str,
                    default='./structured_rl_ckpts/effnet/structured_oneshot/fullfinetune_sp90-86/final_model_0.89.pth',
                    help='finetune 后的 state_dict')
parser.add_argument('--arch_ckpt',  type=str,
                    default='./structured_rl_ckpts/effnet/structured_oneshot/sp90-86/best_model_rwd-3.12pp.pth',
                    help='存有剪枝后完整模型结构的 ckpt（torch.save(model, ...)）')
parser.add_argument('--orig_ckpt',  type=str,
                    default='./effnet/checkpoint/pretrain_effnet_ckpt.pth',
                    help='原始 dense 模型，用于算 channel sparsity')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_, _, test_loader = data_loader(data_dir='./data')

# ── 原始 dense 模型（只用来拿 original_channels）─────────────────────────────
dense_model = model_loader(args.m_name, device)
ckpt = torch.load(args.orig_ckpt, weights_only=False)
dense_model.load_state_dict(ckpt['net'])
dense_model.eval()

original_channels = {name: m.out_channels
                     for name, m in dense_model.named_modules()
                     if isinstance(m, nn.Conv2d)}

# ── 还原剪枝后的模型结构，再 load finetune state_dict ────────────────────────
model = torch.load(args.arch_ckpt, map_location=device, weights_only=False)
state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# ── 准确率 ────────────────────────────────────────────────────────────────────
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        _, pred = model(x).max(1)
        total   += y.size(0)
        correct += pred.eq(y).sum().item()
acc = 100.0 * correct / total

# ── Channel Sparsity ──────────────────────────────────────────────────────────
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