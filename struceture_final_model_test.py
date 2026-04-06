import torch
import torch.nn as nn
from utils.data_loader import data_loader
from utils.model_loader import model_loader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--m_name',     type=str, default='vgg16')
parser.add_argument('--model_path', type=str,
                    default='./structured_rl_ckpts/vgg16/structured_oneshot/fullfinetune_sp0.952/final_model_0.952.pth')
parser.add_argument('--orig_ckpt',  type=str,
                    default='./vgg16/checkpoint/pretrain_vgg16_ckpt.pth',
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


# ── 加载 finetune 后的模型 ────────────────────────────────────────────────────
# fullfinetune 存的是 state_dict，需要先还原结构
# 结构从 RL ckpt 里拿（整模型），再 load state_dict
rl_model_path = './structured_rl_ckpts/vgg16/structured_oneshot/sp0.952-094/best_model_rwd+7.87pp.pth'
model = torch.load(rl_model_path, map_location=device, weights_only=False)
model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
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

# ── 每层详情 ──────────────────────────────────────────────────────────────────
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