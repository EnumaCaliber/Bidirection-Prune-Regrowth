'''Evaluate Top-1 and Top-5 accuracy from a saved checkpoint.'''
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from utils.model_loader import model_loader
from utils.data_loader_tiny_imagenet import data_loader_tiny_imagenet
import random
import numpy as np

parser = argparse.ArgumentParser(description='Evaluate Top-1 & Top-5 Accuracy')
parser.add_argument('--m_name',      type=str, default="shufflenetv2TinyImageNet")
parser.add_argument('--ckpt_path',   type=str, default="./shufflenetv2TinyImageNet/ckpt_after_prune_0.3_epoch_finetune_40/pruned_finetuned_mask_0.9953.pth")
parser.add_argument('--data_dir',    type=str, default='./data')
parser.add_argument('--batch_size',  type=int, default=128)
parser.add_argument('--num_workers', type=int, default=15)
parser.add_argument('--val_split',   type=float, default=0.1)
parser.add_argument('--seed',        type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── 稀疏度计算（从 _mask 键统计）─────────────────────────────────────────────
def get_sparsity(sd):
    total = sum(v.numel() for k, v in sd.items() if k.endswith('_mask'))
    zeros = sum((v == 0).sum().item() for k, v in sd.items() if k.endswith('_mask'))
    return zeros / total * 100 if total > 0 else 0.0


# ── 模型加载（合并 _orig * _mask）────────────────────────────────────────────
def load_model(ckpt_path, model_name, device):
    sd = torch.load(ckpt_path, map_location=device)

    merged, done = {}, set()
    for k in sd:
        if k in done:
            continue
        if k.endswith('_orig'):
            base = k[:-5]                               # e.g. "features.0.weight"
            merged[base] = sd[k] * sd[base + '_mask']  # 权重 * mask → 真实稀疏权重
            done.update([k, base + '_mask'])
        elif not k.endswith('_mask'):
            merged[k] = sd[k]

    model = model_loader(model_name, device)
    model.load_state_dict(merged)
    return model, sd


# ── 加载数据 ──────────────────────────────────────────────────────────────────
print('==> Loading data..')
_, _, test_loader = data_loader_tiny_imagenet(
    data_dir=args.data_dir,
    val_split=args.val_split,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)

# ── 加载模型 ──────────────────────────────────────────────────────────────────
print(f'==> Loading model {args.m_name} from {args.ckpt_path}')
net, sd = load_model(args.ckpt_path, args.m_name, device)
net.eval()
sparsity = get_sparsity(sd)

# ── 评估 ──────────────────────────────────────────────────────────────────────
correct_top1 = 0
correct_top5 = 0
total_loss   = 0
total        = 0

with torch.no_grad():
    for i, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        total += targets.size(0)

        total_loss += F.cross_entropy(outputs, targets).item()

        # Top-1
        _, predicted = outputs.max(1)
        correct_top1 += predicted.eq(targets).sum().item()

        # Top-5
        _, top5_pred = outputs.topk(5, dim=1)
        correct_top5 += top5_pred.eq(targets.view(-1, 1)).any(dim=1).sum().item()

        if (i + 1) % 20 == 0:
            print(f'  [{i+1}/{len(test_loader)}] '
                  f'Top-1: {100.*correct_top1/total:.2f}% | '
                  f'Top-5: {100.*correct_top5/total:.2f}%')

top1 = 100. * correct_top1 / total
top5 = 100. * correct_top5 / total
loss = total_loss / len(test_loader)

print(f'\n{"="*50}')
print(f'File      : {Path(args.ckpt_path).name}')
print(f'Model     : {args.m_name}')
print(f'Sparsity  : {sparsity:.2f}%')
print(f'Top-1     : {top1:.2f}%')
print(f'Top-5     : {top5:.2f}%')
print(f'Loss      : {loss:.4f}')
print(f'{"="*50}')