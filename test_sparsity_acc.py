"""
极简版单文件测试 - 稀疏度 + 准确率
"""
import torch
import torch.nn.functional as F
from utils.model_loader import model_loader
from utils.data_loader import data_loader
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--m_name', type=str, default='vgg16')
parser.add_argument('--model_path', type=str, default='./vgg16/ckpt_after_prune_oneshot/pruned_oneshot_mask_0.995.pth')  # 改为 model_path
args = parser.parse_args()


def test(model, loader):
    model.eval()
    device = next(model.parameters()).device
    correct, total_loss, total = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            total_loss += F.cross_entropy(yhat, y).item()
            _, pred = yhat.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

    model.train()
    return 100. * correct / total, total_loss / len(loader)


def load_model(ckpt_path, model_name, device):
    sd = torch.load(ckpt_path, map_location=device)
    merged, done = {}, set()

    for k in sd:
        if k in done: continue
        if k.endswith("_orig"):
            base = k[:-5]
            merged[base] = sd[k] * sd[base + "_mask"]
            done.update([k, base + "_mask"])
        elif not k.endswith("_mask"):
            merged[k] = sd[k]

    model = model_loader(model_name, device)
    model.load_state_dict(merged)
    return model


def get_sparsity(ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu')
    total = sum(v.numel() for k, v in sd.items() if k.endswith('_mask'))
    zeros = sum((v == 0).sum().item() for k, v in sd.items() if k.endswith('_mask'))
    return zeros / total * 100 if total > 0 else 0


# Main
device = 'cuda' if torch.cuda.is_available() else 'cpu'
_, _, test_loader = data_loader(data_dir='./data')

# 单文件测试
ckpt_path = Path(args.model_path)
sparsity = get_sparsity(ckpt_path)
model = load_model(ckpt_path, args.m_name, device)
acc, loss = test(model, test_loader)

print(f"\n{'='*60}")
print(f"File:       {ckpt_path.name}")
print(f"Sparsity:   {sparsity:.2f}%")
print(f"Accuracy:   {acc:.2f}%")
print(f"Loss:       {loss:.4f}")
print(f"{'='*60}\n")