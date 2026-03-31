import torch
import argparse
import numpy as np
import random
from utils.model_loader import model_loader
from utils.data_loader import data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--m_name', type=str, default='resnet20')
parser.add_argument('--model_path', type=str, default='./resnet20/checkpoint/pretrain_resnet20_ckpt.pth')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_dir', type=str, default='./data')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = torch.nn.CrossEntropyLoss()
_, _, test_loader = data_loader(data_dir=args.data_dir)

# ── test 函数 ─────────────────────────────────────────────────────────────────
def test(net):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    return acc

# ── 加载模型 ──────────────────────────────────────────────────────────────────
model = model_loader(args.m_name, device)
sd = torch.load(args.model_path, map_location=device)['net']
model.load_state_dict(sd)

# ── 测试 ──────────────────────────────────────────────────────────────────────
acc = test(model)

print(f"\n{'=' * 50}")
print(f"File      : {args.model_path}")
print(f"Accuracy  : {acc:.2f}%")
print(f"{'=' * 50}\n")