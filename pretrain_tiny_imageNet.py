'''Train Tiny ImageNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim

import os
import argparse
import wandb

from utils.utils import progress_bar
from utils.model_loader import model_loader

# ← 直接使用你已有的 dataloader
from utils.data_loader_tiny_imagenet import data_loader_tiny_imagenet


parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--m_name', type=str, default="vgg16inyImageNet",
                    help='Model name (e.g., resnet18, vgg16, etc.)')
parser.add_argument('--pruner', type=str, help='pruning method')
parser.add_argument('--iter_start', type=int, default=1)
parser.add_argument('--iter_end',   type=int, default=1)
parser.add_argument('--seed',       type=int, default=42)
parser.add_argument('--max_epochs', type=int, default=1000)
parser.add_argument('--patience',   type=int, default=30)
parser.add_argument('--data_dir',   type=str, default='./data')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--val_split',  type=float, default=0.1)
parser.add_argument('--num_workers',type=int, default=15)

args = parser.parse_args()

# ── Reproducibility ────────────────────────────────────────────────────────────
import random
import numpy as np
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Random seed: {args.seed}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0
epochs_without_improvement = 0

# ── WandB ──────────────────────────────────────────────────────────────────────
run = wandb.init(
    project="tiny-imagenet-pretrain",
    config=args,
    name=f"{args.m_name}_pretrain",
)

# ── Data（直接调用你的 dataloader）────────────────────────────────────────────
print('==> Preparing Tiny ImageNet data..')
train_loader, val_loader, test_loader = data_loader_tiny_imagenet(
    data_dir=args.data_dir,
    val_split=args.val_split,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)

# ── Model（200 classes）────────────────────────────────────────────────────────
print('==> Building model..')
net = model_loader(args.m_name, device)

if args.resume:
    print('==> Resuming from checkpoint..')
    ckpt_path = os.path.join(args.m_name, 'checkpoint', f"pretrain_{args.m_name}_ckpt.pth")
    assert os.path.isfile(ckpt_path), f'Checkpoint not found: {ckpt_path}'
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint.get('acc', 0)
    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f'Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%')

# ── Loss / Optimizer / Scheduler ───────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# ── Train ──────────────────────────────────────────────────────────────────────
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = train_loss / len(train_loader)
    acc = 100. * correct / total
    print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}%', end=' | ')
    run.log({"train loss": avg_loss, "train acc": acc, "epoch": epoch})


# ── Evaluate ───────────────────────────────────────────────────────────────────
def evaluate(epoch):
    global best_acc, epochs_without_improvement
    net.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = val_loss / len(val_loader)
    acc = 100. * correct / total
    print(f'Val Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%', end='')

    run.log({"val loss": avg_loss, "val acc": acc,
             "best val acc": best_acc, "epoch": epoch})

    if acc > best_acc:
        print(f' ✓ NEW BEST (prev: {best_acc:.2f}%)')
        state = {'net': net.state_dict(), 'acc': acc, 'epoch': epoch}
        folder = os.path.join(args.m_name, 'checkpoint')
        os.makedirs(folder, exist_ok=True)
        torch.save(state, os.path.join(folder, f"pretrain_{args.m_name}_ckpt.pth"))
        best_acc = acc
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement <= 5 or epochs_without_improvement % 10 == 0:
            print(f' (no improvement: {epochs_without_improvement})')
        else:
            print()

    return epochs_without_improvement >= args.patience


# ── Test ───────────────────────────────────────────────────────────────────────
def test():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total


# ── Main Loop ──────────────────────────────────────────────────────────────────
print(f"\nPretraining Tiny ImageNet — max {args.max_epochs} epochs, "
      f"patience {args.patience}\n")

for epoch in range(start_epoch, args.max_epochs):
    train(epoch)
    should_stop = evaluate(epoch)
    scheduler.step()

    if should_stop:
        print(f"\nEarly stopping at epoch {epoch} "
              f"(no improvement for {args.patience} epochs)")
        print(f"Best val acc: {best_acc:.2f}%\n")
        break
else:
    print(f"\nReached max epochs ({args.max_epochs}). Best val acc: {best_acc:.2f}%\n")

final_acc = test()
print(f'Final test acc [{args.m_name}]: {final_acc:.2f}%')
run.log({"final test acc": final_acc})
run.finish()