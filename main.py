'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import wandb


from utils.utils import progress_bar
from utils.model_loader import model_loader
from utils.data_loader import data_loader
from iclr2021_solution.tools import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--m_name',type=str, required=True,
                    help='Model name (e.g., resnet18, vgg16, etc.)')
parser.add_argument('--pruner', type=str, help='pruning method')
parser.add_argument('--iter_start', type=int, default=1, help='start iteration for pruning')
parser.add_argument('--iter_end', type=int, default=1, help='end iteration for pruning')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
parser.add_argument('--max_epochs', type=int, default=500, help='maximum pretraining epochs')
parser.add_argument('--patience', type=int, default=30, help='early stopping patience (epochs without improvement)')

args = parser.parse_args()

# Set random seed for reproducibility
import random
import numpy as np
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Random seed set to: {args.seed}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
epochs_without_improvement = 0  # For early stopping

# Data
print('==> Preparing data..')
train_loader, val_loader, test_loader = data_loader(data_dir='./data')

# Model
print('==> Building model..')
net = model_loader(args.m_name, device)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(f'./{args.m_name}/checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./{args.m_name}/checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    net.eval()

    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
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

        # progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100. * correct / total
    print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%', end=' | ')


def evaluate(epoch):
    global best_acc
    global epochs_without_improvement
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):  # val_loader
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    avg_val_loss = test_loss / len(val_loader)
    acc = 100.*correct/total
    print(f'Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}%', end='')
    
    if acc > best_acc:
        print(f' ✓ NEW BEST (prev: {best_acc:.2f}%)')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        target_folder = os.path.join(args.m_name, 'checkpoint')
        os.makedirs(target_folder, exist_ok=True)
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        target_path = os.path.join(target_folder, 'ckpt.pth')
        torch.save(state, target_path)
        best_acc = acc
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement <= 5:
            print(f' (no improvement: {epochs_without_improvement})')
        elif epochs_without_improvement % 10 == 0:
            print(f' (no improvement: {epochs_without_improvement} epochs)')
        else:
            print()
    
    return epochs_without_improvement >= args.patience

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

if args.resume != True:
    print(f"\nStarting pretraining for up to {args.max_epochs} epochs (patience: {args.patience})")
    print(f"Early stopping if no improvement for {args.patience} consecutive epochs\n")
    
    for epoch in range(start_epoch, args.max_epochs):
        train(epoch)
        should_stop = evaluate(epoch)
        scheduler.step()
        
        if should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"No improvement for {args.patience} epochs")
            print(f"Best validation accuracy: {best_acc:.2f}%\n")
            break
    else:
        print(f"\nReached maximum epochs ({args.max_epochs})")
        print(f"Best validation accuracy: {best_acc:.2f}%\n")

acc = test(net)
print(f'best acc for {args.m_name}', acc)

# ---------------------Pretrain Finished-----------------------------
_, sparsity, batch_size, opt_pre, opt_post = model_and_opt_loader(args.m_name, device)
print(f'this is for model {args.m_name} with sparsity: ', sparsity)

pruner = weight_pruner_loader(args.pruner)
trainer = trainer_loader()
utils.prune_weights_reparam(net)

""" PRUNE AND RETRAIN """

# print(f"Pruning for METHOD: {args.pruner}")
# pruner(net, sparsity)    
# result_log = trainer(net ,opt_post,train_loader,test_loader)

target_folder = f'./{args.m_name}/ckpt_after_prune'
os.makedirs(target_folder, exist_ok=True)
# target_folder = f'iterative_0.6_10/{args.m_name}'
# os.makedirs(target_folder, exist_ok=True)


# Initialize result log CSV
import csv
log_csv_path = os.path.join(target_folder, f'training_log_{args.pruner}_{sparsity}.csv')
with open(log_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['iteration', 'test_acc', 'test_loss', 'train_acc', 'train_loss'])

for it in range(args.iter_start,args.iter_end+1):
    print(f"Pruning for iteration {it}: METHOD: {args.pruner}")
    pruner(net, sparsity)    
    result_log = trainer(net,opt_post,train_loader,test_loader)
    
    # Append results to CSV
    with open(log_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([it] + result_log)
    print(f"  Iteration {it} results: Test acc={result_log[0]:.2f}%, Train acc={result_log[2]:.2f}%")

    target_path_mask = os.path.join(target_folder, f'pruned_finetuned_mask_{sparsity}.pth')
    torch.save(net.state_dict(), target_path_mask)

print(f"\nTraining log saved to: {log_csv_path}")