'''Train CIFAR10 with PyTorch - Structured Iterative Pruning (GMP)'''
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import wandb
import copy
from functools import partial
import torch_pruning as tp

from iclr2021_solution.tools.train import trainer_loader_wandb
from utils.model_loader import model_loader
from utils.data_loader import data_loader
from iclr2021_solution.tools import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Structured Iterative Prune')
parser.add_argument('--m_name',      type=str,   default="effnet")
parser.add_argument('--seed',        type=int,   default=42)
parser.add_argument('--pruner',      type=str,   default='l1',
                    choices=['l1', 'lamp', 'taylor'])
parser.add_argument('--target_sp',   type=float, default=0.99,
                    help='channel sparsity')
parser.add_argument('--iterative_steps', type=int, default=25,
                    help='')
parser.add_argument('--finetune_steps',  type=int, default=40*313,
                    help='')
args = parser.parse_args()

run = wandb.init(
    project="model-prune-structured-iterative",
    config=args,
    name=f"{args.m_name}_iterative_{args.pruner}_sp{args.target_sp}_iter{args.iterative_steps}",
)

import random
import numpy as np
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader, val_loader, test_loader = data_loader(data_dir='./data')

# -------------------------------------------------------
# 加载预训练模型
# -------------------------------------------------------
net = model_loader(args.m_name, device)
cktp = f'pretrain_{args.m_name}_ckpt.pth'
checkpoint = torch.load(f'./{args.m_name}/checkpoint/{cktp}')
net.load_state_dict(checkpoint['net'])
net.eval()

dense_model = copy.deepcopy(net)  # 备份 dense

# -------------------------------------------------------
# 真实数据作为 example_inputs
# -------------------------------------------------------
example_inputs, _ = next(iter(train_loader))
example_inputs = example_inputs[:1].to(device)

# -------------------------------------------------------
# 原始通道数记录（用于计算稀疏率）
# -------------------------------------------------------
original_channels = {}
for name, m in net.named_modules():
    if isinstance(m, nn.Conv2d):
        original_channels[name] = m.out_channels

# -------------------------------------------------------
# 工具函数
# -------------------------------------------------------
def get_importance(pruner_name, model, train_loader, device):
    if pruner_name == 'l1':
        return tp.importance.MagnitudeImportance(p=1)
    elif pruner_name == 'lamp':
        return tp.importance.LAMPImportance()
    elif pruner_name == 'taylor':
        importance = tp.importance.TaylorImportance()
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = nn.CrossEntropyLoss()(model(inputs), targets)
            loss.backward()
            break
        model.eval()
        return importance


def compute_channel_sparsity(model):
    total, remaining = 0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and name in original_channels:
            total     += original_channels[name]
            remaining += m.out_channels
    return 1 - remaining / total


def get_flops_params(model):
    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    return macs, params

# -------------------------------------------------------
# 构建迭代 pruner（一次性建好，step() 10次）
# -------------------------------------------------------
ignored_layers = [m for m in net.modules() if isinstance(m, nn.Linear)]

importance = get_importance(args.pruner, net, train_loader, device)

pruner = tp.pruner.MagnitudePruner(
    net,
    example_inputs,
    importance=importance,
    iterative_steps=args.iterative_steps,   # GMP 核心：分多步剪
    ch_sparsity=args.target_sp,             # 最终目标稀疏率
    ignored_layers=ignored_layers,
)

macs_before, params_before = get_flops_params(net)
print(f"剪枝前 | MACs: {macs_before/1e6:.2f}M  Params: {params_before/1e6:.2f}M")

# -------------------------------------------------------
# 保存目录
# -------------------------------------------------------
target_folder = f'./{args.m_name}/ckpt_structured_iterative'
os.makedirs(target_folder, exist_ok=True)

trainer = trainer_loader_wandb()
opt_post = {
    "optimizer": partial(optim.AdamW, lr=0.0003),
    "steps": 40 * 313,
    "scheduler": None
}

# -------------------------------------------------------
# GMP 迭代主循环
# -------------------------------------------------------
for step in range(args.iterative_steps):
    print(f"\n{'='*50}")
    print(f"Step {step+1}/{args.iterative_steps}")

    # ① 剪枝一步（cubic schedule 自动决定剪多少）
    pruner.step()

    # ② 当前稀疏率
    sparsity = compute_channel_sparsity(net)
    macs, params = get_flops_params(net)
    print(f"  channel sparsity : {sparsity:.2%}")
    print(f"  MACs  : {macs/1e6:.2f}M  ({macs_before/macs:.2f}x)")
    print(f"  Params: {params/1e6:.2f}M  ({params_before/params:.2f}x)")

    # ③ finetune
    result_log = trainer(net, opt_post, train_loader, test_loader,
                         run=run, patience=20)

    # ④ wandb log
    wandb.log({
        'step':              step + 1,
        'channel_sparsity':  sparsity,
        'macs':              macs,
        'params':            params,
        'compression_ratio': params_before / params,
    })

    # ⑤ 每步都保存（结构化必须存整个 model）
    formatted_sp = f"{sparsity:.3f}"
    save_path = os.path.join(
        target_folder,
        f'step{step+1:02d}_sp{formatted_sp}.pth'
    )
    torch.save(net, save_path)
    print(f"  已保存: {save_path}")

# 最后保存 dense model 备用
torch.save(dense_model, os.path.join(target_folder, f'dense_{args.m_name}.pth'))

print(f"\n{'='*50}")
print("迭代剪枝完成")
print(f"最终 channel sparsity: {compute_channel_sparsity(net):.2%}")
wandb.finish()