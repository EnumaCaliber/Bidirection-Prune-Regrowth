'''Train CIFAR10 with PyTorch - Structured Pruning'''
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
import copy

from functools import partial
import torch_pruning as tp

from iclr2021_solution.tools.train import trainer_loader_wandb
from utils.utils import progress_bar
from utils.model_loader import model_loader
from utils.data_loader import data_loader
from iclr2021_solution.tools import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Structured Prune')

parser.add_argument('--m_name', type=str, default="vgg16")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--m_prune', type=str, default='vgg16')
parser.add_argument('--iter_start', type=int, default=1)
parser.add_argument('--iter_end', type=int, default=1)
parser.add_argument('--oneshot', type=float, default=0.99, help='channel sparsity')
parser.add_argument('--pruner', type=str, default='l1',
                    choices=['l1', 'lamp', 'taylor'],
                    help='structured pruning importance metric')
args = parser.parse_args()

run = wandb.init(
    project="model-prune-structured-oneshot",
    config=args,
    name=f"{args.m_name}_structured_{args.pruner}_sp{args.oneshot}",
)

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

train_loader, val_loader, test_loader = data_loader(data_dir='./data')

# -------------------------------------------------------
# 加载预训练模型
# -------------------------------------------------------
net = model_loader(args.m_name, device)
cktp = f'pretrain_{args.m_name}_ckpt.pth'
checkpoint = torch.load(f'./{args.m_name}/checkpoint/{cktp}')
net.load_state_dict(checkpoint['net'])
net.eval()

# 保存 dense model 备用（用于后续通道恢复）
dense_model = copy.deepcopy(net)

# -------------------------------------------------------
# 用真实 CIFAR10 数据作为 example_inputs
# -------------------------------------------------------
example_inputs, _ = next(iter(train_loader))
example_inputs = example_inputs[:1].to(device)  # 只取1张

# -------------------------------------------------------
# 结构化剪枝重要性度量
# -------------------------------------------------------
def get_importance(pruner_name, net, train_loader, device):
    if pruner_name == 'l1':
        return tp.importance.MagnitudeImportance(p=1)

    elif pruner_name == 'lamp':
        return tp.importance.LAMPImportance()

    elif pruner_name == 'taylor':
        # Taylor 需要梯度，先跑一次前向+反向
        importance = tp.importance.TaylorImportance()
        net.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            break  # 只需一个 batch 估计梯度
        net.eval()
        return importance

# -------------------------------------------------------
# 结构化 One-shot 剪枝函数
# -------------------------------------------------------
def structured_oneshot_prune(model, ch_sparsity, importance,
                              example_inputs, ignored_layers=[]):
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=importance,
        iterative_steps=1,       # one-shot = 1步
        ch_sparsity=ch_sparsity,
        ignored_layers=ignored_layers,
    )
    pruner.step()
    return model


def get_model_flops_params(model, example_inputs):
    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    return macs, params

# -------------------------------------------------------
# 主流程
# -------------------------------------------------------
_, sparsity, batch_size, opt_pre, opt_post = model_and_opt_loader(args.m_name, device)
trainer = trainer_loader_wandb()

target_folder = f'./{args.m_name}/ckpt_after_prune_structured_oneshot'
os.makedirs(target_folder, exist_ok=True)

# 剪枝前统计
macs_before, params_before = get_model_flops_params(net, example_inputs)
print(f"剪枝前 | MACs: {macs_before/1e6:.2f}M  Params: {params_before/1e6:.2f}M")

# opt 配置
opt_post = {
    "optimizer": partial(optim.AdamW, lr=0.0003),
    "steps": 400 * 313,
    "scheduler": None
}

# ignored layers（分类头不剪）
ignored_layers = []
for name, m in net.named_modules():
    if isinstance(m, nn.Linear):
        ignored_layers.append(m)

for it in range(args.iter_start, args.iter_end + 1):
    print(f"\n{'='*50}")
    print(f"Iteration {it} | Pruner: {args.pruner} | Sparsity: {args.oneshot}")

    # 获取重要性度量
    importance = get_importance(args.pruner, net, train_loader, device)

    # One-shot 结构化剪枝
    structured_oneshot_prune(
        net,
        ch_sparsity=args.oneshot,
        importance=importance,
        example_inputs=example_inputs,
        ignored_layers=ignored_layers,
    )

    # 剪枝后统计
    macs_after, params_after = get_model_flops_params(net, example_inputs)
    print(f"剪枝后 | MACs: {macs_after/1e6:.2f}M  Params: {params_after/1e6:.2f}M")
    print(f"压缩比 | MACs: {macs_before/macs_after:.2f}x  Params: {params_before/params_after:.2f}x")

    wandb.log({
        'macs_before':        macs_before,
        'macs_after':         macs_after,
        'params_before':      params_before,
        'params_after':       params_after,
        'compression_ratio':  params_before / params_after,
    })

    # ③ Finetune
    result_log = trainer(net, opt_post, train_loader, test_loader,
                         run=run, patience=50)


    formatted_sp = round(args.oneshot, 4)
    target_path = os.path.join(
        target_folder,
        f'pruned_structured_{args.pruner}_sp{formatted_sp}_it{it}.pth'
    )
    torch.save(net, target_path)
    print(f"已保存: {target_path}")

    dense_path = os.path.join(target_folder, f'dense_{args.m_name}.pth')
    torch.save(dense_model, dense_path)

wandb.finish()