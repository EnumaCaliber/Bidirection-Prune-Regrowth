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

from functools import partial

from iclr2021_solution.tools.train import trainer_loader_wandb
from utils.utils import progress_bar
from utils.model_loader import model_loader
from utils.data_loader import data_loader
from iclr2021_solution.tools import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Prune')

parser.add_argument('--m_name', type=str, default="vgg16",
                    help='Model name (e.g., resnet18, vgg16, etc.)')
# desnet, effnet, resnet20, vgg16
parser.add_argument('--pruner', type=str, default='lamp', help='pruning method')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
parser.add_argument('--m_prune', type=str, default='oneshot', help="oneshot and iterate")
parser.add_argument('--iter_start', type=int, default=1, help='start iteration for pruning')
parser.add_argument('--iter_end', type=int, default=1, help='end iteration for pruning')
parser.add_argument('--oneshot', type=int, default=0.94, help='end iteration for pruning')
args = parser.parse_args()

run = wandb.init(
    project="model-prune-oneshot-0.94",
    config=args,
    name=f"{args.m_name}_prune_{args.pruner}_oneshot_{args.oneshot}",
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

net = model_loader(args.m_name, device)
cktp = f'pretrain_{args.m_name}_ckpt.pth'
checkpoint = torch.load(f'./{args.m_name}/checkpoint/{cktp}')
net.load_state_dict(checkpoint['net'])
net.eval()

true_mask = 1 - get_model_sparsity(net)
print(f'true mask {true_mask}')


# ---------------------Pretrain Finished-----------------------------
_, sparsity, batch_size, opt_pre, opt_post = model_and_opt_loader(args.m_name, device)

pruner = weight_pruner_loader(args.pruner)
trainer = trainer_loader_wandb()
utils.prune_weights_reparam(net)

""" PRUNE AND RETRAIN """

target_folder = f'./{args.m_name}/ckpt_after_prune_oneshot'
os.makedirs(target_folder, exist_ok=True)



if args.m_prune == 'iterate':
    sparsity = 0.3
    opt_post = {
        "optimizer": partial(optim.AdamW, lr=0.0003),
        "steps": 40*313,  # 40000 for iterative, 400000 for one-shot
        "scheduler": None
    }
else:
    sparsity = args.oneshot
    opt_post = {
        "optimizer": partial(optim.AdamW, lr=0.0003),
        "steps": 400*313,  # 40000 for iterative, 400000 for one-shot
        "scheduler": None
    }

for it in range(args.iter_start, args.iter_end + 1):
    print(f"Pruning for iteration {it}: METHOD: {args.pruner}")
    pruner(net, sparsity)
    result_log = trainer(net, opt_post, train_loader, test_loader,run = run,patience=50)


    true_mask = 1 - get_model_sparsity(net)
    formatted_mask = round(true_mask, 4)


    target_path_mask = os.path.join(target_folder, f'pruned_oneshot_mask_{formatted_mask}_test.pth')
    torch.save(net.state_dict(), target_path_mask)

