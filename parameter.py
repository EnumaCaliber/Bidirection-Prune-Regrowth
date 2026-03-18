import torch
import sys
sys.path.append('.')  # 确保能找到 models 目录

from models.shufflenetv2 import *

net = ShuffleNetV2(2)

# 参数量
total = sum(p.numel() for p in net.parameters())
print(f"Total params: {total:,}")

# forward 测试
x = torch.randn(1, 3, 32, 32)
y = net(x)
print(f"Output shape: {tuple(y.shape)}")
print("Forward pass OK ✓")