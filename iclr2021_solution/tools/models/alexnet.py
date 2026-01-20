import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, BatchNorm2d, MaxPool2d, AvgPool2d


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, use_bn=True):
        super(AlexNet, self).__init__()
        # Adapted for CIFAR-10 (32x32 images) instead of ImageNet (224x224)
        self.features = nn.Sequential(
            # Conv1: smaller kernel and stride for CIFAR-10's smaller size
            Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            MaxPool2d(kernel_size=2),
            # Conv2
            Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MaxPool2d(kernel_size=2),
            # Conv3
            Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv4
            Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv5
            Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MaxPool2d(kernel_size=2),
        )
        # After the conv layers: 32 -> 16 -> 8 -> 4 -> 2
        # So final feature map is 256 * 2 * 2 = 1024
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def AlexNet_CIFAR10(use_bn=True):
    return AlexNet(num_classes=10, use_bn=use_bn)
