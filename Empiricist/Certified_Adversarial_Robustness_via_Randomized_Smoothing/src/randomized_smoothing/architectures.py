from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class NormalizeLayer(nn.Module):
    def __init__(self, mean=CIFAR10_MEAN, std=CIFAR10_STD) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNetCifar(nn.Module):
    def __init__(self, depth: int = 110, num_classes: int = 10) -> None:
        super().__init__()
        if (depth - 2) % 6:
            raise ValueError("CIFAR ResNet depth must be 6n+2 (e.g. 20, 56, 110)")
        blocks = (depth - 2) // 6
        self.depth = depth
        self.normalize = NormalizeLayer()
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self._cin = 16
        self.layer1 = self._stage(16, blocks, 1)
        self.layer2 = self._stage(32, blocks, 2)
        self.layer3 = self._stage(64, blocks, 2)
        self.fc = nn.Linear(64, num_classes)

    def _stage(self, cout: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(BasicBlock(self._cin, cout, s))
            self._cin = cout
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(self.normalize(x))))
        out = self.layer3(self.layer2(self.layer1(out)))
        return self.fc(F.adaptive_avg_pool2d(out, 1).flatten(1))
