from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision import datasets


@dataclass
class GpuCifar10:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor

    def __len__(self) -> int:
        return self.x_train.shape[0]


def load_cifar10(root: str | Path = "./data", device: str = "cuda") -> GpuCifar10:
    root = str(root)
    tr = datasets.CIFAR10(root, train=True, download=True)
    te = datasets.CIFAR10(root, train=False, download=True)

    def to_gpu(ds) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(ds.data).to(device)
        x = x.permute(0, 3, 1, 2).float().div_(255.0).contiguous(
            memory_format=torch.channels_last
        )
        return x, torch.tensor(ds.targets, device=device)

    x_train, y_train = to_gpu(tr)
    x_test, y_test = to_gpu(te)
    return GpuCifar10(x_train, y_train, x_test, y_test)


def augment(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    n = x.shape[0]
    dev = x.device

    flip = torch.rand(n, device=dev, generator=generator) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)

    padded = torch.nn.functional.pad(x, (4, 4, 4, 4))
    dx, dy = torch.randint(0, 9, (2, n), device=dev, generator=generator)
    rows = (torch.arange(32, device=dev).view(1, 32) + dy.view(-1, 1))
    cols = (torch.arange(32, device=dev).view(1, 32) + dx.view(-1, 1))
    idx = torch.arange(n, device=dev).view(-1, 1, 1)
    out = padded[idx, :, rows.unsqueeze(-1), cols.unsqueeze(1)]
    return out.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
