"""Baseline: a standard CNN (ResNet) fine-tuned on real-vs-fake.

Ownership: baseline is the shared reference point (joint work).
API contract:
    build_baseline_model(arch, pretrained, freeze_backbone,
                         num_classes, device) -> nn.Module
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from ..utils.device import get_device

_ARCHS = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
}


def build_baseline_model(
    arch: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    num_classes: int = 2,
    device: torch.device | str = "auto",
) -> nn.Module:
    """Build a torchvision ResNet with a fresh binary classification head.

    - ``pretrained`` uses ImageNet weights and replaces the final layer,
    - ``freeze_backbone`` keeps every layer except the new head trainable,
    - device resolves via ``get_device`` (auto -> cuda -> mps -> cpu).
    """
    if arch not in _ARCHS:
        raise ValueError(f"Unknown baseline arch {arch!r}; choose from {sorted(_ARCHS)}")

    device = get_device(device)
    weights = models.ResNet18_Weights.DEFAULT if (pretrained and arch == "resnet18") else None
    if pretrained and arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
    model = _ARCHS[arch](weights=weights)

    # Fresh binary head - ImageNet 1000-class fc is replaced.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    return model.to(device)
