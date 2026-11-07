"""STUB - Baseline: a standard CNN (ResNet) fine-tuned on real-vs-fake.

Ownership: baseline is the shared reference point (optionally joint work).
API contract:
    build_baseline_model(arch, pretrained, freeze_backbone,
                         num_classes, device) -> nn.Module
"""

from __future__ import annotations

import torch
from torch import nn


def build_baseline_model(
    arch: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    num_classes: int = 2,
    device: torch.device | str = "auto",
) -> nn.Module:
    """TODO(impl): resolve device via utils.device.get_device (auto -> cuda
    -> mps -> cpu), then build a torchvision ResNet with a fresh binary head."""
    raise NotImplementedError("Implement in the baseline milestone (see docs).")