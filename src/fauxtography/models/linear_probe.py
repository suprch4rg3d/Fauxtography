"""Linear probe trained on extracted DINOv2 features.

Ownership: DINOv2 collaborator.
API contract:
    LinearProbe(in_features, num_classes)  # nn.Module, single Linear head
"""

from __future__ import annotations

import torch
from torch import nn


class LinearProbe(nn.Module):
    """A single linear classifier on top of frozen features."""

    def __init__(self, in_features: int, num_classes: int = 2) -> None:
        super().__init__()
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features (B, embed_dim) -> (B, num_classes) logits."""
        return self.head(features)
