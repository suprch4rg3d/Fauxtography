"""STUB — DINOv2 feature extraction.

Ownership: one collaborator owns DINOv2 + linear probe + its analysis.
API contract:
    DINOv2FeatureExtractor(arch, device)
        .extract(images) -> (B, embed_dim) features   # [CLS] token
"""

from __future__ import annotations

import torch


class DINOv2FeatureExtractor:
    """[CLS]-token feature extractor backed by a DINOv2 vision backbone."""

    def __init__(self, arch: str = "dinov2_vitb14", device: torch.device | str = "auto") -> None:
        """TODO(impl): resolve device via utils.device.get_device (auto ->
        cuda -> mps -> cpu), then load a torchvision DINOv2 backbone, keep
        it frozen in eval mode."""
        raise NotImplementedError("Implement in the DINOv2 milestone (see docs).")

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """images (B,3,H,W) ImageNet-normalized -> (B, embed_dim) features."""
        raise NotImplementedError("Implement in the DINOv2 milestone (see docs).")