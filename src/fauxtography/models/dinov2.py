"""DINOv2 - frozen feature extraction ([CLS] token).

Ownership: DINOv2 collaborator. Backed by HuggingFace ``transformers``
(``Dinov2Model``) because the pinned torchvision build ships no DINOv2
weights; ``transformers==5.22`` is already a project dependency.

API contract:
    DINOv2FeatureExtractor(arch, device)
        .extract(images) -> (B, embed_dim) features   # [CLS] token
"""

from __future__ import annotations

import torch

from ..utils.device import get_device

# arch name (configs/dinov2.yaml) -> HuggingFace checkpoint.
_ARCH_TO_HF = {
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
}


class DINOv2FeatureExtractor:
    """[CLS]-token feature extractor backed by a frozen DINOv2 backbone."""

    def __init__(self, arch: str = "dinov2_vitb14", device: torch.device | str = "auto") -> None:
        if arch not in _ARCH_TO_HF:
            raise ValueError(f"Unknown DINOv2 arch {arch!r}; choose from {sorted(_ARCH_TO_HF)}")
        self.device = get_device(device)

        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(_ARCH_TO_HF[arch]).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.embed_dim = self.model.config.hidden_size

    @property
    def hidden_size(self) -> int:
        return self.embed_dim

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """images (B,3,H,W) ImageNet-normalized -> (B, embed_dim) [CLS] features."""
        images = images.to(self.device)
        # Dinov2Model returns a BaseModelOutput; pooler_output is the [CLS]
        # token post-LayerNorm, shaped (B, embed_dim).
        return self.model(images).pooler_output
