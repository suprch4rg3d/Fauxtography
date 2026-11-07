"""STUB - CLIP zero-shot classification of real vs AI-generated images.

Ownership: one collaborator owns CLIP + its analysis.
API contract:
    CLIPZeroShot(config, device)
        .predict(images) -> (preds, probs)
  predictions follow the canonical label order: real=0, fake=1.
"""

from __future__ import annotations

import torch


class CLIPZeroShot:
    """Zero-shot real/fake classifier built on CLIP image/text encoders."""

    def __init__(self, config, device: torch.device | str = "auto") -> None:
        """TODO(impl): resolve device via utils.device.get_device (auto ->
        cuda -> mps -> cpu), then load CLIP via HF transformers and build
        prompt templates from config."""
        raise NotImplementedError("Implement in the CLIP milestone (see docs).")

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """images (B,C,H,W) -> (preds, probs) with labels 0=real, 1=fake."""
        raise NotImplementedError("Implement in the CLIP milestone (see docs).")