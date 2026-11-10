"""CLIP zero-shot classification: real=0, fake=1."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPModel

from ..utils.device import get_device


class CLIPZeroShot:
    """Frozen CLIP image/text encoders for zero-shot real/fake classification."""

    def __init__(self, config, device: torch.device | str = "auto") -> None:
        self.device = get_device() if str(device) == "auto" else torch.device(device)
        self.model = CLIPModel.from_pretrained(config.model_name).to(self.device)
        self.model.eval()

        if not config.prompts_real or not config.prompts_fake:
            raise ValueError("Both real and fake prompt lists must be nonempty.")

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        prompts = list(config.prompts_real) + list(config.prompts_fake)
        text_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        text_inputs = {key: val.to(self.device) for key, val in text_inputs.items()}

        # CLIP features are projected into the same image/text embedding space.
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            if not isinstance(text_features, torch.Tensor):
                text_features = text_features.pooler_output
            text_features = F.normalize(text_features.float(), dim=-1)

        n_real = len(config.prompts_real)
        real_features = text_features[:n_real].mean(dim=0)
        fake_features = text_features[n_real:].mean(dim=0)
        self.text_features = F.normalize(
            torch.stack([real_features, fake_features]), dim=-1
        )  # shape: (2, embedding_dim), always in [real, fake] order

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict from CLIP-normalized pixel tensors (B,3,H,W).

        Returns:
            preds: (B,) integer labels, real=0 and fake=1.
            probs: (B,2) class probabilities in [real, fake] order.
        """
        self.model.eval()
        images = images.to(self.device)
        image_features = self.model.get_image_features(pixel_values=images)
        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output
        image_features = F.normalize(image_features.float(), dim=-1)

        # Use CLIP's learned similarity scale, as in its standard forward pass.
        scale = self.model.logit_scale.exp().float()
        scores = scale * (image_features @ self.text_features.T)
        probs = torch.softmax(scores, dim=-1)
        preds = probs.argmax(dim=-1)
        return preds, probs
