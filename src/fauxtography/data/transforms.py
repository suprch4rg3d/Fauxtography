"""Preprocessing pipelines (contracts) for each model.

Each family exposes ``ModelTransforms(train_transform, eval_transform)``:
- baseline: ImageNet-style resize/crop/flip + ImageNet normalization.
- clip: CLIP resize + CLIP normalization.
- dinov2: DINOv2 resize + ImageNet normalization.

These are the *contracts*; the exact augmentation choices are filled in per
owner in the relevant milestone and tuned as we analyze results.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchvision import transforms


@dataclass
class ModelTransforms:
    train_transform: transforms.Compose
    eval_transform: transforms.Compose


def baseline_transforms(image_size: int = 224, augment: bool = True) -> ModelTransforms:
    """TODO(impl): ResNet pipeline (resize/crop + normal aug + ImageNet norm)."""
    raise NotImplementedError


def clip_transforms(image_size: int = 224) -> ModelTransforms:
    """TODO(impl): CLIP-consistent resize + CLIP normalization."""
    raise NotImplementedError


def dinov2_transforms(image_size: int = 224) -> ModelTransforms:
    """TODO(impl): DINOv2 resize + ImageNet normalization."""
    raise NotImplementedError