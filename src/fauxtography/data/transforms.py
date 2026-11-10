"""Preprocessing pipelines for each model.

Each family exposes ``ModelTransforms(train_transform, eval_transform)``:
- baseline: ImageNet-style resize/crop/flip + ImageNet normalization.
- clip: CLIP resize + CLIP normalization (filled in with the CLIP milestone).
- dinov2: DINOv2 resize + ImageNet normalization (filled in with the DINOv2
  milestone).
"""

from __future__ import annotations

from dataclasses import dataclass

from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class ModelTransforms:
    train_transform: transforms.Compose
    eval_transform: transforms.Compose


def _imagenet_normalize() -> transforms.Normalize:
    return transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


def baseline_transforms(image_size: int = 224, augment: bool = True) -> ModelTransforms:
    """ResNet pipeline: resize + standard augmentation + ImageNet normalisation.

    Evaluation/val sets get resize-only (no random crop or flip) so metrics are
    stable; the train pipeline uses random resize crop + horizontal flip.
    """
    if augment:
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                _imagenet_normalize(),
            ]
        )
    else:
        train_tf = eval_tf = _resize_eval(image_size)
    eval_tf = _resize_eval(image_size)
    return ModelTransforms(train_transform=train_tf, eval_transform=eval_tf)


def _resize_eval(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            _imagenet_normalize(),
        ]
    )


def clip_transforms(image_size: int = 224) -> ModelTransforms:
    """TODO(impl): CLIP-consistent resize + CLIP normalization."""
    raise NotImplementedError


def dinov2_transforms(image_size: int = 224, square: bool = False) -> ModelTransforms:
    """DINOv2 pipeline: ImageNet normalization, train = eval (frozen backbone).

    Default follows the official DINOv2 preprocessing: resize the shorter
    side to `image_size + 32` (256 for 224) then center-crop to
    `image_size`, which preserves aspect ratio. Square-stretch past 224 can
    degrade ViT features more than CNNs, so it is opt-in via ``square=True``
    (kept for A/B comparison with the earlier runs).
    """
    if square:
        eval_tf = _resize_eval(image_size)
    else:
        eval_tf = transforms.Compose(
            [
                transforms.Resize(image_size + 32),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                _imagenet_normalize(),
            ]
        )
    return ModelTransforms(train_transform=eval_tf, eval_transform=eval_tf)
