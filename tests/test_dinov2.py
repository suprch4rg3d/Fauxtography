"""Tests for the DINOv2 feature extractor and linear probe."""

import torch

from fauxtography.data.transforms import dinov2_transforms
from fauxtography.models.linear_probe import LinearProbe


def test_linear_probe_output_shape():
    probe = LinearProbe(in_features=768, num_classes=2)
    out = probe(torch.randn(5, 768))
    assert out.shape == (5, 2)


def test_linear_probe_is_single_linear_layer():
    probe = LinearProbe(768, 2)
    assert len(list(probe.parameters())) == 2  # weight + bias
    assert probe.head.out_features == 2
    assert probe.head.in_features == 768


def test_dinov2_transforms_returns_both_pipelines():
    tf = dinov2_transforms(224)
    assert tf.train_transform is not None
    assert tf.eval_transform is not None
    # Frozen backbone: train and eval must use the same (non-augmenting)
    # pipeline so feature geometry is consistent.
    assert tf.train_transform is tf.eval_transform


def test_dinov2_arch_validation():
    import pytest

    from fauxtography.models.dinov2 import DINOv2FeatureExtractor

    with pytest.raises(ValueError):
        DINOv2FeatureExtractor(arch="not-a-dinov2")
