"""Model implementations for the three compared approaches."""

from .baseline import build_baseline_model
from .clip import CLIPZeroShot
from .dinov2 import DINOv2FeatureExtractor
from .linear_probe import LinearProbe

__all__ = ["CLIPZeroShot", "DINOv2FeatureExtractor", "LinearProbe", "build_baseline_model"]
