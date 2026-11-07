"""Fauxtography - Detecting AI-Generated Images.

Distinguishes real photographs from AI-generated images. Three approaches are
compared:
  1. Baseline: a fine-tuned ResNet CNN.
  2. CLIP: zero-shot text-prompt classification.
  3. DINOv2: deep visual features + a linear probe.
"""

__version__ = "0.1.0"