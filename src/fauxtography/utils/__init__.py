"""Shared utilities: deterministic seeds, device resolution, logging."""

from .device import device_report, get_device
from .logging import get_logger
from .seed import set_seed

__all__ = ["device_report", "get_device", "get_logger", "set_seed"]
