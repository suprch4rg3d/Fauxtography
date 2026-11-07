"""Accelerator resolution.

Every model/script should pick its device through :func:`get_device` so the
choice happens exactly once and in one priority order:

    cuda -> mps (Apple Silicon) -> cpu

On Google Colab this resolves to `cuda` automatically. Callers can still
force a device by passing an explicit name (e.g. "cpu", "mps") instead of
"auto".
"""

from __future__ import annotations

import platform

import torch


def get_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve a device string to a torch device.

    ``"auto"`` (default) picks cuda, then mps on Apple Silicon, then cpu.
    Anything else is passed through to ``torch.device`` so callers can force
    a choice (useful in Colab to test CPU vs GPU behaviour).
    """
    if isinstance(requested, torch.device):
        return requested
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def device_report() -> dict[str, str | bool]:
    """Summary of the environment — used by the ``fauxtography env`` command."""
    torch_cuda = torch.cuda.is_available()
    mps_avail = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(get_device()),
        "cuda_available": torch_cuda,
        "gpu_name": (torch.cuda.get_device_name(0) if torch_cuda else "n/a"),
        "mps_available": mps_avail,
    }
    return report