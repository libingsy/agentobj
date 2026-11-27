"""Device selection that stays on CPU when CUDA is not installed."""

from typing import Optional

import torch


def resolve_device(device: Optional[str] = None) -> str:
    """Return a torch device string. `auto`, `None`, and missing CUDA fall back to CPU."""
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print(f"[device] Requested {device} but CUDA is unavailable. Using cpu.")
        return "cpu"
    return str(device)
