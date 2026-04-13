"""General utility functions."""

import random
import shutil
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Get the available computation device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_div(a: int | float, b: int | float) -> float:
    """Safely divide two numbers."""
    eps = 1e-8
    return a / (b + eps)


def reset_dir(path: Path) -> None:
    """Remove and recreate a directory."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
