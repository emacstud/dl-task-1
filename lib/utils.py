"""General utility functions."""

import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image


BILINEAR = getattr(Image.Resampling, "BILINEAR")
NEAREST = getattr(Image.Resampling, "NEAREST")


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


def list_image_files(path: Path) -> list[Path]:
    """Return sorted image files from a directory."""
    path = Path(path)
    files = [
        p for p in path.iterdir()
        if p.is_file()
    ]
    return sorted(files)


def get_resampling(name: str):
    """Compatibility helper for Pillow versions."""
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


def numbered_png(index: int) -> str:
    """Return a sequential PNG filename like 1.png, 2.png, ..."""
    return f"{index}.png"


def load_rgb_image(path: Path) -> np.ndarray:
    """Load an RGB image as a writable numpy array."""
    return np.array(Image.open(path).convert("RGB")).copy()


def resize_rgb_image(image_rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize an RGB image to square size x size."""
    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
    return np.array(image.resize((size, size), BILINEAR)).copy()


def load_resized_rgb_image(path: Path, size: int) -> np.ndarray:
    """Load and resize an RGB image."""
    image = Image.open(path).convert("RGB").resize((size, size), BILINEAR)
    return np.array(image).copy()


def load_mask(path: Path) -> np.ndarray:
    """Load a segmentation mask as uint8."""
    return np.array(Image.open(path), dtype=np.uint8).copy()


def load_resized_mask(path: Path, size: int) -> np.ndarray:
    """Load and resize a segmentation mask using nearest-neighbor interpolation."""
    mask = Image.open(path).resize((size, size), NEAREST)
    return np.array(mask, dtype=np.uint8).copy()
