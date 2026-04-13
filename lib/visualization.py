"""Visualization utilities for masks and overlays."""

import numpy as np

from .config import CLASS_COLORS


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert a class mask to an RGB image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        rgb[mask == class_id] = color

    return rgb


def make_overlay(image: np.ndarray, rgb_mask: np.ndarray) -> np.ndarray:
    """Blend an RGB mask with an image."""
    alpha = 0.5
    overlay = image.copy()
    fg = np.any(rgb_mask != 0, axis=-1)

    overlay[fg] = (
        alpha * rgb_mask[fg] + (1 - alpha) * overlay[fg]
    ).astype(np.uint8)
    return overlay
