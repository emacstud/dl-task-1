"""Visualization utilities for masks and overlays."""

from pathlib import Path
import numpy as np
from PIL import Image

from lib.utils import numbered_png
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


def _ensure_dirs(out_dir: Path, names: list[str]) -> dict[str, Path]:
    """Create a set of subdirectories and return their paths."""
    dirs: dict[str, Path] = {}
    for name in names:
        path = out_dir / name
        path.mkdir(parents=True, exist_ok=True)
        dirs[name] = path
    return dirs


def save_prediction_outputs(
    index: int,
    image_rgb: np.ndarray,
    pred_mask: np.ndarray,
    out_dir: Path,
    save_input: bool = False,
    input_dir_name: str = "inputs",
) -> None:
    """Save prediction mask, RGB mask, overlay, and optionally the input image."""
    file_name = numbered_png(index)

    dir_names = ["pred_masks", "pred_masks_rgb", "pred_overlays"]
    if save_input:
        dir_names.append(input_dir_name)

    dirs = _ensure_dirs(out_dir, dir_names)

    pred_rgb = mask_to_rgb(pred_mask)
    pred_overlay = make_overlay(image_rgb, pred_rgb)

    if save_input:
        Image.fromarray(image_rgb).save(dirs[input_dir_name] / file_name)

    Image.fromarray(pred_mask).save(dirs["pred_masks"] / file_name)
    Image.fromarray(pred_rgb).save(dirs["pred_masks_rgb"] / file_name)
    Image.fromarray(pred_overlay).save(dirs["pred_overlays"] / file_name)


def save_evaluation_outputs(
    index: int,
    image_rgb: np.ndarray,
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    out_dir: Path,
) -> None:
    """Save input, prediction outputs, and ground-truth outputs."""
    file_name = numbered_png(index)

    save_prediction_outputs(
        index=index,
        image_rgb=image_rgb,
        pred_mask=pred_mask,
        out_dir=out_dir,
        save_input=True,
        input_dir_name="inputs",
    )

    dirs = _ensure_dirs(out_dir, ["gt_masks", "gt_masks_rgb", "gt_overlays"])

    true_rgb = mask_to_rgb(true_mask)
    true_overlay = make_overlay(image_rgb, true_rgb)

    Image.fromarray(true_mask).save(dirs["gt_masks"] / file_name)
    Image.fromarray(true_rgb).save(dirs["gt_masks_rgb"] / file_name)
    Image.fromarray(true_overlay).save(dirs["gt_overlays"] / file_name)