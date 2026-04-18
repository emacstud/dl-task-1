"""Helpers for SAM-based zero-shot segmentation benchmarking with LangSAM."""

from typing import Any

import numpy as np
from lang_sam import LangSAM
from PIL import Image

from .config import CLASS_TO_ID, CLASSES, LSAM_CLASS_PROMPTS, PRIORITY
from .utils import NEAREST


def to_numpy(x: Any) -> np.ndarray | None:
    """Convert tensors / lists to numpy arrays safely."""
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def normalize_raw_masks(raw_masks: Any, target_hw: tuple[int, int]) -> list[np.ndarray]:
    """Convert raw LangSAM/SAM masks to a list of HxW boolean numpy arrays."""
    target_h, target_w = target_hw

    if raw_masks is None:
        return []

    raw_masks = to_numpy(raw_masks)
    if raw_masks is None:
        return []

    masks_list: list[np.ndarray] = []

    if raw_masks.ndim == 2:
        masks_list = [raw_masks]
    elif raw_masks.ndim == 3:
        masks_list = [raw_masks[i] for i in range(raw_masks.shape[0])]
    elif raw_masks.ndim == 4:
        if raw_masks.shape[1] == 1:
            masks_list = [raw_masks[i, 0] for i in range(raw_masks.shape[0])]
        elif raw_masks.shape[-1] == 1:
            masks_list = [raw_masks[i, :, :, 0] for i in range(raw_masks.shape[0])]
        else:
            masks_list = [np.squeeze(raw_masks[i]) for i in range(raw_masks.shape[0])]
    else:
        return []

    norm_masks: list[np.ndarray] = []

    for m in masks_list:
        m = np.asarray(m)
        m = np.squeeze(m)

        if m.ndim != 2:
            continue

        m = m > 0

        if m.shape != (target_h, target_w):
            m_img = Image.fromarray((m.astype(np.uint8) * 255))
            m = np.array(m_img.resize((target_w, target_h), NEAREST)) > 0

        norm_masks.append(m)

    return norm_masks


def get_gdino_boxes(
    model: LangSAM,
    image_pil: Image.Image,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run GroundingDINO and return (boxes, scores).
    """
    raw = model.gdino.predict(
        images_pil=[image_pil],
        texts_prompt=[prompt],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    if isinstance(raw, list):
        raw = raw[0] if raw else {}

    boxes = None
    scores = None

    if isinstance(raw, dict):
        boxes = raw.get("boxes", None)
        scores = raw.get("scores", None)

    elif isinstance(raw, tuple):
        boxes = raw[0] if len(raw) > 0 else None
        scores = raw[1] if len(raw) > 1 else None

        if isinstance(boxes, list):
            boxes = boxes[0] if boxes else None
        if isinstance(scores, list):
            scores = scores[0] if scores else None
    else:
        raise TypeError(f"Unsupported GroundingDINO output type: {type(raw)}")

    boxes = to_numpy(boxes)
    scores = to_numpy(scores)

    if boxes is None:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    else:
        boxes = boxes.reshape(-1, 4)

    if scores is None:
        scores = np.zeros((len(boxes),), dtype=np.float32)
    else:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    return boxes, scores


def get_sam_masks_for_boxes(
    model: LangSAM,
    image_pil: Image.Image,
    boxes: np.ndarray,
    target_hw: tuple[int, int],
) -> list[np.ndarray]:
    """
    Run SAM one box at a time to avoid the buggy batch path.
    Returns a list of boolean masks.
    """
    if len(boxes) == 0:
        return []

    image_np = np.asarray(image_pil).copy()
    all_masks: list[np.ndarray] = []

    for box in boxes:
        one_box = np.asarray(box, dtype=np.float32).reshape(1, 4)

        sam_out = model.sam.predict(image_rgb=image_np, xyxy=one_box)

        if isinstance(sam_out, tuple):
            raw_masks = sam_out[0]
        elif isinstance(sam_out, dict):
            raw_masks = sam_out.get("masks", None)
        else:
            raw_masks = sam_out

        all_masks.extend(normalize_raw_masks(raw_masks, target_hw))

    return all_masks


def predict_langsam_semantic_mask(
    model: LangSAM,
    image_rgb: np.ndarray,
    box_threshold: float,
    text_threshold: float,
    top1_box_only: bool = True,
) -> np.ndarray:
    """
    Build a semantic mask:
    background=0, Eagle=1, Laptop=2, Dog=3
    """
    h, w = image_rgb.shape[:2]
    image_pil = Image.fromarray(image_rgb)

    semantic_mask = np.zeros((h, w), dtype=np.uint8)

    for class_name in sorted(CLASSES, key=lambda x: PRIORITY[x]):
        prompt = LSAM_CLASS_PROMPTS[class_name]

        boxes, scores = get_gdino_boxes(
            model=model,
            image_pil=image_pil,
            prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if len(boxes) == 0:
            continue

        if top1_box_only and len(boxes) > 1:
            if len(scores) == len(boxes) and len(scores) > 0:
                best_idx = int(np.argmax(scores))
            else:
                best_idx = 0
            boxes = boxes[best_idx:best_idx + 1]

        class_masks = get_sam_masks_for_boxes(
            model=model,
            image_pil=image_pil,
            boxes=boxes,
            target_hw=(h, w),
        )

        if not class_masks:
            continue

        merged_class_mask = np.zeros((h, w), dtype=bool)
        for m in class_masks:
            merged_class_mask |= m

        semantic_mask[merged_class_mask] = CLASS_TO_ID[class_name]

    return semantic_mask