"""Model building, training, evaluation, and inference utilities."""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from segmentation_models_pytorch import Unet
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CLASS_IDS
from .losses import compute_total_loss
from .metrics import build_metric_rows
from .utils import safe_div


def build_unet(encoder_name: str, encoder_weights: str | None, num_classes: int) -> Unet:
    """Build a U-Net model."""
    return Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[Unet, Any]:
    """Load a model from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    model = build_unet(
        encoder_name=ckpt["encoder"],
        encoder_weights=None,
        num_classes=ckpt["num_classes"],
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    return model, ckpt


def train_one_epoch(
    model: Unet,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    num_classes: int,
    loss_mode: Literal["weighted_ce_dice", "focal_dice"],
    ce_loss_fn: CrossEntropyLoss | None = None,
    focal_gamma: float = 2.0,
    cls_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> tuple[float, float, list[dict], float]:
    """Train the model for one epoch."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_pixels = 0

    tp = {c: 0 for c in CLASS_IDS}
    fp = {c: 0 for c in CLASS_IDS}
    fn = {c: 0 for c in CLASS_IDS}

    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)

        loss, _, _ = compute_total_loss(
            logits=logits,
            masks=masks,
            num_classes=num_classes,
            loss_mode=loss_mode,
            ce_loss_fn=ce_loss_fn,
            focal_gamma=focal_gamma,
            cls_weight=cls_weight,
            dice_weight=dice_weight,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == masks).sum().item()
        total_pixels += masks.numel()

        for class_id in CLASS_IDS:
            pred_c = preds == class_id
            true_c = masks == class_id

            tp[class_id] += (pred_c & true_c).sum().item()
            fp[class_id] += (pred_c & (~true_c)).sum().item()
            fn[class_id] += ((~pred_c) & true_c).sum().item()

    avg_loss = total_loss / len(loader)
    pixel_acc = total_correct / max(1, total_pixels)
    rows, macro_f1 = build_metric_rows(tp, fp, fn)
    return avg_loss, pixel_acc, rows, macro_f1


@torch.no_grad()
def validate_one_epoch(
    model: Unet,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    loss_mode: Literal["weighted_ce_dice", "focal_dice"],
    ce_loss_fn: CrossEntropyLoss | None = None,
    focal_gamma: float = 2.0,
    cls_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> tuple[float, float, list[dict], float]:
    """Validate the model for one epoch."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_pixels = 0

    tp = {c: 0 for c in CLASS_IDS}
    fp = {c: 0 for c in CLASS_IDS}
    fn = {c: 0 for c in CLASS_IDS}

    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        loss, _, _ = compute_total_loss(
            logits=logits,
            masks=masks,
            num_classes=num_classes,
            loss_mode=loss_mode,
            ce_loss_fn=ce_loss_fn,
            focal_gamma=focal_gamma,
            cls_weight=cls_weight,
            dice_weight=dice_weight,
        )

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == masks).sum().item()
        total_pixels += masks.numel()

        for class_id in CLASS_IDS:
            pred_c = preds == class_id
            true_c = masks == class_id

            tp[class_id] += (pred_c & true_c).sum().item()
            fp[class_id] += (pred_c & (~true_c)).sum().item()
            fn[class_id] += ((~pred_c) & true_c).sum().item()

    avg_loss = total_loss / len(loader)
    pixel_acc = total_correct / max(1, total_pixels)
    rows, macro_f1 = build_metric_rows(tp, fp, fn)

    return avg_loss, pixel_acc, rows, macro_f1


def save_checkpoint(
    checkpoint_path: Path,
    model: Unet,
    encoder: str,
    encoder_weights_used: str,
    num_classes: int,
    size: int,
    loss_mode: str,
    focal_gamma: float,
    cls_weight: float,
    dice_weight: float,
    best_val_macro_f1: float,
    class_mapping: dict[str, int],
) -> None:
    """Save a training checkpoint."""
    torch.save(
        {
            "model_state": model.state_dict(),
            "encoder": encoder,
            "encoder_weights_used": encoder_weights_used,
            "num_classes": num_classes,
            "size": size,
            "loss_mode": loss_mode,
            "focal_gamma": focal_gamma,
            "cls_weight": cls_weight,
            "dice_weight": dice_weight,
            "best_val_macro_f1": best_val_macro_f1,
            "class_mapping": class_mapping,
        },
        checkpoint_path,
    )


@torch.no_grad()
def evaluate_model(model: Unet, loader: DataLoader, device: torch.device) -> tuple[float, list[dict], float]:
    """Evaluate the model on a dataset."""
    model.eval()

    tp = {c: 0 for c in CLASS_IDS}
    fp = {c: 0 for c in CLASS_IDS}
    fn = {c: 0 for c in CLASS_IDS}

    total_correct = 0
    total_pixels = 0

    for images, masks, _ in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        total_correct += (preds == masks).sum().item()
        total_pixels += masks.numel()

        for class_id in CLASS_IDS:
            pred_c = preds == class_id
            true_c = masks == class_id

            tp[class_id] += (pred_c & true_c).sum().item()
            fp[class_id] += (pred_c & (~true_c)).sum().item()
            fn[class_id] += ((~pred_c) & true_c).sum().item()

    pixel_acc = safe_div(total_correct, total_pixels)
    rows, macro_f1 = build_metric_rows(tp, fp, fn)

    return pixel_acc, rows, macro_f1


@torch.no_grad()
def predict_mask(model: Unet, image_rgb: np.ndarray, transform: Any, device: torch.device) -> np.ndarray:
    """Predict a segmentation mask for one image."""
    out = transform(image=image_rgb)
    image_tensor = out["image"].unsqueeze(0).to(device)

    logits = model(image_tensor)
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred
