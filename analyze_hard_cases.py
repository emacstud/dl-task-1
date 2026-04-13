"""Analyze the hardest test images."""

import argparse
import csv
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from lib.config import (
    CHECKPOINTS_DIR,
    CLASS_IDS,
    ID_TO_CLASS,
    OUTPUTS_DIR,
    SEMANTIC_ROOT,
)
from lib.datasets import SemanticSegDataset
from lib.losses import compute_enet_class_weights, compute_total_loss
from lib.metrics import build_metric_rows
from lib.model import load_model_from_checkpoint
from lib.transforms import get_eval_transform
from lib.utils import get_device, safe_div
from lib.visualization import make_overlay, mask_to_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=SEMANTIC_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "best_focal_dice_resnet18_384_imagenet_30e.pth")
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS_DIR / "hard_cases")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--sort_by",
        type=str,
        default="loss",
        choices=["loss", "macro_f1", "error_rate"],
    )
    parser.add_argument("--expected_test_size", type=int, default=100)
    return parser.parse_args()


def build_case_csv_fieldnames() -> list[str]:
    """Build CSV column names for per-image analysis."""
    fieldnames = [
        "rank",
        "stem",
        "loss",
        "cls_loss",
        "dice_loss",
        "pixel_acc",
        "error_rate",
        "macro_f1",
    ]

    for class_id in CLASS_IDS:
        class_name = ID_TO_CLASS[class_id]
        fieldnames.extend([
            f"{class_name}_TP",
            f"{class_name}_FP",
            f"{class_name}_FN",
            f"{class_name}_precision",
            f"{class_name}_recall",
            f"{class_name}_f1",
        ])

    return fieldnames


def case_to_csv_row(case: dict[str, Any]) -> dict[str, Any]:
    """Convert one case record to a CSV row."""
    row = {
        "rank": case["rank"],
        "stem": case["stem"],
        "loss": case["loss"],
        "cls_loss": case["cls_loss"],
        "dice_loss": case["dice_loss"],
        "pixel_acc": case["pixel_acc"],
        "error_rate": case["error_rate"],
        "macro_f1": case["macro_f1"],
    }

    for class_row in case["rows"]:
        class_name = class_row["class_name"]
        row[f"{class_name}_TP"] = class_row["TP"]
        row[f"{class_name}_FP"] = class_row["FP"]
        row[f"{class_name}_FN"] = class_row["FN"]
        row[f"{class_name}_precision"] = class_row["precision"]
        row[f"{class_name}_recall"] = class_row["recall"]
        row[f"{class_name}_f1"] = class_row["f1"]

    return row


def save_cases_csv(cases: list[dict[str, Any]], csv_path: Path) -> None:
    """Save case analysis results to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=build_case_csv_fieldnames())
        writer.writeheader()
        for case in cases:
            writer.writerow(case_to_csv_row(case))


def make_error_map(pred_mask: np.ndarray, true_mask: np.ndarray) -> np.ndarray:
    """Build an RGB error map."""
    error_rgb = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    error_rgb[pred_mask != true_mask] = (255, 0, 0)
    return error_rgb


def sort_cases(cases: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    """Sort cases by difficulty."""
    if sort_by == "loss":
        sorted_cases = sorted(cases, key=lambda x: x["loss"], reverse=True)
    elif sort_by == "error_rate":
        sorted_cases = sorted(cases, key=lambda x: x["error_rate"], reverse=True)
    elif sort_by == "macro_f1":
        sorted_cases = sorted(cases, key=lambda x: x["macro_f1"])
    else:
        raise ValueError(f"Unknown sort criterion: {sort_by}")

    for rank, case in enumerate(sorted_cases, start=1):
        case["rank"] = rank

    return sorted_cases


def save_case_visuals(case: dict[str, Any], output_dir: Path) -> None:
    """Save visual outputs for one hard case."""
    stem = case["stem"]
    rank = case["rank"]

    case_dir = output_dir / f"rank_{rank:02d}_{stem}"
    case_dir.mkdir(parents=True, exist_ok=True)

    image_rgb = case["image_rgb"]
    pred_mask = case["pred_mask"]
    true_mask = case["true_mask"]

    pred_rgb = mask_to_rgb(pred_mask)
    true_rgb = mask_to_rgb(true_mask)
    pred_overlay = make_overlay(image_rgb, pred_rgb)
    true_overlay = make_overlay(image_rgb, true_rgb)

    error_rgb = make_error_map(pred_mask, true_mask)
    error_overlay = make_overlay(image_rgb, error_rgb)

    Image.fromarray(image_rgb).save(case_dir / "input.png")
    Image.fromarray(true_mask).save(case_dir / "gt_mask.png")
    Image.fromarray(pred_mask).save(case_dir / "pred_mask.png")
    Image.fromarray(true_rgb).save(case_dir / "gt_mask_rgb.png")
    Image.fromarray(pred_rgb).save(case_dir / "pred_mask_rgb.png")
    Image.fromarray(true_overlay).save(case_dir / "gt_overlay.png")
    Image.fromarray(pred_overlay).save(case_dir / "pred_overlay.png")
    Image.fromarray(error_rgb).save(case_dir / "error_map.png")
    Image.fromarray(error_overlay).save(case_dir / "error_overlay.png")


def main() -> None:
    args = parse_args()

    device = get_device()
    print("Device:", device)

    model, ckpt = load_model_from_checkpoint(args.checkpoint, device)

    size = ckpt["size"]
    num_classes = ckpt["num_classes"]
    loss_mode = ckpt.get("loss_mode", "weighted_ce_dice")
    focal_gamma = ckpt.get("focal_gamma", 2.0)
    cls_weight = ckpt.get("cls_weight", 1.0)
    dice_weight = ckpt.get("dice_weight", 1.0)

    print("Loaded checkpoint:")
    print("size:", size)
    print("num_classes:", num_classes)
    print("loss_mode:", loss_mode)

    data_root = Path(args.data_root)
    test_images_dir = data_root / "test" / "images"
    test_masks_dir = data_root / "test" / "masks"
    train_masks_dir = data_root / "train" / "masks"

    test_ds = SemanticSegDataset(
        test_images_dir,
        test_masks_dir,
        transform=get_eval_transform(size),
        return_stem=True,
    )

    print("Test samples:", len(test_ds))
    if len(test_ds) != args.expected_test_size:
        print("WARNING: expected 100 unseen test images, found", len(test_ds))

    ce_loss_fn = None
    if loss_mode == "weighted_ce_dice":
        class_weights, _, _ = compute_enet_class_weights(train_masks_dir, num_classes)
        ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Using loss for analysis: weighted CE + Dice")
    elif loss_mode == "focal_dice":
        print("Using loss for analysis: Focal + Dice")

    cases: list[dict[str, Any]] = []

    for idx in tqdm(range(len(test_ds)), desc="Analyzing test cases"):
        image_tensor, mask_tensor, stem = cast(tuple[Tensor, Tensor, str], test_ds[idx])

        image_batch = image_tensor.unsqueeze(0).to(device)
        mask_batch = mask_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_batch)

            total_loss, cls_loss, dice_loss = compute_total_loss(
                logits=logits,
                masks=mask_batch,
                num_classes=num_classes,
                loss_mode=loss_mode,
                ce_loss_fn=ce_loss_fn,
                focal_gamma=focal_gamma,
                cls_weight=cls_weight,
                dice_weight=dice_weight,
            )

            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        true_mask = mask_tensor.cpu().numpy().astype(np.uint8)

        total_correct = int((pred_mask == true_mask).sum())
        total_pixels = int(true_mask.size)
        pixel_acc = safe_div(total_correct, total_pixels)
        error_rate = 1.0 - pixel_acc

        tp = {c: 0 for c in CLASS_IDS}
        fp = {c: 0 for c in CLASS_IDS}
        fn = {c: 0 for c in CLASS_IDS}

        for class_id in CLASS_IDS:
            pred_c = pred_mask == class_id
            true_c = true_mask == class_id

            tp[class_id] += int((pred_c & true_c).sum())
            fp[class_id] += int((pred_c & (~true_c)).sum())
            fn[class_id] += int(((~pred_c) & true_c).sum())

        rows, macro_f1 = build_metric_rows(tp, fp, fn)

        original_image_path = test_images_dir / f"{stem}.jpg"
        image_rgb = np.array(Image.open(original_image_path).convert("RGB"))
        image_rgb = cv2.resize(
            image_rgb,
            (size, size),
            interpolation=cv2.INTER_LINEAR,
        )

        case = {
            "rank": -1,
            "stem": stem,
            "loss": float(total_loss.item()),
            "cls_loss": float(cls_loss),
            "dice_loss": float(dice_loss),
            "pixel_acc": float(pixel_acc),
            "error_rate": float(error_rate),
            "macro_f1": float(macro_f1),
            "rows": rows,
            "pred_mask": pred_mask,
            "true_mask": true_mask,
            "image_rgb": image_rgb,
        }
        cases.append(case)

    cases = sort_cases(cases, args.sort_by)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cases_csv = output_dir / "all_cases.csv"
    top_k_csv = output_dir / "top_k_cases.csv"

    save_cases_csv(cases, all_cases_csv)

    top_k = min(args.top_k, len(cases))
    hardest_cases = cases[:top_k]
    save_cases_csv(hardest_cases, top_k_csv)

    for case in hardest_cases:
        save_case_visuals(case, output_dir)

    print("\nAnalysis complete.")
    print(f"Sorted by: {args.sort_by}")
    print(f"Saved all cases CSV to: {all_cases_csv.resolve()}")
    print(f"Saved top-{top_k} CSV to: {top_k_csv.resolve()}")
    print(f"Saved top-{top_k} visualizations to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
