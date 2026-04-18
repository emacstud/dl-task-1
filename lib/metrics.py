import csv
from pathlib import Path

import numpy as np

from .config import CLASS_IDS, ID_TO_CLASS
from .utils import safe_div


def build_metric_rows(tp, fp, fn):
    rows = []
    f1s = []

    for class_id in CLASS_IDS:
        precision = safe_div(tp[class_id], tp[class_id] + fp[class_id])
        recall = safe_div(tp[class_id], tp[class_id] + fn[class_id])
        f1 = safe_div(2 * precision * recall, precision + recall)

        rows.append({
            "class_id": class_id,
            "class_name": ID_TO_CLASS[class_id],
            "TP": tp[class_id],
            "FP": fp[class_id],
            "FN": fn[class_id],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    return rows, macro_f1


def save_metrics_csv(out_csv: Path, pixel_acc: float, macro_f1: float, rows: list[dict]):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["pixel_accuracy", pixel_acc])
        writer.writerow(["macro_f1_selected_classes", macro_f1])

        writer.writerow([])
        writer.writerow(["class_name", "TP", "FP", "FN", "precision", "recall", "f1"])
        for row in rows:
            writer.writerow([
                row["class_name"],
                row["TP"],
                row["FP"],
                row["FN"],
                row["precision"],
                row["recall"],
                row["f1"],
            ])


def init_stat_dicts() -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Initialize TP/FP/FN dictionaries for selected classes."""
    tp = {c: 0 for c in CLASS_IDS}
    fp = {c: 0 for c in CLASS_IDS}
    fn = {c: 0 for c in CLASS_IDS}
    return tp, fp, fn


def update_stat_dicts(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    tp: dict[int, int],
    fp: dict[int, int],
    fn: dict[int, int],
) -> None:
    """Update TP/FP/FN counts for one predicted/ground-truth mask pair."""
    for class_id in CLASS_IDS:
        pred_c = pred_mask == class_id
        true_c = true_mask == class_id

        tp[class_id] += int((pred_c & true_c).sum())
        fp[class_id] += int((pred_c & (~true_c)).sum())
        fn[class_id] += int(((~pred_c) & true_c).sum())