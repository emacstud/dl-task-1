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