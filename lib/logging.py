"""Utilities for training logs and plots."""

import csv
from pathlib import Path

from matplotlib import pyplot as plt

from lib.config import CLASS_IDS, ID_TO_CLASS


def build_history_fieldnames() -> list[str]:
    """Build CSV column names for training history."""
    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "train_pixel_acc",
        "train_macro_f1",
        "val_loss",
        "val_pixel_acc",
        "val_macro_f1",
        "best_val_macro_f1",
        "improved",
    ]

    for class_id in CLASS_IDS:
        class_name = ID_TO_CLASS[class_id]
        fieldnames.extend([
            f"val_{class_name}_TP",
            f"val_{class_name}_FP",
            f"val_{class_name}_FN",
            f"val_{class_name}_precision",
            f"val_{class_name}_recall",
            f"val_{class_name}_f1",
        ])

    return fieldnames


def init_history_csv(csv_path: Path) -> None:
    """Create the training history CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=build_history_fieldnames())
        writer.writeheader()


def append_history_csv(
    csv_path: Path,
    epoch: int,
    lr: float,
    train_loss: float,
    train_acc: float,
    train_macro_f1: float,
    val_loss: float,
    val_acc: float,
    val_macro_f1: float,
    best_val_macro_f1: float,
    improved: bool,
    val_rows: list[dict],
) -> None:
    """Append one epoch of metrics to the history CSV."""
    row = {
        "epoch": epoch,
        "lr": lr,
        "train_loss": train_loss,
        "train_pixel_acc": train_acc,
        "train_macro_f1": train_macro_f1,
        "val_loss": val_loss,
        "val_pixel_acc": val_acc,
        "val_macro_f1": val_macro_f1,
        "best_val_macro_f1": best_val_macro_f1,
        "improved": int(improved),
    }

    for class_row in val_rows:
        class_name = class_row["class_name"]
        row[f"val_{class_name}_TP"] = class_row["TP"]
        row[f"val_{class_name}_FP"] = class_row["FP"]
        row[f"val_{class_name}_FN"] = class_row["FN"]
        row[f"val_{class_name}_precision"] = class_row["precision"]
        row[f"val_{class_name}_recall"] = class_row["recall"]
        row[f"val_{class_name}_f1"] = class_row["f1"]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=build_history_fieldnames())
        writer.writerow(row)


def save_line_plot(
    epochs: list[int],
    curves: dict[str, list[float]],
    title: str,
    ylabel: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Saves a line plot."""
    plt.figure(figsize=(8, 5))

    for label, values in curves.items():
        plt.plot(epochs, values, marker="o", linewidth=2, label=label)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)

    if ylim is not None:
        plt.ylim(*ylim)

    if len(curves) > 1:
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_training_history(csv_path: Path, plots_dir: Path) -> None:
    """Generate plots from the training history CSV."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("History CSV is empty, skipping plot generation.")
        return

    epochs = [int(row["epoch"]) for row in rows]

    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    train_acc = [float(row["train_pixel_acc"]) for row in rows]
    val_acc = [float(row["val_pixel_acc"]) for row in rows]
    val_macro_f1 = [float(row["val_macro_f1"]) for row in rows]
    lrs = [float(row["lr"]) for row in rows]

    save_line_plot(
        epochs=epochs,
        curves={
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        title="Training and Validation Loss",
        ylabel="Loss",
        out_path=plots_dir / "loss.png",
    )

    save_line_plot(
        epochs=epochs,
        curves={
            "train_pixel_acc": train_acc,
            "val_pixel_acc": val_acc,
            "val_macro_f1": val_macro_f1,
        },
        title="Accuracy and Validation Macro F1",
        ylabel="Score",
        out_path=plots_dir / "accuracy_macro_f1.png",
        ylim=(0.0, 1.0),
    )

    save_line_plot(
        epochs=epochs,
        curves={
            "learning_rate": lrs,
        },
        title="Learning Rate by Epoch",
        ylabel="Learning Rate",
        out_path=plots_dir / "learning_rate.png",
    )

    precision_curves: dict[str, list[float]] = {}
    recall_curves: dict[str, list[float]] = {}
    f1_curves: dict[str, list[float]] = {}

    for class_id in CLASS_IDS:
        class_name = ID_TO_CLASS[class_id]

        precision_curves[class_name] = [
            float(row[f"val_{class_name}_precision"]) for row in rows
        ]
        recall_curves[class_name] = [
            float(row[f"val_{class_name}_recall"]) for row in rows
        ]
        f1_curves[class_name] = [
            float(row[f"val_{class_name}_f1"]) for row in rows
        ]

    save_line_plot(
        epochs=epochs,
        curves=precision_curves,
        title="Validation Precision per Class",
        ylabel="Precision",
        out_path=plots_dir / "val_precision_per_class.png",
        ylim=(0.0, 1.0),
    )

    save_line_plot(
        epochs=epochs,
        curves=recall_curves,
        title="Validation Recall per Class",
        ylabel="Recall",
        out_path=plots_dir / "val_recall_per_class.png",
        ylim=(0.0, 1.0),
    )

    save_line_plot(
        epochs=epochs,
        curves=f1_curves,
        title="Validation F1 per Class",
        ylabel="F1",
        out_path=plots_dir / "val_f1_per_class.png",
        ylim=(0.0, 1.0),
    )
