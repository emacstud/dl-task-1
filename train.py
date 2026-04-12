import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.config import (
    CHECKPOINTS_DIR,
    CLASS_TO_ID,
    DEFAULT_NUM_CLASSES,
    DEFAULT_SEED,
    SEMANTIC_ROOT,
    TRAIN_OUTPUTS_DIR
)
from lib.datasets import SemanticSegDataset
from lib.logging import append_history_csv, init_history_csv, plot_training_history
from lib.losses import compute_enet_class_weights
from lib.model import build_unet, save_checkpoint, train_one_epoch, validate_one_epoch
from lib.transforms import get_eval_transform, get_train_transform
from lib.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=SEMANTIC_ROOT)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--encoder", type=str, default="resnet18")
    parser.add_argument( "--encoder_weights", type=str, default="imagenet", choices=["none", "imagenet"])

    parser.add_argument("--learning_rate_reduce_patience", type=int, default=3)
    parser.add_argument("--learning_rate_reduce_factor", type=float, default=0.5)

    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=6)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    parser.add_argument("--save_dir", type=Path, default=CHECKPOINTS_DIR)

    parser.add_argument("--loss_mode", type=str, default="focal_dice", choices=["weighted_ce_dice", "focal_dice"])
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)

    parser.add_argument("--history_csv", type=Path, default=TRAIN_OUTPUTS_DIR / "training_history.csv")
    parser.add_argument("--plots_dir", type=Path, default=TRAIN_OUTPUTS_DIR)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print("Device:", device)

    data_root = Path(args.data_root)
    train_images = data_root / "train" / "images"
    train_masks = data_root / "train" / "masks"
    val_images = data_root / "val" / "images"
    val_masks = data_root / "val" / "masks"

    num_classes = DEFAULT_NUM_CLASSES

    train_ds = SemanticSegDataset(
        train_images,
        train_masks,
        transform=get_train_transform(args.size),
    )
    val_ds = SemanticSegDataset(
        val_images,
        val_masks,
        transform=get_eval_transform(args.size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    class_weights, class_counts, class_freqs = compute_enet_class_weights(train_masks, num_classes)
    print("Class counts:", class_counts.tolist())
    print("Class frequencies:", class_freqs.tolist())
    print("Class weights:", class_weights.tolist())

    ce_loss_fn = None

    if args.loss_mode == "weighted_ce_dice":
        ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Using loss: weighted CE + Dice")

    elif args.loss_mode == "focal_dice":
        print("Using loss: Focal + Dice")
        print("Focal gamma:", args.focal_gamma)

    print("Dice weight:", args.dice_weight)

    encoder_weights = None if args.encoder_weights == "none" else args.encoder_weights

    model = build_unet(
        encoder_name=args.encoder,
        encoder_weights=encoder_weights,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.learning_rate_reduce_factor,
        patience=args.learning_rate_reduce_patience,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history_csv = Path(args.history_csv)
    plots_dir = Path(args.plots_dir)
    init_history_csv(history_csv)
    print("Training history CSV:", history_csv.resolve())

    best_val_macro_f1 = -float("inf")
    best_path = save_dir / "best_model.pth"

    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs} (lr={current_lr:.2e})")

        train_loss, train_acc, _, train_macro_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            loss_mode=args.loss_mode,
            ce_loss_fn=ce_loss_fn,
            focal_gamma=args.focal_gamma,
            cls_weight=args.cls_weight,
            dice_weight=args.dice_weight,
        )

        val_loss, val_acc, val_rows, val_macro_f1 = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=num_classes,
            loss_mode=args.loss_mode,
            ce_loss_fn=ce_loss_fn,
            focal_gamma=args.focal_gamma,
            cls_weight=args.cls_weight,
            dice_weight=args.dice_weight,
        )

        print(f"train loss: {train_loss:.4f} | train pixel acc: {train_acc:.4f}")
        print(f"val loss: {val_loss:.4f} | val pixel acc: {val_acc:.4f}")
        print(f"train macro F1: {train_macro_f1:.4f}")
        print(f"val macro F1: {val_macro_f1:.4f}")

        for r in val_rows:
            print(
                f"{r['class_name']:10s} "
                f"P={r['precision']:.4f} "
                f"R={r['recall']:.4f} "
                f"F1={r['f1']:.4f}"
            )

        scheduler.step(val_macro_f1)
        improved = val_macro_f1 > (best_val_macro_f1 + args.early_stop_min_delta)

        # Save best model by highest validation macro F1
        if improved:
            best_val_macro_f1 = val_macro_f1
            epochs_without_improvement = 0

            save_checkpoint(
                checkpoint_path=best_path,
                model=model,
                encoder=args.encoder,
                encoder_weights_used=args.encoder_weights,
                num_classes=num_classes,
                size=args.size,
                loss_mode=args.loss_mode,
                focal_gamma=args.focal_gamma,
                cls_weight=args.cls_weight,
                dice_weight=args.dice_weight,
                best_val_macro_f1=best_val_macro_f1,
                class_mapping=CLASS_TO_ID,
            )
            print(
                f"Saved best model -> {best_path} "
                f"(best val macro F1 = {best_val_macro_f1:.4f})"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement in val macro F1 "
                f"for {epochs_without_improvement}/{args.early_stop_patience} epoch(s)"
            )

        append_history_csv(
            csv_path=history_csv,
            epoch=epoch,
            lr=current_lr,
            train_loss=train_loss,
            train_acc=train_acc,
            train_macro_f1=train_macro_f1,
            val_loss=val_loss,
            val_acc=val_acc,
            val_macro_f1=val_macro_f1,
            best_val_macro_f1=best_val_macro_f1,
            improved=improved,
            val_rows=val_rows,
        )

        if args.early_stop and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch}. "
                f"Best val macro F1: {best_val_macro_f1:.4f}"
            )
            break

    plot_training_history(history_csv, plots_dir)

    print("\nTraining complete.")
    print("Best checkpoint:", best_path.resolve())
    print(f"Best validation macro F1: {best_val_macro_f1:.4f}")

    
if __name__ == "__main__":
    main()