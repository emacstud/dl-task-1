"""Evaluate a trained segmentation model on the test set."""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from lib.config import CHECKPOINTS_DIR, EVALUATE_OUTPUTS_DIR, SEMANTIC_ROOT
from lib.datasets import SemanticSegDataset
from lib.metrics import save_metrics_csv
from lib.model import evaluate_model, load_model_from_checkpoint
from lib.transforms import get_eval_transform
from lib.utils import get_device, list_image_files, load_resized_rgb_image, reset_dir
from lib.visualization import save_evaluation_outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=SEMANTIC_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "best_focal_dice_resnet34_448_imagenet_20e.pth")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=EVALUATE_OUTPUTS_DIR)
    parser.add_argument("--out_csv", type=Path, default=Path("evaluation_metrics.csv"))
    parser.add_argument("--expected_test_size", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()
    print("Device:", device)

    output_dir = Path(args.output_dir)
    reset_dir(output_dir)

    model, ckpt = load_model_from_checkpoint(args.checkpoint, device)

    encoder = ckpt["encoder"]
    num_classes = ckpt["num_classes"]
    size = ckpt["size"]

    print("Loaded checkpoint:")
    print("encoder:", encoder)
    print("num_classes:", num_classes)
    print("input size:", size)

    test_images = Path(args.data_root) / "test" / "images"
    test_masks = Path(args.data_root) / "test" / "masks"

    image_paths = list_image_files(test_images)
    stem_to_index = {p.stem: i for i, p in enumerate(image_paths, start=1)}
    stem_to_path = {p.stem: p for p in image_paths}

    test_ds = SemanticSegDataset(
        test_images,
        test_masks,
        transform=get_eval_transform(size),
        return_stem=True,
    )

    print("Test samples:", len(test_ds))
    if len(test_ds) != args.expected_test_size:
        print("WARNING: expected 100 unseen test images, found", len(test_ds))

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    def on_sample(pred_mask, true_mask, stem: str):
        index = stem_to_index[stem]
        image_rgb = load_resized_rgb_image(stem_to_path[stem], size)
        save_evaluation_outputs(
            index=index,
            image_rgb=image_rgb,
            pred_mask=pred_mask,
            true_mask=true_mask,
            out_dir=output_dir,
        )

    pixel_acc, rows, macro_f1 = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        on_sample=on_sample,
        desc="Evaluating",
    )

    print("\n=== TEST RESULTS ===")
    print(f"Overall pixel accuracy: {pixel_acc:.6f}")
    print(f"Macro F1: {macro_f1:.6f}\n")

    print(f"{'Class':10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    for r in rows:
        print(f"{r['class_name']:10s} {r['precision']:10.4f} {r['recall']:10.4f} {r['f1']:10.4f}")

    save_metrics_csv(args.out_csv, pixel_acc, macro_f1, rows)
    print(f"\nSaved metrics to: {args.out_csv}")


if __name__ == "__main__":
    main()
