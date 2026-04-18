"""Run inference on unseen images."""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lib.config import CHECKPOINTS_DIR, ID_TO_CLASS, INFER_OUTPUTS_DIR, UNSEEN_ROOT
from lib.model import load_model_from_checkpoint, predict_mask
from lib.transforms import get_infer_transform
from lib.utils import (
    get_device,
    list_image_files,
    load_rgb_image,
    reset_dir,
    resize_rgb_image,
)
from lib.visualization import save_prediction_outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "best_focal_dice_resnet34_448_imagenet_20e.pth")
    parser.add_argument("--input", type=Path, default=UNSEEN_ROOT)
    parser.add_argument("--output_dir", type=Path, default=INFER_OUTPUTS_DIR)
    parser.add_argument("--save_resized_input", action="store_true")
    return parser.parse_args()


def print_prediction_summary(pred_mask: np.ndarray):
    unique_ids, counts = np.unique(pred_mask, return_counts=True)
    total = pred_mask.size

    print("Predicted classes:")
    for class_id, count in zip(unique_ids.tolist(), counts.tolist()):
        class_name = ID_TO_CLASS.get(class_id, f"class_{class_id}")
        frac = count / total
        print(f"{class_name:10s} pixels={count:8d} ({frac:.2%})")


def main():
    args = parse_args()

    device = get_device()
    print("Device:", device)

    model, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    size = ckpt["size"]
    transform = get_infer_transform(size=size)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    image_paths = list_image_files(input_path)
    stem_to_index = {p.stem: i for i, p in enumerate(image_paths, start=1)}
    out_dir = Path(args.output_dir)
    reset_dir(out_dir)

    resized_inputs_dir = out_dir / "resized_inputs"
    if args.save_resized_input:
        resized_inputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(image_paths)} image(s)")

    for image_path in tqdm(image_paths, desc="Inference"):
        index = stem_to_index[image_path.stem]
        image_rgb = load_rgb_image(image_path)

        pred_mask = predict_mask(
            model=model,
            image_rgb=image_rgb,
            transform=transform,
            device=device,
        )

        resized_image = resize_rgb_image(image_rgb, size)

        save_prediction_outputs(
            index=index,
            image_rgb=resized_image,
            pred_mask=pred_mask,
            out_dir=out_dir,
            save_input=args.save_resized_input,
            input_dir_name="resized_inputs",
        )

        print(f"\nImage {index}: {image_path.name}")
        print_prediction_summary(pred_mask)

        print(f"\nImage: {image_path.name}")
        print_prediction_summary(pred_mask)

    print("\nInference complete.")
    print("Saved outputs to:", out_dir.resolve())
    print(" - pred_masks: raw class-id masks")
    print(" - pred_masks_rgb: colored masks")
    print(" - pred_overlays: mask overlays")


if __name__ == "__main__":
    main()
