import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from lib.model import predict_mask
from lib.transforms import get_infer_transform
from lib.config import CHECKPOINTS_DIR, ID_TO_CLASS, OUTPUTS_DIR, UNSEEN_ROOT
from lib.utils import get_device
from lib.visualization import mask_to_rgb, make_overlay
from lib.model import load_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default=CHECKPOINTS_DIR / "best_model.pth")
    parser.add_argument("--input", type=str, default=UNSEEN_ROOT)
    parser.add_argument("--output_dir", type=str, default=OUTPUTS_DIR)
    parser.add_argument("--save_resized_input", action="store_true")

    return parser.parse_args()


def save_outputs(image_path: Path, image_rgb: np.ndarray, pred_mask: np.ndarray, out_dir: Path):
    stem = image_path.stem

    masks_dir = out_dir / "pred_masks"
    masks_rgb_dir = out_dir / "pred_masks_rgb"
    overlays_dir = out_dir / "pred_overlays"

    masks_dir.mkdir(parents=True, exist_ok=True)
    masks_rgb_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    pred_rgb = mask_to_rgb(pred_mask)
    overlay = make_overlay(image_rgb, pred_rgb)

    Image.fromarray(pred_mask).save(masks_dir / f"{stem}.png")
    Image.fromarray(pred_rgb).save(masks_rgb_dir / f"{stem}.png")
    Image.fromarray(overlay).save(overlays_dir / f"{stem}.png")


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

    image_paths = [p for p in input_path.iterdir()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resized_inputs_dir = out_dir / "resized_inputs"
    if args.save_resized_input:
        resized_inputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(image_paths)} image(s)")

    for image_path in tqdm(image_paths, desc="Inference"):
        image_rgb = np.array(Image.open(image_path).convert("RGB"))

        pred_mask = predict_mask(
            model=model,
            image_rgb=image_rgb,
            transform=transform,
            device=device,
        )

        resized_image = cv2.resize(
            image_rgb,
            (size, size),
            interpolation=cv2.INTER_LINEAR,
        )

        save_outputs(
            image_path=image_path,
            image_rgb=resized_image,
            pred_mask=pred_mask,
            out_dir=out_dir,
        )

        if args.save_resized_input:
            Image.fromarray(resized_image).save(resized_inputs_dir / image_path.name)

        print(f"\nImage: {image_path.name}")
        print_prediction_summary(pred_mask)

    print("\nInference complete.")
    print("Saved outputs to:", out_dir.resolve())
    print(" - pred_masks      : raw class-id masks")
    print(" - pred_masks_rgb  : colored masks")
    print(" - pred_overlays   : mask overlays")


if __name__ == "__main__":
    main()
