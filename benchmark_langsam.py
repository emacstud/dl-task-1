"""Benchmark a pre-trained SAM-based model (LangSAM) on the labeled test set."""

import argparse
from pathlib import Path

from lang_sam import LangSAM
from tqdm import tqdm

from lib.config import OUTPUTS_DIR, SEMANTIC_ROOT
from lib.langsam import predict_langsam_semantic_mask
from lib.metrics import (
    build_metric_rows,
    init_stat_dicts,
    save_metrics_csv,
    update_stat_dicts,
)
from lib.utils import (
    get_device,
    list_image_files,
    load_resized_mask,
    load_resized_rgb_image,
    reset_dir,
    safe_div,
)
from lib.visualization import save_evaluation_outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=SEMANTIC_ROOT)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--out_csv", type=Path, default=OUTPUTS_DIR / "benchmark_langsam_metrics.csv")
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS_DIR / "benchmark_langsam")
    parser.add_argument("--save_visuals", action="store_true")
    parser.add_argument("--box_threshold", type=float, default=0.4)
    parser.add_argument("--text_threshold", type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()
    print("Device:", device)

    test_images_dir = Path(args.data_root) / "test" / "images"
    test_masks_dir = Path(args.data_root) / "test" / "masks"

    image_paths = list_image_files(test_images_dir)
    stem_to_index = {p.stem: i for i, p in enumerate(image_paths, start=1)}
    print("Test samples:", len(image_paths))

    output_dir = Path(args.output_dir)
    if args.save_visuals:
        reset_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    model = LangSAM()

    tp, fp, fn = init_stat_dicts()

    total_correct = 0
    total_pixels = 0

    for image_path in tqdm(image_paths, desc="Benchmarking LangSAM"):
        index = stem_to_index[image_path.stem]

        mask_path = test_masks_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        image_rgb = load_resized_rgb_image(image_path, args.size)
        true_mask = load_resized_mask(mask_path, args.size)

        pred_mask = predict_langsam_semantic_mask(
            model=model,
            image_rgb=image_rgb,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

        total_correct += int((pred_mask == true_mask).sum())
        total_pixels += int(true_mask.size)

        update_stat_dicts(pred_mask, true_mask, tp, fp, fn)

        if args.save_visuals:
            save_evaluation_outputs(
                index=index,
                image_rgb=image_rgb,
                pred_mask=pred_mask,
                true_mask=true_mask,
                out_dir=output_dir,
            )

    pixel_acc = safe_div(total_correct, total_pixels)
    rows, macro_f1 = build_metric_rows(tp, fp, fn)

    print("\n=== LANGSAM BENCHMARK RESULTS ===")
    print(f"Overall pixel accuracy: {pixel_acc:.6f}")
    print(f"Macro F1: {macro_f1:.6f}")

    print(f"\n{'Class':10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    for r in rows:
        print(
            f"{r['class_name']:10s} "
            f"{r['precision']:10.4f} "
            f"{r['recall']:10.4f} "
            f"{r['f1']:10.4f}"
        )

    save_metrics_csv(args.out_csv, pixel_acc, macro_f1, rows)
    print(f"\nSaved metrics to: {args.out_csv.resolve()}")

    if args.save_visuals:
        print(f"Saved visual outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
