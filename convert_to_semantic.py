"""Convert COCO-style annotations to semantic segmentation masks."""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from lib.config import CLASS_TO_ID, COCO_ROOT, SEMANTIC_ROOT
from lib.openimages import build_semantic_mask, find_image_path
from lib.utils import reset_dir
from lib.visualization import make_overlay, mask_to_rgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", type=Path, default=COCO_ROOT)
    parser.add_argument("--output_root", type=Path, default=SEMANTIC_ROOT)
    return parser.parse_args()


def convert_split(split: str, coco_root: Path, output_root: Path):
    split_dir = coco_root / split
    labels_path = split_dir / "labels.json"

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    coco = COCO(str(labels_path))

    cats = coco.loadCats(coco.getCatIds())
    catid_to_name = {cat["id"]: cat["name"] for cat in cats}

    out_img_dir = output_root / split / "images"
    out_mask_dir = output_root / split / "masks"
    out_rgb_dir = output_root / split / "masks_rgb"
    out_overlay_dir = output_root / split / "overlays"

    for directory in (out_img_dir, out_mask_dir, out_rgb_dir, out_overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_ids = coco.getImgIds()

    for image_id in tqdm(image_ids, desc=f"Converting {split}"):
        info = coco.loadImgs([image_id])[0]
        file_name = info["file_name"]

        src_img_path = find_image_path(split_dir, file_name)
        stem = str(image_id)

        dst_img_path = out_img_dir / f"{stem}.jpg"
        dst_mask_path = out_mask_dir / f"{stem}.png"
        dst_rgb_path = out_rgb_dir / f"{stem}.png"
        dst_overlay_path = out_overlay_dir / f"{stem}.png"

        shutil.copy(src_img_path, dst_img_path)

        image_rgb = np.array(Image.open(src_img_path).convert("RGB"))

        semantic_mask = build_semantic_mask(coco, image_id, catid_to_name)
        Image.fromarray(semantic_mask).save(dst_mask_path)

        rgb_mask = mask_to_rgb(semantic_mask)
        Image.fromarray(rgb_mask).save(dst_rgb_path)

        overlay = make_overlay(image_rgb, rgb_mask)
        Image.fromarray(overlay).save(dst_overlay_path)


def main():
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        split_output_dir = output_root / split
        reset_dir(split_output_dir)
        convert_split(split, Path(args.coco_root), output_root)

    print("\nConversion complete.")
    print(f"Semantic dataset root: {output_root.resolve()}")
    print(f"Class mapping: {CLASS_TO_ID}")


if __name__ == "__main__":
    main()