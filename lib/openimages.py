"""Utilities for loading and exporting OpenImages data."""

import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
from fiftyone import Dataset
from fiftyone import ViewField as F
from pycocotools.coco import COCO

from .config import CLASS_TO_ID, PRIORITY


def load_openimages_split(split: str, classes: list[str], max_samples: int, seed: int) -> Dataset:
    """Load an OpenImages split from FiftyOne Zoo."""
    return foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["segmentations"],
        classes=classes,
        only_matching=True,
        max_samples=max_samples,
        shuffle=True,
        seed=seed,
    )


def export_coco_dataset(ds: Dataset, export_dir: Path, classes: list[str], label_field: str = "ground_truth") -> None:
    """Export selected labels in COCO format."""
    export_dir = Path(export_dir)

    view = ds.filter_labels(label_field, F("label").is_in(classes))
    view.export(
        export_dir=str(export_dir),
        dataset_type=fo.types.COCODetectionDataset, # type: ignore
        label_field=label_field,
    )


def export_images_only(ds: Dataset, export_dir: Path) -> None:
    """Export images without annotations."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    for sample in ds:
        src = Path(sample.filepath)
        dst = export_dir / src.name

        if dst.exists():
            dst = export_dir / f"{sample.id}_{src.name}"

        shutil.copy(src, dst)


def find_image_path(split_dir: Path, file_name: str) -> Path:
    """Find an exported image path in a split directory."""
    direct_path = split_dir / file_name
    if direct_path.exists():
        return direct_path

    nested_path = split_dir / "data" / Path(file_name).name
    if nested_path.exists():
        return nested_path

    raise FileNotFoundError(f"Could not locate image for file_name={file_name}")


def build_semantic_mask(coco: COCO, image_id: int, catid_to_name: dict[int, str]) -> np.ndarray:
    """Build a semantic mask for one image."""
    img_info = coco.loadImgs([image_id])[0]
    height, width = img_info["height"], img_info["width"]

    sem = np.zeros((height, width), dtype=np.uint8)

    ann_ids = coco.getAnnIds(imgIds=[image_id])
    anns = coco.loadAnns(ann_ids)

    selected_anns = []
    for ann in anns:
        class_name = catid_to_name.get(ann["category_id"])
        if class_name in PRIORITY:
            selected_anns.append((PRIORITY[class_name], class_name, ann))

    selected_anns.sort(key=lambda x: x[0])

    for _, class_name, ann in selected_anns:
        mask = coco.annToMask(ann)
        sem[mask == 1] = CLASS_TO_ID[class_name]

    return sem