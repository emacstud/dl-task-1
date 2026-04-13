"""Download and split the OpenImages dataset."""

import argparse

from lib.config import CLASSES, COCO_ROOT, DEFAULT_SEED, UNSEEN_ROOT
from lib.openimages import (
    export_coco_dataset,
    export_images_only,
    load_openimages_split,
)
from lib.utils import reset_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train_samples", type=int, default=1400)
    parser.add_argument("--val_samples", type=int, default=300)
    parser.add_argument("--test_samples", type=int, default=100)
    parser.add_argument("--unseen_samples", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    train_total = args.train_samples + args.unseen_samples
    train_ds = load_openimages_split(
        split="train",
        classes=CLASSES,
        max_samples=train_total,
        seed=args.seed,
    )

    shuffled_train = train_ds.shuffle(seed=args.seed)
    train_view = shuffled_train.limit(args.train_samples)
    unseen_view = shuffled_train.skip(args.train_samples).limit(args.unseen_samples)

    print(f"Train samples: {len(train_view)}")
    print(f"Unseen samples: {len(unseen_view)}")

    train_path = COCO_ROOT / "train"

    reset_dir(train_path)
    reset_dir(UNSEEN_ROOT)

    export_coco_dataset(train_view, train_path, CLASSES)
    export_images_only(unseen_view, UNSEEN_ROOT)

    val_total = args.val_samples + args.test_samples
    val_ds = load_openimages_split(
        split="validation",
        classes=CLASSES,
        max_samples=val_total,
        seed=args.seed,
    )

    shuffled_val = val_ds.shuffle(seed=args.seed)
    test_view = shuffled_val.limit(args.test_samples)
    val_view = shuffled_val.skip(args.test_samples).limit(args.val_samples)

    print(f"Validation samples: {len(val_view)}")
    print(f"Test samples: {len(test_view)}")

    val_path = COCO_ROOT / "val"
    test_path = COCO_ROOT / "test"

    reset_dir(val_path)
    reset_dir(test_path)

    export_coco_dataset(val_view, val_path, CLASSES)
    export_coco_dataset(test_view, test_path, CLASSES)

    print("\nExport complete.")
    print("Next step: run convert_to_semantic.py")


if __name__ == "__main__":
    main()
