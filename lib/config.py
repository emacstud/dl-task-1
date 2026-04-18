"""Project configuration constants."""

from pathlib import Path

CLASSES = ["Eagle", "Laptop", "Dog"]

CLASS_TO_ID = {
    "background": 0,
    "Eagle": 1,
    "Laptop": 2,
    "Dog": 3,
}

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

CLASS_IDS = [1, 2, 3]

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
}

PRIORITY = {
    "Eagle": 1,
    "Laptop": 2,
    "Dog": 3,
}

LSAM_CLASS_PROMPTS = {
    "Eagle": "eagle",
    "Laptop": "laptop",
    "Dog": "dog",
}

DEFAULT_SEED = 42
DEFAULT_NUM_CLASSES = len(CLASS_TO_ID.keys())

DATA_DIR = Path("data")
COCO_ROOT = DATA_DIR / "openimages_coco"
SEMANTIC_ROOT = DATA_DIR / "openimages_semantic"
UNSEEN_ROOT = DATA_DIR / "unseen"

CHECKPOINTS_DIR = Path("checkpoints")
OUTPUTS_DIR = Path("outputs")
TRAIN_OUTPUTS_DIR = Path("outputs_train")

EVALUATE_OUTPUTS_DIR = OUTPUTS_DIR / "evaluate"
INFER_OUTPUTS_DIR = OUTPUTS_DIR / "infer"
BENCHMARK_LANGSAM_OUTPUTS_DIR = OUTPUTS_DIR / "benchmark_langsam"