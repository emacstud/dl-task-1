from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SemanticSegDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, transform=None, return_stem: bool = False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.return_stem = return_stem

        self.image_paths = sorted(self.images_dir.glob("*.jpg"))
        if not self.image_paths:
            raise RuntimeError(f"No .jpg files found in {self.images_dir}")

        for path in self.image_paths[:10]:
            mask_path = self.masks_dir / f"{path.stem}.png"
            if not mask_path.exists():
                raise RuntimeError(f"Missing mask for {path.name}: expected {mask_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.masks_dir / f"{img_path.stem}.png"

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask)

        mask = mask.long()

        if self.return_stem:
            return image, mask, img_path.stem
        return image, mask
