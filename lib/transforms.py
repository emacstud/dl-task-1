import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


def get_train_transform(size: int):
    return A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.85, 1.15),
            translate_percent=(-0.05, 0.05),
            rotate=(-10, 10),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2(),
    ])


def get_eval_transform(size: int):
    return A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2(),
    ])


def get_infer_transform(size: int):
    return A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2(),
    ])
