from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def multiclass_dice_loss(logits, targets, num_classes: int, ignore_background: bool = True, eps: float = 1e-6):
    probs = torch.softmax(logits, dim=1)

    targets_oh = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    if ignore_background:
        probs = probs[:, 1:, :, :]
        targets_oh = targets_oh[:, 1:, :, :]

    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets_oh, dims)
    union = torch.sum(probs + targets_oh, dims)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def multiclass_focal_loss(logits, targets, gamma=2.0, reduction="mean"):
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    targets_unsq = targets.unsqueeze(1)

    log_pt = torch.gather(log_probs, 1, targets_unsq).squeeze(1)
    pt = torch.gather(probs, 1, targets_unsq).squeeze(1)

    focal_factor = (1.0 - pt) ** gamma
    loss = -focal_factor * log_pt

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def compute_enet_class_weights(masks_dir: Path, num_classes: int, c: float = 1.02):
    counts = np.zeros(num_classes, dtype=np.int64)

    for path in tqdm(sorted(Path(masks_dir).glob("*.png"))):
        mask = np.array(Image.open(path), dtype=np.int64)
        counts += np.bincount(mask.reshape(-1), minlength=num_classes)[:num_classes]

    freqs = counts / counts.sum()
    weights = 1.0 / np.log(c + freqs)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32), counts, freqs


def compute_total_loss(
    logits,
    masks,
    num_classes,
    loss_mode,
    ce_loss_fn=None,
    focal_gamma=2.0,
    cls_weight=1.0,
    dice_weight=1.0,
):
    if loss_mode == "weighted_ce_dice":
        if ce_loss_fn is None:
            raise ValueError("ce_loss_fn must be provided for weighted_ce_dice")
        cls_loss = ce_loss_fn(logits, masks)
    elif loss_mode == "focal_dice":
        cls_loss = multiclass_focal_loss(logits, masks, gamma=focal_gamma, reduction="mean")
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    dice = multiclass_dice_loss(logits, masks, num_classes=num_classes, ignore_background=True)
    total = cls_weight * cls_loss + dice_weight * dice
    return total, cls_loss.detach().item(), dice.detach().item()
