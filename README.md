# Task 1 - Semantic Segmentation

Author: Edvin Macel
LSP: 2515991
Selected classes: Eagle, Laptop, Dog


## Task description
This project implements a semantic image segmentation pipeline for OpenImages.  
The model classifies each pixel into **4 classes**:

- `0` - background
- `1` - Eagle
- `2` - Laptop
- `3` - Dog


The project includes:
1. dataset download and split creation,
2. conversion from instance annotations to semantic masks,
3. model training,
4. evaluation on **100 unseen test images**,
5. inference on additional unseen images.


---

## Project pipeline

The script must be run in the following order:

1. `download_dataset.py`
2. `convert_to_semantic.py`
3. `train.py`
4. `evaluate.py`
5. `infer.py`

---

## 1. Download dataset

### Script
```bash
python download_dataset.py
```

### Purpose
Downloads OpenImages samples for the selected classes and creates:
- training split,
- validation split,
- test split,
- additional unseen images for inference demo.

### Input
- OpenImages dataset loaded through FiftyOne Zoo

### Output
- `data/openimages_coco/train/`
- `data/openimages_coco/val/`
- `data/openimages_coco/test/`
- `data/unseen/`

### Arguments
| Argument | Default | Description |
|---|---:|---|
| `--seed` | `42` | Random seed |
| `--train_samples` | `1400` | Number of training samples |
| `--val_samples` | `300` | Number of validation samples |
| `--test_samples` | `100` | Number of test samples |
| `--unseen_samples` | `100` | Number of additional unseen images for inference |

---

## 2. Convert COCO annotations to semantic segmentation masks

### Script
```bash
python convert_to_semantic.py
```

### Purpose
Converts the exported COCO-style dataset into a semantic segmentation dataset with:
- RGB images,
- integer class masks,
- RGB masks for visualization,
- overlay images.

### Input
- `data/openimages_coco/`

### Output
- `data/openimages_semantic/train/images/`
- `data/openimages_semantic/train/masks/`
- `data/openimages_semantic/train/masks_rgb/`
- `data/openimages_semantic/train/overlays/`
- `data/openimages_semantic/val/images/`
- `data/openimages_semantic/val/masks/`
- `data/openimages_semantic/val/masks_rgb/`
- `data/openimages_semantic/val/overlays/`
- `data/openimages_semantic/test/images/`
- `data/openimages_semantic/test/masks/`
- `data/openimages_semantic/test/masks_rgb/`
- `data/openimages_semantic/test/overlays/`

### Arguments
| Argument | Default | Description |
|---|---:|---|
| `--coco_root` | `data/openimages_coco` | Input COCO dataset root |
| `--output_root` | `data/openimages_semantic` | Output semantic dataset root |

---

## 3. Train the segmentation model

### Script
```bash
python train.py
```

### Purpose
Trains a **U-Net** semantic segmentation model on the semantic dataset.  
The best checkpoint is selected using **validation macro F1**.

### Input
- `data/openimages_semantic/train/images/`
- `data/openimages_semantic/train/masks/`
- `data/openimages_semantic/val/images/`
- `data/openimages_semantic/val/masks/`

### Output
- `checkpoints/best_model.pth`
- `outputs_train/training_history.csv`
- training plots saved in `outputs_train/`

### Arguments
| Argument | Default | Description |
|---|---:|---|
| `--data_root` | `data/openimages_semantic` | Semantic dataset root |
| `--size` | `384` | Input image size |
| `--epochs` | `25` | Number of epochs |
| `--batch_size` | `4` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--seed` | `42` | Random seed |
| `--num_workers` | `0` | Number of DataLoader workers |
| `--encoder` | `resnet18` | U-Net encoder backbone |
| `--encoder_weights` | `imagenet` | Encoder pretrained weights (`none` or `imagenet`) |
| `--learning_rate_reduce_patience` | `3` | Patience for LR scheduler |
| `--learning_rate_reduce_factor` | `0.5` | LR reduction factor |
| `--early_stop` | `False` | Enable early stopping |
| `--early_stop_patience` | `6` | Early stopping patience |
| `--early_stop_min_delta` | `1e-4` | Minimum improvement for early stopping |
| `--save_dir` | `checkpoints` | Directory for model checkpoints |
| `--loss_mode` | `focal_dice` | Loss mode (`weighted_ce_dice` or `focal_dice`) |
| `--focal_gamma` | `2.0` | Gamma for focal loss |
| `--cls_weight` | `1.0` | Weight of classification loss term |
| `--dice_weight` | `1.0` | Weight of Dice loss term |
| `--history_csv` | `outputs_train/training_history.csv` | CSV file with training history |
| `--plots_dir` | `outputs_train` | Directory for training plots |

---

## 4. Evaluate on the test set

### Script
```bash
python evaluate.py
```

### Purpose
Evaluates the best trained model on the test split (**100 unseen images**).  
The script computes:
- pixel accuracy,
- precision,
- recall,
- F1 score,
- macro F1 for the selected classes.

### Input
- `data/openimages_semantic/test/images/`
- `data/openimages_semantic/test/masks/`
- checkpoint file, by default: `checkpoints/best_model.pth`

### Output
- `evaluation_metrics.csv`

### Arguments
| Argument | Default | Description |
|---|---:|---|
| `--data_root` | `data/openimages_semantic` | Semantic dataset root |
| `--checkpoint` | `checkpoints/best_model.pth` | Trained model checkpoint |
| `--batch_size` | `4` | Batch size |
| `--num_workers` | `0` | Number of DataLoader workers |
| `--out_csv` | `evaluation_metrics.csv` | Output CSV with evaluation metrics |
| `--expected_test_size` | `100` | Expected number of test images |

---

## 5. Run inference on additional unseen images

### Script
```bash
python infer.py
```

### Purpose
Runs the trained model on unseen images without labels and saves predicted masks and overlays.

### Input
- images from `data/unseen/`
- checkpoint file, by default: `checkpoints/best_model.pth`

### Output
- `outputs/pred_masks/`
- `outputs/pred_masks_rgb/`
- `outputs/pred_overlays/`
- optionally: `outputs/resized_inputs/`

### Arguments
| Argument | Default | Description |
|---|---:|---|
| `--checkpoint` | `checkpoints/best_model.pth` | Trained model checkpoint |
| `--input` | `data/unseen` | Input directory with unseen images |
| `--output_dir` | `outputs` | Output directory |
| `--save_resized_input` | `False` | If set, also saves resized input images |

---


## Main folders

| Folder | Description |
|---|---|
| `data/openimages_coco/` | COCO-style exported dataset |
| `data/openimages_semantic/` | Semantic segmentation dataset |
| `data/unseen/` | Additional unseen images for inference |
| `checkpoints/` | Saved trained models |
| `outputs_train/` | Training CSV history and plots |
| `outputs/` | Inference outputs |

---