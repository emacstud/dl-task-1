# Task 1 - Semantic Segmentation

Author: Edvin Macel
LSP: 2515991
Selected classes: Eagle, Laptop, Dog

## Note
This is an updated version, submitted on 18.04.2026 with added benchmarking against LangSAM. The original submission contained only the mandatory task requirements was submitted before the deadline on 13.04.2026.


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
5. benchmarking a pre-trained **LangSAM** model on the same labeled test set,
6. inference on additional unseen images,
7. analysis of the hardest test cases.


---

## Project pipeline

The script must be run in the following order:

1. `download_dataset.py`
2. `convert_to_semantic.py`
3. `train.py`
4. `evaluate.py`
5. `benchmark_langsam.py`
6. `infer.py`
7. `analyze_hard_cases.py`

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

## 5. Benchmark LangSAM on the test set

### Script
```bash
python benchmark_langsam.py
```

### Purpose
Benchmarks a pre-trained **SAM-based model (LangSAM)** on the labeled semantic test split.

Unlike `evaluate.py`, this script does **not** use a model checkpoint trained in this project.  
Instead, it applies LangSAM with text prompts for the selected classes and converts the predictions into the same semantic mask format used throughout the project.

The script computes:
- pixel accuracy,
- precision,
- recall,
- F1 score,
- macro F1 for the selected classes.

### Input
- `data/openimages_semantic/test/images/`
- `data/openimages_semantic/test/masks/`

### Output
- `outputs/benchmark_langsam_metrics.csv`
- optionally, if `--save_visuals` is enabled:
  - `outputs/benchmark_langsam/inputs/`
  - `outputs/benchmark_langsam/gt_masks/`
  - `outputs/benchmark_langsam/gt_masks_rgb/`
  - `outputs/benchmark_langsam/gt_overlays/`
  - `outputs/benchmark_langsam/pred_masks/`
  - `outputs/benchmark_langsam/pred_masks_rgb/`
  - `outputs/benchmark_langsam/pred_overlays/`

### Arguments
| Argument | Default | Description |
|---|---:|---|
| `--data_root` | `data/openimages_semantic` | Semantic dataset root |
| `--size` | `384` | Input image size used for resizing test images and masks |
| `--out_csv` | `outputs/benchmark_langsam_metrics.csv` | Output CSV file with LangSAM benchmark metrics |
| `--output_dir` | `outputs/benchmark_langsam` | Directory for optional visual outputs |
| `--save_visuals` | `False` | If set, saves visual prediction results and ground truth comparisons |
| `--box_threshold` | `0.4` | Box confidence threshold used by LangSAM |
| `--text_threshold` | `0.3` | Text matching threshold used by LangSAM |

---

## 6. Run inference on additional unseen images

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

## 7. Analyze the hardest test cases

### Script
```bash
python analyze_hard_cases.py
```

### Purpose
Analyzes the most difficult test images for the trained model.  
For each test image, the script computes per-image:
- total loss,
- classification loss,
- Dice loss,
- pixel accuracy,
- error rate,
- macro F1,
- per-class precision, recall, and F1.

The images can be sorted by:
- highest loss,
- highest error rate,
- lowest macro F1.

For the top hardest cases, the script also saves:
- input image,
- ground-truth mask,
- predicted mask,
- RGB masks,
- overlays,
- residual / error map.

### Input
- `data/openimages_semantic/test/images/`
- `data/openimages_semantic/test/masks/`
- `data/openimages_semantic/train/masks/` (for class weights if needed)
- checkpoint file, by default: `checkpoints/best_model.pth`

### Output
- `outputs/hard_cases/all_cases.csv`
- `outputs/hard_cases/top_k_cases.csv`
- `outputs/hard_cases/rank_XX_<stem>/` folders containing:
  - `input.png`
  - `gt_mask.png`
  - `pred_mask.png`
  - `gt_mask_rgb.png`
  - `pred_mask_rgb.png`
  - `gt_overlay.png`
  - `pred_overlay.png`
  - `error_map.png`
  - `error_overlay.png`

### Arguments
| Argument | Default | Description |
|---|---:|---|
| `--data_root` | `data/openimages_semantic` | Semantic dataset root |
| `--checkpoint` | `checkpoints/best_model.pth` | Trained model checkpoint |
| `--output_dir` | `outputs/hard_cases` | Output directory for analysis results |
| `--top_k` | `10` | Number of hardest cases to save visually |
| `--sort_by` | `loss` | Sorting criterion (`loss`, `macro_f1`, `error_rate`) |
| `--expected_test_size` | `100` | Expected number of test images |

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
| `outputs/hard_cases/` | Hard case analysis CSV files and visualizations |

---

## Observations

Two training configurations were tested using the same backbone and preprocessing setup:

- **Model architecture:** U-Net with **ResNet18** encoder  
- **Input size:** **384 × 384**
- **Training length:** **30 epochs**
- **Compared losses:**
  - **Weighted Cross-Entropy + Dice**
  - **Focal + Dice**

### Final test-set results
Both models achieved **almost identical final macro F1** on the unseen 100-image test set:

- **Weighted CE + Dice:** macro F1 = **0.8818**
- **Focal + Dice:** macro F1 = **0.8821**

This shows that **both loss functions are valid choices** for this task and lead to very similar overall performance.

### Observations
Although the final macro F1 is nearly the same, there are some differences in behavior:

- **Focal + Dice** achieved slightly higher **pixel accuracy** on the test set.
- **Focal + Dice** performed better on **Eagle** and **Dog**.
- The difference in overall macro F1 is very small, so neither loss clearly dominates in all classes.

### Training behavior
The validation loss curves also show slightly different convergence behavior:

- For **Focal + Dice**, the **validation loss plateaued after approximately epoch 12**.
- For **Weighted CE + Dice**, the **validation loss plateaued after approximately epoch 16**.

This suggests that **Focal + Dice converged faster**, while **Weighted CE + Dice continued improving for a few more epochs** before stabilizing.

An additional U-Net configuration with a **ResNet34** encoder and **448 × 448** input transformations was trained for **20 epochs**. However, it also achieved a macro F1 of only **0.87**. On the other hand, the validation loss of this model had not yet reached a plateau, which suggests that further training epochs may still improve its performance.

---
