# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CNN-based deepfake image classification using PyTorch and timm. Binary classification (0=real, 1=fake) with support for various pretrained architectures (xception, resnet, efficientnet, mobilenet, etc.).

## Common Commands

```bash
# Training (configure constants in train.py first)
python src/train.py

# Evaluation
python src/evaluation.py \
    --data-config config/data_config.yaml \
    --model-config config/architecture.yaml \
    --weight-path results/train_YYMMDDHHM/Epoch_BEST/weight/Epoch_BEST.pth \
    --save-dir results/eval_output \
    --batch-size 8

# Grad-CAM visualization
python src/grad_cam.py \
    --csv-path results/eval_output/test_results/pred_results_info.csv \
    --model-config config/architecture.yaml \
    --weight-path path/to/weights.pth \
    --save-dir results/gradcam \
    --actual-label 1 --predict-label 0

# Test dataset loading
python src/dataset.py \
    --data-config config/data_config.yaml \
    --model-config config/architecture.yaml

# Test model creation
python src/model.py
```

## Architecture

### Core Pipeline

```
config/data_config.yaml     config/architecture.yaml
         │                           │
         └─────────┬─────────────────┘
                   ▼
          LoadDataInfo (dataset.py)
                   │
                   ▼
            CnnDataset (dataset.py)
                   │
                   ▼
    ┌──────────────┴──────────────┐
    │   create_sampler()          │  (sampler.py)
    │   - weighted                │
    │   - balanced                │
    │   - none                    │
    └──────────────┬──────────────┘
                   ▼
              DataLoader
                   │
                   ▼
    create_classifier_model()      (model.py, wraps timm)
                   │
                   ▼
           train() / evaluate()    (train.py, evaluation.py)
                   │
                   ▼
           postprocess()           (utils.py)
                   │
                   ▼
       ./results/train_YYMMDDHHM/
```

### Key Design Decisions

- **Class weighting**: `BCEWithLogitsLoss(pos_weight=n_real/n_fake)` handles imbalanced datasets
- **Sampling strategies** (sampler.py):
  - `weighted`: WeightedRandomSampler with replacement
  - `balanced`: BalancedBatchSampler ensuring 1:1 ratio per batch
  - `none`: Default shuffle
- **Asymmetric augmentation** (augmentation.py): Different transforms for Real vs Fake images
  - Fake images get more aggressive augmentation (lower JPEG quality, smaller downscale)
- **ImageNet normalization**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Configuration

**config/architecture.yaml**:
```yaml
training:
  lr: 0.001
  weight-decay: 0.001
  bs: 8
  early-stop: 20
  sampling:
    strategy: "weighted"    # "weighted" | "balanced" | "none"
    epoch_mode: "full"      # "minority" | "full"

model:
  name: xception           # any timm model name
  image-size: 299
  num-classes: 1           # binary classification uses 1
```

**config/data_config.yaml**: Define paths to train/validation/test directories for fake and real images.

### Output Structure

Training creates timestamped directories:
```
./results/train_YYMMDDHHM/
├── architecture.yaml          # Copy of training config
├── data_config.yaml           # Copy of data config
├── acc_loss_info.csv          # Training curves data
├── acc_per_epoch_line.png
├── loss_per_epoch_line.png
├── Epoch_BEST/
│   ├── weight/Epoch_BEST.pth
│   ├── train_results/
│   └── validation_results/
└── Epoch_LAST/
    └── ...
```

## File Reference

| File | Purpose |
|------|---------|
| [train.py](src/train.py) | Training loop, edit constants at top for configuration |
| [evaluation.py](src/evaluation.py) | CLI for model evaluation |
| [model.py](src/model.py) | `create_classifier_model()` - thin wrapper around timm |
| [dataset.py](src/dataset.py) | `LoadDataInfo`, `CnnDataset` classes |
| [sampler.py](src/sampler.py) | `create_sampler()` factory, `BalancedBatchSampler` |
| [augmentation.py](src/augmentation.py) | Transform classes, enable by uncommenting in dataset.py |
| [utils.py](src/utils.py) | `check_correct()`, `postprocess()`, `calculate_metrics()` |
| [grad_cam.py](src/grad_cam.py) | Grad-CAM visualization CLI |

## Development Notes

- PyTorch installation requires matching CUDA version (see requirements.txt comments)
- To enable augmentation: uncomment the augmentation import and transform sections in [dataset.py](src/dataset.py:186-201)
- Model weights are not committed; configure `WEIGHT_PATH` in train.py or use `--weight-path` CLI args
- Training hyperparameter guidance in [사용전략.md](사용전략.md) (Korean)
