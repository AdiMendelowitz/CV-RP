# Advanced CV and Efficient Models

From-scratch PyTorch implementations of efficient neural network architectures and model
compression techniques, benchmarked on CIFAR-10 image classification. All compression
experiments use a ResNet-18 baseline (93.43% top-1, 11.17M parameters) trained from
scratch in the companion `computer-vision-foundations` module.

## Contents

| Area | Directory | Description |
|---|---|---|
| Efficient Architectures | `code/efficient_architectures/` | EfficientNet, ConvNeXt, receptive field analysis |
| Model Compression | `code/compression/` | Distillation, PTQ, pruning, inference benchmarking |
| Object Detection | `code/detection/` | YOLOv8n PCB defect detection, NumPy metrics |
| Semantic Segmentation | `code/segmentation/` | U-Net from scratch, combined BCE+Dice loss |

---

## Installation

```bash
# From the repository root
source .venv/Scripts/activate
pip install torch torchvision timm scikit-learn ultralytics
```

All scripts resolve data to `<repo_root>/data/` automatically via
`Path(__file__).resolve().parents[N] / "data"`. No manual path configuration is needed.

---

## Results Summary

All benchmark numbers are in `experiments/results.md`. The table below collects the
primary result from each module.

| Module | Model | Task | Primary Metric | Value |
|---|---|---|---|---|
| Efficient Architectures | ConvNeXt-Tiny | Linear probe, CIFAR-10 | Top-1 (%) | 95.08 |
| Efficient Architectures | EfficientNet-B0 | Linear probe, CIFAR-10 | Top-1 (%) | 90.06 |
| Efficient Architectures | ResNet-18 | Linear probe, CIFAR-10 | Top-1 (%) | 83.83 |
| Efficient Architectures | ViT-Tiny | Linear probe, CIFAR-10 | Top-1 (%) | 80.72 |
| Compression | Static INT8 PTQ | CIFAR-10 classification | Top-1 (%) | 93.44 |
| Compression | Pruning 40% L1 | CIFAR-10 classification | Top-1 (%) | 93.27 |
| Compression | SmallCNN distilled | CIFAR-10 classification | Top-1 (%) | 78.34 |
| Detection | YOLOv8n | PCB defect detection | mAP@0.5 | 0.9896 |
| Segmentation | U-Net | Carvana car masking | Validation Dice | approximately 0.99 |

---

## Efficient Architectures

From-scratch implementations of two modern convolutional architectures, benchmarked via a
frozen linear probe on CIFAR-10. See `code/efficient_architectures/README.md` for full
implementation details and receptive field analysis.

**EfficientNet** (Tan and Le, ICML 2019) builds on the Mobile Inverted Bottleneck (MBConv)
with Squeeze-Excitation channel attention and compounds width, depth, and resolution scaling
from a single coefficient. Implemented variants: B0, B1, B4. Source:
`code/efficient_architectures/efficientnet.py`.

**ConvNeXt** (Liu et al., CVPR 2022) modernizes a ResNet training recipe by replacing 3x3
convolutions with 7x7 depthwise convolutions, BatchNorm with Layer Normalization, ReLU with
GELU, and adding inverted-bottleneck MLP blocks and learnable layer scale. Implemented
variants: Tiny, Small, Base. Source: `code/efficient_architectures/convnext.py`.

### Architecture Benchmark

Linear probe evaluation over frozen ImageNet pretrained backbones (via timm). Features were
L2-normalized and passed to a logistic regression classifier (C=0.316). Inference times are
single-image CPU medians over 200 runs.

| Architecture | Top-1 (%) | Parameters (M) | Accuracy / Param (%/M) |
|---|---|---|---|
| ConvNeXt-Tiny | 95.08 | 27.8 | 3.42 |
| EfficientNet-B0 | 90.06 | 4.0 | 22.52 |
| ResNet-18 | 83.83 | 11.2 | 7.49 |
| ViT-Tiny | 80.72 | 5.5 | 14.68 |

EfficientNet-B0 achieves the best accuracy-per-parameter ratio at 22.52 %/M (backbone parameters, num_classes=0), consistent with
the compound scaling hypothesis. ConvNeXt-Tiny leads in absolute accuracy. ViT-Tiny's lower
score reflects its data hunger: the frozen linear probe setting amplifies the gap between
transformer and ConvNet representations at this scale, where self-attention cannot fully
leverage its long-range modeling capacity.

Script: `experiments/benchmark_architecture.py`.

### Inference Example

```python
# code/efficient_architectures/convnext.py
import torch
from convnext import convnext_tiny

model = convnext_tiny(num_classes=1000)
model.eval()
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    logits = model(x)        # shape: (1, 1000)
pred = logits.argmax(dim=1)
```

---

## Model Compression

Three compression techniques applied to the same ResNet-18 checkpoint. Latency is
single-sample CPU inference, median of 100 runs. See `code/compression/README.md` for
full experimental details.

| Method | Params (M) | Size (MB) | Latency (ms) | Top-1 (%) |
|---|---|---|---|---|
| ResNet-18 (FP32 baseline) | 11.17 | 42.70 | 9.47 | 93.43 |
| + Static INT8 (PTQ) | 11.17 | 10.80 | 6.21 | 93.44 |
| + Dynamic INT8 | 11.17 | 42.69 | 18.80 | 93.44 |
| + Pruning 40% L1 unstructured | 11.17 | 42.70 | 9.47 | 93.27 |
| SmallCNN distilled (T=4, a=0.3) | 0.17 | 0.65 | 0.86 | 78.34 |
| SmallCNN CE baseline | 0.17 | 0.65 | 0.84 | 78.27 |

Static INT8 PTQ achieves 3.95x size reduction and 1.52x speedup with no accuracy degradation.
Knowledge distillation achieves 65.6x parameter reduction and 14.7x speedup at a 15.09pp
accuracy cost versus the teacher. L1 unstructured pruning at 40% sparsity degrades accuracy
by 0.16pp after fine-tuning but produces no CPU speedup because standard dense kernels
process zeroed weights identically to non-zeros.

### Inference Example

```python
# code/compression/distillation.py
import torch
from distillation import build_student

model = build_student(num_classes=10)
ckpt = torch.load(
    "checkpoints/distillation/best_student_distill.pth", map_location="cpu"
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
x = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    logits = model(x)        # shape: (1, 10)
pred = logits.argmax(dim=1)
```

---

## Object Detection

YOLOv8n fine-tuned for 6-class PCB defect detection. A from-scratch NumPy implementation of
detection metrics (IoU, NMS, 11-point AP, mAP) is provided separately from the Ultralytics
training pipeline. Source: `code/detection/metrics.py`.

**Dataset:** 8,001 matched image-label pairs (6 classes: mouse_bite, spur, missing_hole,
short, open_circuit, spurious_copper), split 80/10/10 with seed 42.

**Results (test set, 801 images, 1,621 instances):**

| Metric | Value |
|---|---|
| mAP@0.5 | 0.9896 |
| mAP@0.5:0.95 | 0.6025 |
| Precision | 0.9769 |
| Recall | 0.9837 |

Model: 3.0M parameters, 8.1 GFLOPs, 6.0MB. Inference speed: 3.9ms per image on Tesla T4.
Training notebook: `code/detection/yolov8_pcb_kaggle.ipynb`. Full error analysis including
per-class AP, confusion matrix, and PR curves: [code/detection/yolov8_pcb.md](code/detection/yolov8_pcb.md).

The 0.39 gap between mAP@0.5 and mAP@0.5:0.95 reflects the difficulty of tight box
localization for small defects (roughly 3-7% of image width). The primary failure mode is
false positives on background texture, not cross-class confusion.

### Inference Example

```python
# code/detection/
from ultralytics import YOLO

model = YOLO("best_pcb_yolov8n.pt")
results = model.predict("image.jpg", conf=0.487, imgsz=640)
for r in results:
    print(r.boxes.xyxy, r.boxes.cls, r.boxes.conf)
```

---

## Semantic Segmentation

U-Net (Ronneberger et al., MICCAI 2015) implemented from scratch as a symmetric
encoder-decoder with skip connections via feature map concatenation. Channel progression:
1 (or 3) -> 64 -> 128 -> 256 -> 512 -> 1024 through the encoder, then mirrored back to the
output class count in the decoder. All convolutions use padding=1, so the output mask has
the same spatial dimensions as the input image. Training uses a combined loss weighted by
alpha=0.3 (BCE) and 0.7 (Dice), applied as `alpha * BCE + (1 - alpha) * Dice`. Source:
`code/segmentation/unet.py`, `code/segmentation/segmentation_loss.py`.

### Carvana Image Masking (Kaggle, 2017)

Binary segmentation of car silhouettes from RGB images at 512x512 resolution.

**Dataset:** Carvana Image Masking Challenge competition dataset. 90/10 random
train/validation split (seed 42).

**Training configuration:**

| Parameter | Value |
|---|---|
| Input size | 512 x 512, RGB |
| Batch size | 8 |
| Epochs | 20 |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | OneCycleLR (max_lr=1e-4) |
| Loss | 0.3 * BCE + 0.7 * Dice |
| AMP | fp16 (GradScaler) |
| Gradient clipping | max_norm=1.0 |
| Hardware | Kaggle T4 |
| Seed | 42 |

**Result:** Validation Dice approximately 0.99 at epoch 20, as read from the training
curves plot (`code/segmentation/outputs/training_curves.png`). The exact final scalar is
not available in the log file, which contains only notebook conversion output. The Dice
curve rises steeply from 0.70 at epoch 1 to 0.95 by epoch 5, then continues to plateau
near 0.99 from epoch 10 onward. Train and validation combined losses converge closely
throughout, with no sign of overfitting at epoch 20.

Notebook: `code/segmentation/train_unet_carvana.ipynb`. Checkpoint:
`code/segmentation/outputs/checkpoints/best_unet_carvana.pth`.

### LGG MRI Brain Tumour Segmentation

Binary segmentation of lower-grade glioma tumour regions from grayscale MRI slices.

**Dataset:** LGG MRI Segmentation (Buda et al., Computers in Biology and Medicine, 2019).
Folder layout: one directory per patient (TCGA_* prefix), each containing paired
`<slice>.tif` and `<slice>_mask.tif` files. Split is patient-level (val_fraction=0.1)
to prevent anatomy leakage between train and validation sets.

**Training configuration:**

| Parameter | Value |
|---|---|
| Input size | 256 x 256, grayscale |
| Batch size | 4 |
| Epochs | 20 |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Loss | 0.3 * BCE + 0.7 * Dice |
| Augmentation | Random horizontal flip, rotation +-10 degrees |
| Gradient clipping | max_norm=1.0 |
| Seed | 42 |

A trained checkpoint exists at `code/segmentation/checkpoints/best_unet_lgg.pth`, but
the exact final validation Dice and loss are not recoverable from files on disk without
loading the checkpoint. Script: `code/segmentation/train_unet.py`.

### Inference Example

```python
# code/segmentation/unet.py
import torch
from unet import UNet

model = UNet(in_channels=3, num_classes=1)
ckpt = torch.load(
    "outputs/checkpoints/best_unet_carvana.pth", map_location="cpu"
)
model.load_state_dict(ckpt["model"])
model.eval()
x = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    mask = torch.sigmoid(model(x)) > 0.5   # shape: (1, 1, 512, 512)
```

---

## Code Structure

```
Advanced CV & Efficient Models/
|-- code/
|   |-- efficient_architectures/
|   |   |-- efficientnet.py              # EfficientNet-B0/B1/B4 from scratch
|   |   |-- convnext.py                  # ConvNeXt-Tiny/Small/Base from scratch
|   |   `-- receptive_field_analysis.py  # Theoretical and empirical RF analysis
|   |-- compression/
|   |   |-- distillation.py              # SmallCNN, distillation_loss, build_student
|   |   |-- train_distillation.py        # Distillation and CE baseline training
|   |   |-- quantization.py              # Dynamic and static INT8 PTQ
|   |   |-- inference_benchmark.py       # Teacher vs student latency benchmark
|   |   `-- visualize_soft_targets.py    # Soft target distributions at T=1,2,4,8
|   |-- detection/
|   |   |-- metrics.py                   # IoU, NMS, AP, mAP (NumPy, from scratch)
|   |   |-- test_metrics.py              # Pytest suite for metrics
|   |   `-- yolov8_pcb_kaggle.ipynb      # YOLOv8n PCB defect detection
|   `-- segmentation/
|       |-- unet.py                      # U-Net encoder-decoder from scratch
|       |-- segmentation_loss.py         # Dice loss, IoU score, combined BCE+Dice
|       |-- train_unet.py                # LGG MRI training script
|       `-- train_unet_carvana.ipynb     # Carvana training notebook
|-- experiments/
|   `-- benchmark_architecture.py        # Linear probe benchmark via timm
`-- notes/
    |-- efficientnet_architecture.md
    |-- knowledge_distillation.md
    |-- quantiziation_pruning.md
    `-- networks_comparison.md
```

---

## Reproducing Results

All scripts resolve data to `<repo_root>/data/` via
`Path(__file__).resolve().parents[N] / "data"`. Run from the repo root.

```bash
# Architecture linear probe benchmark (requires timm, scikit-learn)
python "Advanced CV & Efficient Models/experiments/benchmark_architecture.py"

# Post-training quantization
python "Advanced CV & Efficient Models/code/compression/quantization.py"

# Inference benchmark: teacher vs student
python "Advanced CV & Efficient Models/code/compression/inference_benchmark.py"

# LGG segmentation training
python "Advanced CV & Efficient Models/code/segmentation/train_unet.py"

# Knowledge distillation (Kaggle T4 recommended, ~25 min)
# Run code/compression/distillation-kaggle.ipynb on Kaggle with resnet_input dataset attached.

# Unstructured pruning (Kaggle T4 recommended)
# Run code/compression/pruning_kaggle.ipynb on Kaggle with resnet_input dataset attached.
```

Environment: Python 3.12, PyTorch, timm, scikit-learn, ultralytics.

---

## References

- Hinton, G., Vinyals, O., and Dean, J. (2015). Distilling the Knowledge in a Neural
  Network. arXiv:1503.02531.
- Han, S., Pool, J., Tran, J., and Dally, W. (2015). Learning both Weights and Connections
  for Efficient Neural Networks. NeurIPS 2015.
- Jacob, B. et al. (2018). Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference. CVPR 2018.
- Nagel, M. et al. (2021). A White Paper on Neural Network Quantization. arXiv:2106.08295.
- Tan, M. and Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional
  Neural Networks. ICML 2019.
- Liu, Z. et al. (2022). A ConvNet for the 2020s. CVPR 2022.
- Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional Networks for
  Biomedical Image Segmentation. MICCAI 2015.