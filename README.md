# Computer Vision Portfolio

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-80%20passed-brightgreen.svg)](computer-vision-foundations/code/tests/test_core.py)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

From-scratch PyTorch implementations of core computer vision systems: classical
image processing, deep residual networks, Vision Transformers, self-supervised
contrastive learning, efficient architectures (EfficientNet, ConvNeXt), model
compression (quantization, pruning, knowledge distillation), object detection,
semantic segmentation, and medical image analysis on ISIC 2018. All
implementations are validated against published benchmarks or challenge
leaderboards.

---

## Repository Structure

```
CV-RP/
├── computer-vision-foundations/        # From-scratch implementations: classical CV, CNNs, ViT, SimCLR
│   └── code/
│       ├── classical_cv/               # NumPy-only: convolution, Sobel, Canny, geometric transforms
│       ├── cnn_scratch/                # NumPy CNN with full forward pass and backprop (MNIST)
│       ├── pytorch_cnn/                # ResNet-18 from scratch (CIFAR-10)
│       ├── vision_transformers/        # ViT-Tiny from scratch (CIFAR-10)
│       ├── self_supervised_learning/   # SimCLR contrastive learning pipeline
│       └── tests/
│           └── test_core.py            # 38 unit tests across 15 test classes
├── Advanced CV & Efficient Models/     # Efficient architectures, compression, detection, segmentation
│   ├── code/
│   │   ├── efficient_architectures/    # EfficientNet-B0/B1/B4 and ConvNeXt-Tiny/Small/Base from scratch
│   │   ├── compression/                # Knowledge distillation, INT8 PTQ, L1 pruning, inference benchmarking
│   │   ├── detection/                  # YOLOv8n PCB defect detection; NumPy metrics (IoU, NMS, AP, mAP)
│   │   └── segmentation/               # U-Net from scratch; Carvana and LGG MRI segmentation
│   └── experiments/
│       └── benchmark_architecture.py   # Linear probe benchmark via timm
├── cv-detection-segmentation/          # Instance segmentation components and ISIC 2018 medical imaging
│   ├── instance_segmentation/
│   │   ├── hungarian_loss.py           # DETR set prediction loss with Hungarian matching (from scratch)
│   │   ├── roi_align.py                # RoI Align with bilinear interpolation (from scratch)
│   │   ├── mask_head.py                # Mask R-CNN mask head (from scratch)
│   │   ├── focal_loss.py               # Alpha-balanced focal loss (from scratch)
│   │   ├── isic_pipeline.py            # End-to-end lesion segmentation and classification pipeline
│   │   └── (Kaggle notebooks)          # ISIC 2018 Task 1 segmentation, Task 3 classification, pipeline
│   └── notes/                          # Paper reading notes: DETR, Mask R-CNN, Deformable DETR
└── data/                               # Shared dataset cache; all scripts resolve here via Path(__file__)
```

---

## Results

### CV Foundations

| Model | Task | Accuracy | Parameters |
|-------|------|----------|------------|
| NumPy CNN | MNIST digit classification | 90.94% | ~13K |
| ResNet-18 (from scratch) | CIFAR-10 classification | 93.43% | 11,173,962 |
| ViT-Tiny (from scratch) | CIFAR-10 classification | 86.70% | 5,356,234 |
| SimCLR (linear eval, frozen encoder) | CIFAR-10 | 68.23% | -- |

ResNet-18: SGD with momentum 0.9, weight decay 5e-4, cosine annealing over
200 epochs, batch size 128. ViT-Tiny (Pre-LN, DeiT-Tiny configuration per
Touvron et al., 2021): AdamW with selective weight decay, linear warmup,
MixUp, CutMix, TrivialAugmentWide, label smoothing. SimCLR: NT-Xent loss at
tau=0.5, 100 pretraining epochs, batch size 256. All architectures implemented
from `nn.Module` primitives; no torchvision.models used.

The 6.7pp gap between ViT-Tiny and ResNet-18 on CIFAR-10 is consistent with
published literature. ViT's lack of convolutional inductive biases requires
substantially more data to overcome; Dosovitskiy et al. (2020) place the
crossover at approximately 14M images. CIFAR-10's 50K training images are
three orders of magnitude below that threshold.

The NumPy CNN matches PyTorch's training accuracy to within 0.01%, confirming
mathematical equivalence between manual gradient derivation and autograd.

---

### Architecture Linear Probe Benchmark

Frozen ImageNet pretrained backbones via timm; L2-normalized features; logistic
regression head (C=0.316); CIFAR-10, images resized to 224 x 224; CPU;
single-image batch. Inference is median over 200 runs, backbone forward pass
only.

| Architecture | Top-1 (%) | Parameters (M) | Inference (ms/img) |
|--------------|-----------|----------------|--------------------|
| ConvNeXt-Tiny | 95.08 | 27.8 | 90.6 |
| EfficientNet-B0 | 90.06 | 4.0 | 29.2 |
| ResNet-18 | 83.83 | 11.2 | 26.8 |
| ViT-Tiny | 80.72 | 5.5 | 26.9 |

EfficientNet-B0 achieves 22.5%/M accuracy-per-parameter, the best across all
four architectures, consistent with the compound scaling hypothesis (Tan and
Le, ICML 2019). ConvNeXt-Tiny leads in absolute accuracy at a 7x parameter
cost. ViT-Tiny's lower linear probe score reflects its data dependency in the
frozen representation setting.

---

### Model Compression

All experiments use the same ResNet-18 checkpoint (93.43%, 11.17M params,
trained from scratch on CIFAR-10). Latency is single-sample CPU inference,
median of 100 runs.

| Method | Params (M) | Size (MB) | Latency (ms) | Top-1 (%) |
|--------|------------|-----------|--------------|-----------|
| ResNet-18 FP32 baseline | 11.17 | 42.70 | 9.47 | 93.43 |
| Static INT8 PTQ | 11.17 | 10.80 | 6.21 | 93.44 |
| Dynamic INT8 PTQ | 11.17 | 42.69 | 18.80 | 93.44 |
| Pruning 40% L1 unstructured | 11.17 | 42.70 | 9.47 | 93.27 |
| SmallCNN distilled (T=4, alpha=0.3) | 0.17 | 0.65 | 0.86 | 78.33 |
| SmallCNN CE baseline | 0.17 | 0.65 | 0.84 | 78.97 |

Static INT8 PTQ achieves 3.95x size reduction and 1.52x speedup with zero
accuracy cost. SmallCNN knowledge distillation achieves 65.6x parameter
reduction (0.17M vs 11.17M) and 14.4x speedup, but underperforms the CE
baseline by 0.64pp (78.33% vs 78.97%), likely due to the optimisation
difficulty of a very small student. L1 unstructured pruning at 40% sparsity
degrades accuracy by 0.16pp after fine-tuning but produces no CPU speedup
because dense matrix kernels process zero-valued weights identically to
non-zeros; structured pruning would be required for hardware acceleration.

---

### Object Detection

YOLOv8n fine-tuned for 6-class PCB defect detection (mouse_bite, spur,
missing_hole, short, open_circuit, spurious_copper). Dataset: 8,001 matched
image-label pairs, 80/10/10 split with seed 42. Test set: 801 images,
1,621 instances.

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.9896 |
| mAP@0.5:0.95 | 0.6025 |
| Precision | 0.9769 |
| Recall | 0.9837 |

Model: 3,006,818 parameters, 8.1 GFLOPs, 6.0MB. Inference: 3.9ms per image
on Tesla T4. The 0.39 gap between mAP@0.5 and mAP@0.5:0.95 reflects
localisation difficulty for small defects (approximately 3-7% of image width).
Detection metrics (IoU, NMS, 11-point AP, mAP) are implemented from scratch
in NumPy alongside the Ultralytics training pipeline.

---

### Semantic Segmentation

U-Net implemented from scratch as a symmetric encoder-decoder with skip
connections via feature map concatenation. Channel progression:
64 -> 128 -> 256 -> 512 -> 1024 through the encoder, mirrored in the decoder.
Training loss: 0.3 * BCE + 0.7 * Dice.

| Dataset | Dice | Configuration |
|---------|------|---------------|
| Carvana Image Masking | approximately 0.99 | 512x512, 20 epochs, AdamW, OneCycleLR, fp16 AMP, T4 GPU |
| LGG MRI Brain Tumour | patient-level split | 256x256, 20 epochs, AdamW, ReduceLROnPlateau |

LGG MRI uses a patient-level train/validation split (val_fraction=0.1) to
prevent anatomy leakage across slices from the same patient. All slices from
a given TCGA patient appear exclusively in either train or validation.

---

### Medical Imaging: ISIC 2018 Skin Lesion Analysis

#### Task 1: Lesion Segmentation

Architecture: Mask R-CNN with ResNet-50-FPN backbone pretrained on COCO
(torchvision), fine-tuned for 2-class lesion segmentation. Box and mask
predictor heads replaced; backbone and FPN weights retained. Bounding boxes
derived from tight bounding rectangles of ground-truth masks.

Metric: thresholded Jaccard index (T=0.65), matching the official challenge
evaluation protocol. Images with IoU below the threshold contribute zero,
directly penalising gross segmentation failures.

| Method | Thresholded Jaccard |
|--------|---------------------|
| ISIC 2018 challenge winner | 0.802 |
| This work (Mask R-CNN, ResNet-50-FPN) | 0.7822 |

Training: AdamW, lr=5e-4, weight decay 1e-4, StepLR (step=7, gamma=0.5),
batch size 4, 512x512, 20 epochs on Kaggle T4. Performance is stable across
replications (0.780 +/- 0.003). The primary failure mode is images containing
clinical measurement artefacts (rulers, measurement triangles), consistent
with Codella et al. (2019). The 2.0pp gap to the challenge winner was achieved
with horizontal flip augmentation only, no test-time augmentation, and no
ensemble.

The Hungarian matching loss, RoI Align with bilinear interpolation, the Mask
R-CNN mask head, and the focal loss function are each implemented from scratch
in `cv-detection-segmentation/instance_segmentation/` to establish
component-level understanding alongside the fine-tuning pipeline.

#### Task 3: Disease Classification

Architecture: EfficientNet-B0 (timm, ImageNet pretrained), fine-tuned for
7-class dermoscopic disease classification (NV, MEL, BKL, BCC, AKIEC, VASC,
DF). Class imbalance (NV constitutes 66.9% of training images; NV/DF ratio
approximately 58:1) addressed via per-class weighted cross-entropy
(weight_c = total / (num_classes * count_c)). Data split is lesion-level on
`lesion_id` to prevent information leakage from multi-crop HAM10000 images.

Metric: balanced accuracy (mean per-class recall), per the official challenge
protocol.

| Model | Balanced Accuracy | MEL Recall | Parameters |
|-------|------------------|------------|------------|
| ISIC 2018 Task 3 challenge winner | 0.885 | -- | -- |
| EfficientNet-B3, weighted CE | 0.7498 | 0.505 | 10,706,991 |
| EfficientNet-B0, weighted CE (best) | 0.7457 | 0.624 | 4,016,515 |
| EfficientNet-B0, focal loss gamma=2.0 | 0.7376 | 0.592 | 4,016,515 |

Best B0 configuration: albumentations augmentation (horizontal flip, vertical
flip, rotation, RandomResizedCrop, ColorJitter, GaussianBlur, ISONoise,
CoarseDropout), progressive unfreezing (head-only for epochs 1-5 at
lr_head=1e-3, full network from epoch 6 at lr=3e-4), CosineAnnealingLR,
dropout 0.4, weight decay 1e-3, 25 epochs. B3 uses the same configuration
with drop_rate=0.4 set explicitly to match B0 regularisation intent.

MEL recall is the persistent bottleneck across all configurations (0.505-0.624
across seven runs), driven by the visual similarity between MEL and NV and the
6:1 NV/MEL sample imbalance. Focal loss (gamma=2.0) produced no meaningful
improvement over weighted CE, confirming that the dominant failure mode is
overfitting rather than hard-example difficulty. The gap to the challenge
winner is primarily explained by ensemble size and backbone scale: winning
submissions used ensembles of EfficientNet-B4/B5 with extensive test-time
augmentation. The effective capacity ceiling for a single model at 224px is
approximately 0.744-0.750 balanced accuracy across all configurations tested.

#### End-to-End Pipeline

Mask R-CNN (Task 1) and EfficientNet-B0 (Task 3, Run 5) composed into a
single inference pipeline: Mask R-CNN localises the lesion, the predicted
bounding box crops the region, and EfficientNet-B0 classifies the crop.
Evaluated on 50 HAM10000 validation images. Ground-truth segmentation masks
are not available for these images because the Task 1 and HAM10000 datasets
use disjoint ISIC image ID ranges; segmentation quality is assessed
qualitatively.

| Metric | Value |
|--------|-------|
| Detection failures | 0 / 50 (0.0%) |
| Classification failures given detection | 15 / 50 (30.0%) |
| Full pipeline success | 35 / 50 (70.0%) |
| Pipeline balanced accuracy | 0.5219 |
| Standalone classifier balanced accuracy | 0.7441 |

The 22pp gap between pipeline and standalone classification reflects the
distribution shift introduced by crop-based inference: EfficientNet-B0 was
trained on full resized images and receives a tightly cropped region in the
pipeline, losing the spatial context it relies on.

---

## Technical Stack

Python 3.12, PyTorch, timm, albumentations, torchvision, NumPy, scikit-learn,
ultralytics (YOLOv8). Training on Kaggle T4 GPU. Development on Windows 11,
PyCharm.

All from-scratch architectures (ResNet-18, ViT-Tiny, SimCLR, EfficientNet,
ConvNeXt, U-Net, Hungarian matching, RoI Align, mask head, focal loss) are
implemented from `nn.Module` primitives or pure NumPy/PyTorch with no
high-level model library dependencies.

---

## Code Quality

80 unit tests passing across two test suites.

```bash
# CV foundations (38 tests)
pytest computer-vision-foundations/code/tests/test_core.py -v

# Instance segmentation components (42 tests)
pytest cv-detection-segmentation/instance_segmentation/ -v
```

| Suite | Tests | Coverage |
|-------|-------|----------|
| `test_core.py` | 38 | Classical CV, ResNet-18, ViT-Tiny, NT-Xent loss, SimCLR, detection metrics |
| `test_focal_loss.py` | 13 | CE equivalence at gamma=0, monotone down-weighting, input validation |
| `test_hungarian.py` | 17 | GIoU bounds, cost matrix, assignment optimality, set prediction loss |
| `test_roi_align_mask_head.py` | 12 | Output shape, bilinear interpolation correctness, gradient flow |

All code is black (line length 120) and flake8 clean with full type hints.
Data paths use `Path(__file__).resolve().parents[N] / "data"` anchored to
script location throughout. MIT licensed.

---

## Setup

```bash
git clone https://github.com/AdiMendelowitz/CV-RP.git
cd CV-RP
.venv\Scripts\Activate.ps1       # Windows PowerShell
pip install -e .                 # pyproject.toml manages all dependencies

# Run CV foundations tests
pytest computer-vision-foundations/code/tests/test_core.py -v

# Run instance segmentation tests
pytest cv-detection-segmentation/instance_segmentation/ -v
```

GPU-dependent training runs (ViT-Tiny, SimCLR, distillation, ISIC) are
provided as Kaggle notebooks with T4 GPU runtimes.

---

## References

He, K. et al. Deep Residual Learning for Image Recognition.
arXiv:1512.03385, 2015.

Dosovitskiy, A. et al. An Image is Worth 16x16 Words: Transformers for Image
Recognition at Scale. arXiv:2010.11929, 2020.

Touvron, H. et al. Training data-efficient image transformers and distillation
through attention. arXiv:2012.12877, 2021.

Chen, T. et al. A Simple Framework for Contrastive Learning of Visual
Representations. arXiv:2002.05709, 2020.

Tan, M. and Le, Q. EfficientNet: Rethinking Model Scaling for Convolutional
Neural Networks. ICML 2019. arXiv:1905.11946.

Liu, Z. et al. A ConvNet for the 2020s. CVPR 2022. arXiv:2201.03545.

Ronneberger, O., Fischer, P., and Brox, T. U-Net: Convolutional Networks for
Biomedical Image Segmentation. MICCAI 2015. arXiv:1505.04597.

He, K. et al. Mask R-CNN. ICCV 2017. arXiv:1703.06870.

Carion, N. et al. End-to-End Object Detection with Transformers. ECCV 2020.
arXiv:2005.12872.

Lin, T.-Y. et al. Focal Loss for Dense Object Detection. ICCV 2017.
arXiv:1708.02002.

Hinton, G., Vinyals, O., and Dean, J. Distilling the Knowledge in a Neural
Network. arXiv:1503.02531, 2015.

Codella, N. et al. Skin Lesion Analysis Toward Melanoma Detection 2018.
arXiv:1902.03368, 2019.

Tschandl, P., Rosendahl, C., and Kittler, H. The HAM10000 Dataset, a Large
Collection of Multi-Source Dermatoscopic Images of Common Pigmented Skin
Lesions. Scientific Data, 2018.

Zhu, X. et al. Deformable DETR: Deformable Transformers for End-to-End
Object Detection. ICLR 2021. arXiv:2010.04159.