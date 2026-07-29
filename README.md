# Applied Machine Learning Portfolio

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-225-brightgreen.svg)](computer-vision-foundations/code/tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

From-scratch PyTorch implementations across computer vision, adversarial robustness and
time-series forecasting: classical image processing, deep residual networks, Vision
Transformers, self-supervised contrastive learning, efficient architectures and model
compression, object detection, semantic segmentation, medical image analysis on ISIC 2018,
gradient-based adversarial attacks with a tested evaluation harness, and transformer-based
multivariate forecasting. Core architectures are built from `nn.Module` primitives or pure
NumPy rather than high-level model libraries, and results are reported on standard
benchmarks and challenge datasets. All code is black and flake8 clean with full type hints.

---

## Repository Structure

```
CV-RP/
├── computer-vision-foundations/        # From-scratch: classical CV, CNNs, ViT, SimCLR
│   └── code/
│       ├── classical_cv/               # NumPy-only: convolution, Sobel, Canny, geometric transforms
│       ├── cnn_scratch/                # NumPy CNN with full forward pass and backprop (MNIST)
│       ├── pytorch_cnn/                # ResNet-18 from scratch (CIFAR-10)
│       ├── vision_transformers/        # ViT-Tiny from scratch (CIFAR-10)
│       ├── self_supervised_learning/   # SimCLR contrastive learning pipeline
│       └── tests/                      # Core unit tests (classical CV, ResNet, ViT, SimCLR, metrics)
├── Advanced CV & Efficient Models/     # Efficient architectures, compression, detection, segmentation
│   ├── code/
│   │   ├── efficient_architectures/    # EfficientNet and ConvNeXt from scratch
│   │   ├── compression/                # Knowledge distillation, INT8 PTQ, L1 pruning, inference benchmarking
│   │   ├── detection/                  # YOLOv8n PCB defect detection; NumPy metrics (IoU, NMS, AP, mAP)
│   │   └── segmentation/               # U-Net from scratch; Carvana and LGG MRI segmentation
│   └── experiments/                    # Linear-probe architecture benchmark
├── cv-detection-segmentation/          # Instance segmentation components and ISIC 2018 medical imaging
│   └── instance_segmentation/
│       ├── hungarian_loss.py           # DETR set prediction loss with Hungarian matching (from scratch)
│       ├── roi_align.py                # RoI Align with bilinear interpolation (from scratch)
│       ├── mask_head.py                # Mask R-CNN mask head (from scratch)
│       ├── focal_loss.py               # Alpha-balanced focal loss (from scratch)
│       ├── isic_pipeline.py            # End-to-end lesion segmentation and classification pipeline
│       ├── isic2018-task1-segmentation.ipynb
│       ├── isic2018-task3-classification.ipynb
│       └── isic_pipeline.ipynb
├── adversarial-ml-toolkit/             # FGSM, PGD, Carlini-Wagner; PGD adversarial training
│   ├── attacks/                        # FGSM, PGD, C&W L2 (from scratch)
│   ├── defenses/                       # PGD adversarial training
│   ├── models/                         # ResNet-18 (CIFAR stem) and the normalisation wrapper
│   ├── experiments/                    # Epsilon sweep and robustness evaluation
│   ├── Notes/                          # Derivations and analysis
│   └── tests/
└── time-series-forecasting/            # Transformer forecasting: PatchTST, iTransformer, TimeMixer
    ├── models/                         # PatchTST, iTransformer, TimeMixer implementations
    ├── baselines/                      # Linear baseline
    ├── data/                           # ETT dataset loader
    ├── experiments/                    # Training notebooks
    ├── results/                        # Benchmark CSVs and forecast plots
    └── tests/                          # Model and dataset unit tests
```

---

## Results

### CV Foundations

| Model | Task | Accuracy | Parameters |
|-------|------|----------|------------|
| NumPy CNN | MNIST digit classification | 90.94% | approximately 103,000 |
| ResNet-18 (from scratch) | CIFAR-10 classification | 93.43% | 11,173,962 |
| ViT-Tiny (from scratch) | CIFAR-10 classification | 86.70% | 5,356,234 |
| SimCLR (linear eval, frozen encoder) | CIFAR-10 | 68.23% | -- |

ResNet-18: SGD with momentum 0.9, weight decay 5e-4, cosine annealing over
200 epochs, batch size 128. ViT-Tiny (Pre-LN, DeiT-Tiny configuration per
Touvron et al., 2021): AdamW with selective weight decay, linear warmup,
MixUp, CutMix, TrivialAugmentWide, label smoothing. SimCLR: NT-Xent loss at
tau=0.5, 100 pretraining epochs, batch size 256. All architectures are
implemented from `nn.Module` primitives, with no torchvision.models used.

The 6.7pp gap between ViT-Tiny and ResNet-18 on CIFAR-10 is consistent with
published literature. The absence of convolutional inductive biases in ViT
requires substantially more data to overcome, and Dosovitskiy et al. (2020)
place the crossover at approximately 14M images. CIFAR-10's 50K training images
sit a factor of 280 below that threshold.

The NumPy CNN matches PyTorch's training accuracy to within 0.01%, confirming
mathematical equivalence between manual gradient derivation and autograd.

---

### Architecture Linear Probe Benchmark

Frozen ImageNet pretrained backbones via timm, L2-normalised features, and a
logistic regression head (C=0.316) on CIFAR-10, images resized to 224 x 224,
CPU, single-image batch. Inference is the median over 200 runs.

Backbones are instantiated with `num_classes=0`, so both the parameter counts
and the timings below cover the feature extractor only and exclude any
classification head. They are therefore not comparable to the head-inclusive
counts in the table above, where ResNet-18 carries a 10-class head.

| Architecture | Top-1 (%) | Backbone parameters (M) | Inference (ms/img) |
|--------------|-----------|-------------------------|--------------------|
| ConvNeXt-Tiny | 95.08 | 27.8 | 90.6 |
| EfficientNet-B0 | 90.06 | 4.0 | 29.2 |
| ResNet-18 | 83.83 | 11.2 | 26.8 |
| ViT-Tiny | 80.72 | 5.5 | 26.9 |

EfficientNet-B0 achieves 22.5%/M accuracy-per-parameter, the best across the
four architectures, consistent with the compound scaling hypothesis (Tan and
Le, ICML 2019). ConvNeXt-Tiny leads in absolute accuracy at a 7x parameter
cost. The lower linear-probe score for ViT-Tiny reflects its data dependency
in the frozen representation setting.

---

### Model Compression

All experiments use the same ResNet-18 checkpoint (93.43%, 11.17M params,
trained from scratch on CIFAR-10). Accuracy is top-1 on the CIFAR-10 test set.
Latency is single-sample CPU inference, median of 100 timed runs after 20
warmup iterations.

| Method | Params (M) | Size (MB) | Latency (ms) | Top-1 (%) |
|--------|------------|-----------|--------------|-----------|
| ResNet-18 FP32 baseline | 11.17 | 42.70 | 9.47 | 93.43 |
| Static INT8 PTQ | 11.17 | 10.80 | 6.21 | 93.44 |
| Dynamic INT8 PTQ | 11.17 | 42.69 | 18.80 | 93.44 |
| Pruning 40% L1 unstructured | 11.17 | 42.70 | 9.47 | 93.27 |
| SmallCNN distilled (T=4, alpha=0.3) | 0.17 | 0.65 | 0.86 | 78.33 |
| SmallCNN CE baseline | 0.17 | 0.65 | 0.84 | 78.97 |

Static INT8 PTQ achieves 3.95x size reduction and 1.52x speedup with no
accuracy cost (93.44% against 93.43%). SmallCNN knowledge distillation achieves
65.6x parameter reduction (170,378 against 11,173,962) and 11.0x speedup while
underperforming the CE baseline by 0.64pp, attributable to the optimisation
difficulty of a very small student. L1 unstructured pruning at 40% sparsity
degrades accuracy by 0.16pp after fine-tuning and produces no CPU speedup,
because dense matrix kernels process zero-valued weights identically to
non-zero ones; structured pruning would be required for hardware acceleration.

Latency figures are single-execution medians. An independent run of the same
benchmark on the same machine measured the FP32 baseline at 12.44 ms against
the 9.47 ms above, a 31% spread, so treat the ratios as indicative of order
rather than precise. The accuracy figures are reproducible: they are stored in
the checkpoints themselves (`val_acc` 0.7833 and 0.7897 at epoch 29), and both
are computed on the test split despite that field name.

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
The detection metrics (IoU, NMS, 11-point AP, mAP) are implemented from
scratch in NumPy alongside the Ultralytics training pipeline.

---

### Semantic Segmentation

U-Net implemented from scratch as a symmetric encoder-decoder with skip
connections via feature-map concatenation. Channel progression:
64 -> 128 -> 256 -> 512 -> 1024 through the encoder, mirrored in the decoder.
Training loss: 0.3 * BCE + 0.7 * Dice.

| Dataset | Dice | Configuration |
|---------|------|---------------|
| Carvana Image Masking | 0.9955 | 512x512, 20 epochs, AdamW, OneCycleLR, fp16 AMP, T4 GPU |
| LGG MRI Brain Tumour | not evaluated | 256x256, 20 epochs, AdamW, ReduceLROnPlateau |

Carvana Dice is the mean over 508 validation images, ranging from 0.9868 to
0.9973. The LGG MRI model trains and checkpoints but has no recorded evaluation
metric, so no figure is quoted for it. That run uses a patient-level
train/validation split (val_fraction=0.1) to prevent anatomy leakage across
slices from the same patient: all slices from a given TCGA patient appear
exclusively in either train or validation.

---

### Adversarial Robustness

Gradient-based attacks implemented from scratch against the same ResNet-18
checkpoint, with a shared evaluation harness. Measured on the first 1,000
CIFAR-10 test images on CPU, at eps = 8/255 for the L-infinity attacks. Success
counts samples classified correctly before the attack and incorrectly after it,
over the 934 clean-correct samples.

| Attack | Steps | Accuracy | Success | Mean L-inf | Mean L2 |
|--------|-------|----------|---------|------------|---------|
| none | 0 | 0.9340 | 0.0000 | 0.000000 | 0.000000 |
| FGSM | 1 | 0.1660 | 0.8223 | 0.031373 | 1.723553 |
| PGD | 20 | 0.0000 | 1.0000 | 0.031373 | 1.322516 |
| PGD | 50 | 0.0000 | 1.0000 | 0.031373 | 1.378558 |
| C&W L2 | 100 | 0.0000 | 1.0000 | 0.024777 | 0.230860 |

A perturbation at the full budget on every pixel of a 3x32x32 image has an L2
of 1.7388. FGSM spends 99.1% of that ceiling and still leaves 16.6% of images
correct, while PGD-20 spends 76.1% and leaves none: the advantage of iteration
is direction rather than magnitude. C&W is an L2 attack with no L-infinity
budget, so its accuracy is not comparable to the rows above it; what it
measures is minimum distortion, 5.7 times below PGD-20's L2. PGD adversarial
training is implemented and its evaluation is in progress, so no defended
robustness figure is quoted here.

Full derivations and the threat-model discussion are in
[`adversarial-ml-toolkit/Notes/adversarial_ml_notes.md`](adversarial-ml-toolkit/Notes/adversarial_ml_notes.md).

---

### Medical Imaging: ISIC 2018 Skin Lesion Analysis

#### Task 1: Lesion Segmentation

Architecture: Mask R-CNN with a ResNet-50-FPN backbone pretrained on COCO
(torchvision), fine-tuned for 2-class lesion segmentation. The box and mask
predictor heads are replaced while backbone and FPN weights are retained.
Bounding boxes are derived from tight bounding rectangles of ground-truth
masks.

Metric: thresholded Jaccard index (T=0.65), matching the official challenge
evaluation protocol. Images with IoU below the threshold contribute zero,
which directly penalises gross segmentation failures.

Best run: 0.7822 over 519 validation images, with 46 of 519 (8.9%) scoring
zero. Three replications under an identical configuration give 0.7803, 0.7764
and 0.7822.

Training: AdamW, lr=5e-4, weight decay 1e-4, StepLR (step=7, gamma=0.5),
batch size 4, 512x512, 20 epochs on Kaggle T4. The primary failure mode is
images containing clinical measurement artefacts (rulers, measurement
triangles), consistent with Codella et al. (2019). Results were obtained with
horizontal flip augmentation only, no test-time augmentation, and no ensemble.

The Hungarian matching loss, RoI Align with bilinear interpolation, the Mask
R-CNN mask head, and the focal loss function are each implemented from scratch
in `cv-detection-segmentation/instance_segmentation/` to establish
component-level understanding alongside the fine-tuning pipeline.

#### Task 3: Disease Classification

Architecture: EfficientNet (timm, ImageNet pretrained), fine-tuned for 7-class
dermoscopic disease classification (NV, MEL, BKL, BCC, AKIEC, VASC, DF). Class
imbalance (NV constitutes 66.9% of training images, with an NV/DF ratio of
approximately 58:1) is addressed via per-class weighted cross-entropy
(weight_c = total / (num_classes * count_c)). The data split is lesion-level
on `lesion_id` to prevent information leakage from multi-crop HAM10000 images.

Metric: balanced accuracy (mean per-class recall), per the official challenge
protocol, at the best-validation checkpoint.

| Model | Balanced Accuracy | MEL Recall | Parameters |
|-------|------------------|------------|------------|
| EfficientNet-B3, weighted CE | 0.7498 | 0.505 | 10,706,991 |
| EfficientNet-B0, weighted CE | 0.7457 | 0.624 | 4,016,515 |
| EfficientNet-B0, focal loss gamma=2.0 | 0.7376 | 0.592 | 4,016,515 |

B3 leads on balanced accuracy by 0.41pp at 2.7x the parameter count. The B0
weighted-CE configuration carries the best MEL recall and is the model used in
the end-to-end pipeline below.

Best B0 configuration: albumentations augmentation (horizontal flip, vertical
flip, rotation, RandomResizedCrop, ColorJitter, GaussianBlur, ISONoise,
CoarseDropout), progressive unfreezing (head-only for epochs 1-5 at
lr_head=1e-3, full network from epoch 6 at lr=3e-4), CosineAnnealingLR,
dropout 0.4, weight decay 1e-3, 25 epochs. B3 uses the same configuration
with drop_rate=0.4 set explicitly to match the B0 regularisation intent.

MEL recall is the persistent bottleneck across all configurations (0.505-0.624
across seven runs), driven by the visual similarity between MEL and NV and the
6:1 NV/MEL sample imbalance. Focal loss (gamma=2.0) produced no meaningful
improvement over weighted CE, indicating that the dominant failure mode is
overfitting rather than hard-example difficulty. The effective capacity ceiling
for a single model at 224px is approximately 0.744-0.750 balanced accuracy
across all configurations tested.

#### End-to-End Pipeline

Mask R-CNN (Task 1) and EfficientNet-B0 (Task 3) are composed into a single
inference pipeline: Mask R-CNN localises the lesion, the predicted bounding box
crops the region, and EfficientNet-B0 classifies the crop. Evaluated on 50
HAM10000 validation images drawn from the same lesion-level split used in
Task 3. Ground-truth segmentation masks are not available for these images
because the Task 1 and HAM10000 datasets use disjoint ISIC image ID ranges, so
segmentation quality is assessed qualitatively.

| Metric | Value |
|--------|-------|
| Detection failures | 0 / 50 (0.0%) |
| Classification failures given detection | 15 / 50 (30.0%) |
| Full pipeline success | 35 / 50 (70.0%) |
| Pipeline balanced accuracy | 0.5219 |

Against the classifier's standalone balanced accuracy of 0.7457, the pipeline
loses roughly 22 points, which reflects the distribution shift introduced by
crop-based inference: EfficientNet-B0 was trained on full resized images and
receives a tightly cropped region in the pipeline, losing the spatial context
it relies on. The two figures are measured on different populations, 50 images
against the full validation split, so the size of that gap is indicative rather
than measured on matched data.

---

### Time-Series Forecasting

Long-horizon multivariate forecasting on ETTh1 (7 variates, hourly electricity
transformer temperature), with PatchTST, iTransformer, and TimeMixer each
implemented from scratch in `time-series-forecasting/models/` alongside a
linear baseline. Metrics are test-set MSE and MAE at seed 42; the look-back
length differs by model and is stated per row, so numbers are comparable within
a model across horizons rather than across models.

PatchTST (look-back 512, from scratch, seed 42):

| Horizon | MSE | MAE |
|---------|------|------|
| 96 | 0.398 | 0.421 |
| 192 | 0.442 | 0.449 |
| 336 | 0.467 | 0.467 |
| 720 | 0.542 | 0.526 |

The implementation reproduces the expected horizon scaling on ETTh1: error rises
monotonically with prediction length, in line with the PatchTST architecture
(Nie et al., ICLR 2023). Figures are single-seed under a fixed training budget.

iTransformer (look-back 96) and TimeMixer (look-back 512), reproduced test
metrics:

| Model | Horizon | MSE | MAE | Parameters |
|-------|---------|------|------|------------|
| iTransformer | 96 | 0.484 | 0.483 | 162,528 |
| iTransformer | 192 | 0.545 | 0.517 | 168,768 |
| iTransformer | 336 | 0.611 | 0.564 | 178,128 |
| iTransformer | 720 | 0.717 | 0.628 | 203,088 |
| TimeMixer | 96 | 0.454 | 0.452 | 33,667 |
| TimeMixer | 192 | 0.499 | 0.481 | 38,563 |
| TimeMixer | 336 | 0.543 | 0.514 | 45,907 |
| TimeMixer | 720 | 0.675 | 0.602 | 65,491 |

The linear baseline (look-back 512) records MSE 0.389 / MAE 0.405 at horizon 96
and MSE 0.485 / MAE 0.471 at horizon 336, the only two horizons it was run at.
It serves as the sanity floor: a transformer that fails to beat it is not
learning useful temporal structure beyond the trend. On that test, all three
transformers fail at horizon 96, where the linear model's 0.389 beats PatchTST's
0.398 and beats the other two by a wide margin. Only PatchTST clears the floor
at horizon 336, at 0.467 against 0.485. This is consistent with the literature
on linear baselines for long-horizon ETT forecasting and is reported here rather
than omitted. All four models share the ETT dataset loader in
`time-series-forecasting/data/` and a common test harness in
`time-series-forecasting/tests/`.

---

## Technical Stack

Python 3.12, PyTorch 2.7.1, timm, albumentations, torchvision, NumPy,
scikit-learn, ultralytics (YOLOv8). Training on Kaggle T4 GPU. Development on
Windows 11, PyCharm.

All from-scratch architectures (ResNet-18, ViT-Tiny, SimCLR, EfficientNet,
ConvNeXt, U-Net, Hungarian matching, RoI Align, mask head, focal loss, FGSM,
PGD, Carlini-Wagner) are implemented from `nn.Module` primitives or pure NumPy
and PyTorch, with no high-level model-library dependencies.

---

## Code Quality

225 tests across six suites, covering ResNet-18, ViT-Tiny, NT-Xent loss and
SimCLR, detection metrics, U-Net segmentation, focal loss, Hungarian matching,
RoI Align and the mask head, the forecasting models and dataset loaders, and
the three adversarial attacks.

| Suite | Tests |
|-------|-------|
| computer-vision-foundations | 38 |
| time-series-forecasting | 59 |
| cv-detection-segmentation | 42 |
| Advanced CV, detection | 25 |
| Advanced CV, segmentation | 16 |
| adversarial-ml-toolkit | 45 |

```bash
python -m pytest computer-vision-foundations/code/tests/ \
  time-series-forecasting/tests/ \
  cv-detection-segmentation/instance_segmentation/ \
  "Advanced CV & Efficient Models/code/detection/" \
  "Advanced CV & Efficient Models/code/segmentation/" -v
```

The adversarial toolkit suite runs from its own root, where its `conftest.py`
makes `attacks`, `models` and `defenses` importable:

```bash
cd adversarial-ml-toolkit && python -m pytest tests/ -v
```

All code is black (line length 120) and flake8 clean with full type hints.
Data paths use `Path(__file__).resolve().parents[N] / "data"`, anchored to
script location throughout.

---

## Setup

```powershell
git clone https://github.com/AdiMendelowitz/CV-RP.git
cd CV-RP

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .                 # pyproject.toml manages all dependencies
```

GPU-dependent training runs (ViT-Tiny, SimCLR, distillation, ISIC, adversarial
training, forecasting) are provided as Kaggle notebooks with T4 GPU runtimes.

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

Goodfellow, I., Shlens, J., and Szegedy, C. Explaining and Harnessing
Adversarial Examples. ICLR 2015. arXiv:1412.6572.

Madry, A. et al. Towards Deep Learning Models Resistant to Adversarial
Attacks. ICLR 2018. arXiv:1706.06083.

Carlini, N. and Wagner, D. Towards Evaluating the Robustness of Neural
Networks. IEEE S&P 2017. arXiv:1608.04644.

Codella, N. et al. Skin Lesion Analysis Toward Melanoma Detection 2018.
arXiv:1902.03368, 2019.

Tschandl, P., Rosendahl, C., and Kittler, H. The HAM10000 Dataset, a Large
Collection of Multi-Source Dermatoscopic Images of Common Pigmented Skin
Lesions. Scientific Data, 2018.

Nie, Y. et al. A Time Series is Worth 64 Words: Long-term Forecasting with
Transformers. ICLR 2023. arXiv:2211.14730.

Liu, Y. et al. iTransformer: Inverted Transformers Are Effective for Time
Series Forecasting. ICLR 2024. arXiv:2310.06625.

Wang, S. et al. TimeMixer: Decomposable Multiscale Mixing for Time Series
Forecasting. ICLR 2024. arXiv:2405.14616.