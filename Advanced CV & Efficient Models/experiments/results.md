# Results

Numbers in this file are read directly from result files on disk. Cells marked "pending"
indicate that no pre-computed aggregate scalar was found; the underlying data exists but
requires further computation.

---

## Architecture Benchmark

Linear probe evaluation over frozen ImageNet pretrained backbones (via timm). Features
are L2-normalized before a logistic regression classifier (C=0.316). Inference times are
single-image CPU medians over 200 runs. Dataset: CIFAR-10, images resized to 224 x 224.

Source: experiments/experiments/architecture_benchmark.md

| Model | Top-1 (%) | Params (M) | Latency (ms/img) |
|---|---|---|---|
| ConvNeXt-Tiny | 95.08 | 27.8 | 90.6 |
| EfficientNet-B0 | 90.06 | 4.0 | 29.2 |
| ResNet-18 | 83.83 | 11.2 | 26.8 |
| ViT-Tiny | 80.72 | 5.5 | 26.9 |

Accuracy reflects representation quality of the pretrained backbone under linear probe,
not fine-tuned performance. Backbone forward pass only; logistic regression head excluded
from latency measurement.

---

## Compression Results

ResNet-18 baseline: 93.43% top-1, 11.17M parameters, 42.70MB, trained from scratch on
CIFAR-10 in computer-vision-foundations/. All compression techniques apply to the same
checkpoint. Latency is single-sample CPU median over 100 runs.

Source: code/compression/README.md

| Method | Params (M) | Size (MB) | Latency (ms) | Top-1 (%) |
|---|---|---|---|---|
| ResNet-18 FP32 baseline | 11.17 | 42.70 | 9.47 | 93.43 |
| Static INT8 PTQ | 11.17 | 10.80 | 6.21 | 93.44 |
| Dynamic INT8 PTQ | 11.17 | 42.69 | 18.80 | 93.44 |
| Pruning 40% L1 unstructured | 11.17 | 42.70 | 9.47 | 93.27 |
| SmallCNN distilled (T=4, alpha=0.3) | 0.17 | 0.65 | 0.86 | 78.33 |
| SmallCNN CE baseline | 0.17 | 0.65 | 0.84 | 78.97 |

Pruning latency is identical to the FP32 baseline because L1 unstructured sparsity
produces no CPU speedup without sparse compute kernels. Dynamic INT8 applies only to the
final Linear layer (5,120 of 11.17M parameters), so no size reduction occurs and
dequantization overhead increases latency.

---

## Detection Results

YOLOv8n fine-tuned on the PCB Defect Dataset (Norbert Elter, Kaggle). Model: 3,006,818
parameters, 8.1 GFLOPs, 6.0MB on disk. Test set: 801 images, 1,621 instances, 6 classes.
Dataset split: 80/10/10 random split, seed 42.

Source: code/detection/yolov8_pcb.md

| Metric | Value |
|---|---|
| mAP@0.5 | 0.9896 |
| mAP@0.5:0.95 | 0.6025 |
| Precision | 0.9769 |
| Recall | 0.9837 |

Inference speed: 3.9ms per image on Tesla T4. The 0.39 gap between mAP@0.5 and
mAP@0.5:0.95 reflects difficulty in tight box localization for small defects
(approximately 3-7% of image width).

---

## Segmentation Results

U-Net trained on the Carvana Image Masking Challenge dataset. Binary segmentation of car
silhouettes from RGB images at 512 x 512 resolution. Train/validation split: 90/10
random, seed 42. Loss: 0.3 * BCE + 0.7 * Dice.

Sources: code/segmentation/outputs/training_curves.png (training validation curve),
code/segmentation/outputs/evaluation_results.csv (per-image test set metrics)

| Split | Metric | Value |
|---|---|---|
| Validation (training run, epoch 20) | Dice | approximately 0.99 |
| Test set | Mean Dice | pending |
| Test set | Mean IoU | pending |

Per-image Dice and IoU for the evaluation set are recorded in
code/segmentation/outputs/evaluation_results.csv. No pre-computed aggregate scalar was
produced by the evaluation notebook (outputs are stripped). The training validation Dice
value of approximately 0.99 is read from the training curves plot at epoch 20.