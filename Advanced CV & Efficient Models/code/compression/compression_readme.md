# Advanced CV and Efficient Models

This module covers efficient neural network architectures and model compression techniques applied to CIFAR-10
image classification. All implementations are from scratch using PyTorch primitives. Pretrained backbones from
`timm` are used only for the architecture benchmark, where the goal is evaluation rather than implementation.

## Contents

- [Architecture Benchmarks](#architecture-benchmarks)
- [Model Compression](#model-compression)
  - [Knowledge Distillation](#knowledge-distillation)
  - [Post-Training Quantization](#post-training-quantization)
  - [Unstructured Pruning](#unstructured-pruning)
- [Compression Summary](#compression-summary)
- [Visualizations](#visualizations)
- [Code Structure](#code-structure)
- [Reproducing Results](#reproducing-results)
- [References](#references)

---

## Architecture Benchmarks

A linear probe evaluation was conducted over frozen pretrained backbones from `timm` to compare
accuracy-per-parameter efficiency across four architectures on CIFAR-10. Features were extracted from the
frozen encoder, L2-normalized, and fed into a logistic regression classifier (C=0.316). This protocol
isolates representational quality from task-specific fine-tuning.

| Architecture | Top-1 Accuracy (%) | Parameters (M) | Accuracy / Param (%) |
|---|---|---|---|
| ConvNeXt-Tiny | 95.08 | 28.6 | 3.32 |
| EfficientNet-B0 | 90.06 | 4.0 | 22.52 |
| ResNet-18 | 83.83 | 11.7 | 7.17 |
| ViT-Tiny | 80.72 | 5.7 | 14.16 |

EfficientNet-B0 achieves the best accuracy-per-parameter ratio at 22.52%/M, validating the compound scaling
hypothesis of Tan and Le (2019). ConvNeXt-Tiny leads in absolute accuracy, consistent with the finding of Liu
et al. (2022) that ConvNet designs remain competitive with vision transformers on smaller datasets and compute
budgets. ViT-Tiny's comparatively low accuracy reflects its well-documented data hunger: attention mechanisms
require large-scale pretraining to learn competitive visual representations, and the frozen linear probe setting
exacerbates this gap.

---

## Model Compression

### Knowledge Distillation

A lightweight SmallCNN student (170K parameters, depthwise separable convolutions) was trained to mimic a
ResNet-18 teacher (11.17M parameters) using the distillation objective of Hinton et al. (2015):

```
L = alpha * T^2 * KL(softmax(z_t / T) || softmax(z_s / T)) + (1 - alpha) * CE(z_s, y)
```

where `z_s` and `z_t` are student and teacher logits respectively, `T` is the temperature, and `alpha`
balances the distillation and task losses. The KL divergence is taken with the teacher distribution as the
reference (first argument), which is the convention established in Hinton et al. (2015) and matches the
standard PyTorch implementation using `F.kl_div(log_softmax(z_s/T), softmax(z_t/T))`.

**Hyperparameters:** T=4, alpha=0.30, 30 epochs, Adam (lr=1e-3), cosine annealing.

| Model | Parameters | Size (MB) | Latency (ms/batch) | Throughput (img/s) | Top-1 (%) |
|---|---|---|---|---|---|
| ResNet-18 (teacher) | 11,173,962 | 42.7 | 744.6 | 172 | 93.43 |
| SmallCNN (distilled) | 170,378 | 0.65 | 55.8 | 2,294 | 78.34 |
| SmallCNN (CE baseline) | 170,378 | 0.65 | 54.6 | 2,343 | 78.27 |

The distilled student achieves a **65.6x parameter reduction** and **13.3x inference speedup** over the teacher
at a top-1 accuracy cost of 15.09 percentage points. The distillation gain over the CE baseline (0.07pp) is
within run-to-run variance at this training length, indicating the student architecture is the binding
constraint rather than the training signal. Longer training or a larger student would be expected to widen
the distillation gap.

The choice of T=4 is supported empirically by the soft target analysis. At T=1 the teacher's distribution
is near-degenerate for correctly classified samples (max probability > 0.94), providing almost no inter-class
relational signal. At T=4 the distribution exposes semantically meaningful structure: the airplane sample
assigns non-trivial probability to ship and bird, reflecting shared visual features. At T=8 the distribution
approaches uniform, which dilutes the signal.

### Post-Training Quantization

Post-training quantization (PTQ) was applied to a ResNet-18 checkpoint (93.13% top-1 on CIFAR-10) using
PyTorch's quantization pipeline. Both dynamic and static schemes were evaluated. All benchmarks run on CPU;
single-sample latency (batch=1) is reported for consistent comparison.

**Quantization map:** `x_int = clamp(round(x_float / scale) + zero_point, q_min, q_max)`

Per-channel symmetric quantization was used for weights and per-tensor asymmetric unsigned INT8 for
activations. The affine quantization formulation follows Jacob et al. (2018); per-channel weight granularity
follows the best-practice guidance of Nagel et al. (2021). Calibration used 100 CIFAR-10 training batches
with PyTorch's default MinMax observers. Conv-BN fusion was applied prior to calibration to eliminate
intermediate requantization steps.

*Note: the PTQ and pruning experiments used this 93.13% checkpoint. The distillation teacher above
(93.43%) is from a separate training run with a different random seed; both are ResNet-18 on CIFAR-10.*

| Method | Size (MB) | Latency (ms) | Top-1 (%) | Accuracy Delta (pp) |
|---|---|---|---|---|
| FP32 baseline | 42.70 | 9.61 | 93.13 | -- |
| INT8 dynamic (Linear only) | 42.69 | 13.20 | 93.12 | -0.01 |
| INT8 static (PTQ) | 10.80 | 5.56 | 93.02 | -0.11 |

Dynamic quantization produces no meaningful size reduction because ResNet-18 is almost entirely Conv2d layers;
PyTorch does not support dynamic quantization for Conv2d, so only the final fully-connected layer (5,120 of
11.17M parameters) is quantized. The resulting latency increase reflects dequantization overhead on a
convolution-dominated model where the quantized fc layer contributes negligibly to total compute.

Static PTQ achieves a **3.95x size reduction** (42.70 MB to 10.80 MB) and **1.73x latency reduction**
(9.61 ms to 5.56 ms) with only 0.11pp accuracy degradation, consistent with the near-zero accuracy loss
reported for well-calibrated INT8 PTQ on ResNet-class models in Nagel et al. (2021).

### Unstructured Pruning

L1 unstructured magnitude pruning (Han et al., 2015) was applied to all Conv2d weight tensors at three
sparsity levels, followed by 5 epochs of fine-tuning with a small recovery learning rate (SGD, lr=1e-3).

| Target Sparsity | Actual Sparsity | Top-1 Before FT (%) | Top-1 After FT (%) | Delta vs Baseline (pp) |
|---|---|---|---|---|
| 0% (baseline) | 0.0% | 93.13 | 93.13 | -- |
| 20% | 20.0% | 93.16 | 93.42 | +0.29 |
| 40% | 40.0% | 92.95 | 93.38 | +0.25 |
| 60% | 60.0% | 91.53 | 93.27 | +0.14 |

At 20% and 40% sparsity, fine-tuned accuracy exceeds the dense baseline. Removing the lowest-magnitude weights
acts as a mild regularizer, suggesting the model was slightly overparameterized for CIFAR-10. At 60% sparsity
the pre-fine-tuning degradation is 1.6pp, but 5 epochs of recovery training closes this to 0.14pp below
baseline.

These gains do not translate to runtime improvements. L1 unstructured pruning zeroes individual weights within
a dense float32 tensor; after mask removal the weight matrix is structurally identical to the original,
and standard CPU matrix multiplication kernels do not exploit sparsity. Structured pruning (filter or channel
removal) would be required to reduce model size and latency.

---

## Compression Summary

All methods applied to the same ResNet-18 checkpoint (93.13% baseline). Latency figures are single-sample
CPU measurements except where noted. The student model latency is a batch-128 figure and is not directly
comparable to the single-sample rows above.

| Method | Parameters (M) | Size (MB) | Latency (ms) | Top-1 (%) |
|---|---|---|---|---|
| ResNet-18 baseline | 11.17 | 42.70 | 9.61 | 93.13 |
| + Dynamic quantization | 11.17 | 42.69 | 13.20 | 93.12 |
| + Static PTQ | 11.17 | 10.80 | 5.56 | 93.02 |
| + Unstructured pruning 40% | 11.17 | 42.70 | 9.61 | 93.38 |
| Student (distillation)* | 0.17 | 0.65 | 55.8* | 78.34 |

*Batch-128 latency from `inference_benchmark.py`. Static PTQ and dynamic quantization latencies are
single-sample measurements from `quantization.py` and are not directly comparable to the batch figure.

Static PTQ is the most practical compression technique for this architecture: it delivers real runtime gains
with negligible accuracy cost and requires no retraining. Distillation yields the greatest parameter reduction
but at a substantial accuracy cost that reflects the capacity gap between teacher and student, not the
distillation method itself.

---

## Visualizations

All plots are saved to `experiments/` and `code/compression/plots/`.

**`plots/distillation/soft_target_distributions.png`**
Softmax output of the teacher at T=1, 2, 4, 8 for four representative CIFAR-10 samples. Illustrates how
temperature controls the entropy of soft targets and exposes inter-class similarity structure that hard labels
discard. The airplane sample at T=4 shows non-trivial probability mass on ship and bird, reflecting shared
visual features between these classes.

**`plots/distillation/distill_loss_breakdown.png`**
Per-epoch training loss decomposed into KD loss (T^2 * KL divergence) and CE loss over 30 epochs of
distillation. Shows the relative contribution of each component and convergence behavior.

**`plots/distillation/val_accuracy_comparison.png`**
Validation accuracy over 30 epochs comparing distillation vs CE baseline student training.

**`experiments/pruning/sparsity_accuracy.png`**
Top-1 accuracy at each sparsity level before and after fine-tuning. Shows the regularization effect at
low sparsity and the recovery capacity of 5-epoch fine-tuning at 60% sparsity.

---

## Code Structure

```
Advanced CV & Efficient Models/
├── code/
│   ├── efficient_architectures/
│   │   ├── efficientnet.py           # EfficientNet-B0 from scratch (compound scaling)
│   │   └── convnext.py               # ConvNeXt-Tiny from scratch
│   └── compression/
│       ├── distillation.py           # KD loss, SmallCNN student, KnowledgeDistillationTrainer
│       ├── train_distillation.py     # Training script: distill and CE baseline modes
│       ├── quantization.py           # PTQ: dynamic and static quantization benchmarks
│       ├── pruning.py                # L1 unstructured pruning across sparsity levels
│       ├── inference_benchmark.py    # Latency and throughput: teacher vs student
│       └── visualize_soft_targets.py # Soft target distributions at T=1,2,4,8
└── experiments/
    ├── benchmark_architecture.py     # Linear probe benchmark over timm backbones
    └── pruning/
        └── sparsity_accuracy.png
```

---

## Reproducing Results

All scripts read data from `<repo root>/data/` via `Path(__file__).resolve().parents[N] / "data"`.
Run all commands from the repo root (`ml-research-12weeks/`).

```bash
# Architecture linear probe benchmark
python "Advanced CV & Efficient Models/experiments/benchmark_architecture.py"

# Knowledge distillation (distill + CE baseline, ~5 hours CPU)
python "Advanced CV & Efficient Models/code/compression/train_distillation.py"

# Inference benchmark: teacher vs student
python "Advanced CV & Efficient Models/code/compression/inference_benchmark.py"

# PTQ: dynamic and static quantization
python "Advanced CV & Efficient Models/code/compression/quantization.py" \
    --checkpoint "computer-vision-foundations/code/pytorch_cnn/best_resnet18_cifar10.pth"

# L1 unstructured pruning (recommended on GPU)
python "Advanced CV & Efficient Models/code/compression/pruning.py"

# Soft target visualization
python "Advanced CV & Efficient Models/code/compression/visualize_soft_targets.py"
```

**Environment:** Python 3.12, PyTorch (CPU local / T4 GPU on Kaggle/Colab), timm, scikit-learn.

```bash
pip install -e .
```

---

## References

- Hinton, G., Vinyals, O., and Dean, J. (2015). Distilling the Knowledge in a Neural Network.
  arXiv:1503.02531. https://arxiv.org/abs/1503.02531
- Han, S., Pool, J., Tran, J., and Dally, W. (2015). Learning both Weights and Connections for
  Efficient Neural Networks. *NeurIPS*. https://arxiv.org/abs/1506.02626
- Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., and Kalenichenko, D.
  (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only
  Inference. *CVPR*. https://arxiv.org/abs/1712.05877
- Nagel, M., Fournarakis, M., Amjad, R. A., Bondarenko, Y., van Baalen, M., and Blankevoort, T.
  (2021). A White Paper on Neural Network Quantization.
  https://arxiv.org/abs/2106.08295
- Tan, M. and Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural
  Networks. *ICML*. https://arxiv.org/abs/1905.11946
- Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., and Xie, S. (2022). A ConvNet for
  the 2020s. *CVPR*. https://arxiv.org/abs/2201.03545