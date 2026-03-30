# Efficient Architectures & Linear Probe Benchmark

---

## 1. What Was Done and Why

The goal was to compare four pretrained ImageNet backbones (ResNet-18, ViT-Tiny,
EfficientNet-B0, and ConvNeXt-Tiny) on CIFAR-10 without fine-tuning, in order to
isolate **representation quality** from training dynamics.

The protocol is a **linear probe**: freeze the backbone entirely, extract features for
all 60k images, L2-normalise them, then fit a logistic regression classifier. Accuracy
reflects how linearly separable the backbone's feature space is on a downstream task,
which is the standard evaluation used in the self-supervised learning literature
(SimCLR, MoCo, DINO) and equally valid for supervised pretrained models.

`timm.create_model(name, pretrained=True, num_classes=0)` loads ImageNet weights and
strips the classification head, returning the pooled feature vector directly.

---

## 2. benchmark_architecture.py

```
CIFAR-10 (224×224, ImageNet normalisation)
        │
        ▼
timm backbone (frozen, num_classes=0)
        │
        ▼
 feature vectors  ──► L2 normalise ──► LogisticRegression (C=0.316)
        │                                        │
   inference timer                          top-1 accuracy
  (200 runs, batch=1, CPU median)
```

**Key implementation decisions:**

- `num_classes=0` strips the head inside timm, no manual surgery needed.
- L2 normalisation before logistic regression is correct here because cosine similarity
  is the appropriate distance metric for embedding spaces, and normalisation makes the
  regularisation parameter C scale-invariant across models with different feature
  magnitudes.
- `C=0.316` is `10^(-0.5)`, a standard weak-regularisation value for linear probes.
- Inference timing uses **median over 200 runs** (not mean) to be robust against OS
  scheduling spikes, which are common in CPU benchmarks.
- Images are resized to 224×224 with ImageNet mean/std normalisation. This is required
  because all four models were pretrained at this resolution with this normalisation,
  using CIFAR-10's native 32×32 would produce out-of-distribution activations.

---

## 3. Benchmark Results

| Model           | Top-1 Acc (%) | Params (M) | Inference (ms/img, CPU) | Acc/Param |
|-----------------|:-------------:|:----------:|:-----------------------:|:---------:|
| ResNet-18       | 83.83         | 11.2       | 26.8                    | 7.5       |
| ViT-Tiny        | 80.72         | 5.5        | 26.9                    | 14.7      |
| EfficientNet-B0 | 90.06         | 4.0        | 29.2                    | **22.5**  |
| ConvNeXt-Tiny   | 95.08         | 27.8       | 90.6                    | 3.4       |

**Protocol:** frozen ImageNet backbone + logistic regression (C=0.316, L2-normalised
features), input 224×224. Inference time: median over 200 runs, batch=1, CPU.

*Parameter counts reflect the backbone only (`num_classes=0`), with the classification
head removed. Published full-model counts (including head) are slightly higher:
ResNet-18 ~11.7M, ViT-Tiny ~5.7M, EfficientNet-B0 ~5.3M, ConvNeXt-Tiny ~28.6M.*

---

## 4. Architecture Analysis

### What does the compound coefficient φ control in EfficientNet

φ (phi) is a user-specified integer that controls how compute budget is distributed
across three scaling dimensions simultaneously:

```
depth      ∝  1.20^φ
width      ∝  1.10^φ
resolution ∝  1.15^φ
```

The standard approach before EfficientNet was to scale only one dimension, either
deeper networks (ResNet), wider networks (WideResNet), or higher resolution inputs,
treating each as independent. Tan & Le (2019) showed these dimensions interact:
a deeper network benefits from wider layers to avoid representational bottlenecks,
and wider layers benefit from higher resolution to capture finer spatial detail.

The three base coefficients (1.20, 1.10, 1.15) were found via a small grid search
with φ=1. NAS was used separately to find the B0 baseline architecture; the grid
search for α, β, γ is a subsequent step applied on top of that fixed baseline.
B1 through B7 apply φ=1 through 7 to the **same architecture**, producing a family
of models that scales smoothly along the accuracy-efficiency frontier. EfficientNet-B0
(φ=0) has 4.0M backbone parameters; EfficientNet-B7 (φ=7) has 66M — the same
structural blueprint, scaled by compound φ.

The benchmark result supports this directly: EfficientNet-B0 achieves 90.06% with
only 4.0M backbone parameters, the best accuracy-per-parameter ratio (22.5) in the
group, demonstrating that balanced scaling extracts more value per parameter than the
simple depth scaling ResNet uses.

---

### How does ConvNeXt differ from a standard ResNet block at the implementation level

The ConvNeXt paper (Liu et al., 2022) modernised ResNet-50 into ConvNeXt by applying
one change at a time and measuring the accuracy gain at each step:

| Change                                              | Gain    |
|-----------------------------------------------------|---------|
| Training recipe (epochs, augment, AdamW)            | +2.7pp  |
| Stage ratio (3:4:6:3 → 3:3:9:3)                    | +0.6pp  |
| Patchify stem (4×4 non-overlap conv)                | +0.1pp  |
| ResNeXt-style (depthwise conv + width 64→96)        | +1.0pp  |
| Inverted bottleneck (narrow → wide → narrow)        | +0.1pp  |
| Move depthwise conv to first position               | -0.1pp  |
| Kernel size 3×3 → 7×7 (depthwise)                  | +0.7pp  |
| Activation: ReLU → GELU                             | +0.0pp  |
| Fewer activations (one per block)                   | +0.4pp  |
| Fewer normalizations (BN → LN, one per block)       | +0.5pp  |
| Separate downsampling layers                        | +0.4pp  |
| **Total**                                           | **+6.4pp** |

At the implementation level, a ConvNeXt block differs from a ResNet BasicBlock in
five concrete ways:

1. **Depthwise separable conv with a large kernel.** ResNet uses 3×3 standard conv
   (all channels simultaneously). ConvNeXt uses a 7×7 depthwise conv (one filter per
   channel) followed by 1×1 pointwise convs. The 7×7 depthwise kernel is cheap in
   FLOPs because it operates per-channel, yet gives a receptive field comparable to
   the local attention window in Swin Transformer.

2. **Inverted bottleneck.** ResNet bottlenecks go wide→narrow→wide (512→128→512),
   compressing first. ConvNeXt goes narrow→wide→narrow (C → 4C → C), matching the
   FFN expansion ratio used in Transformers. The expensive depthwise conv operates
   in the narrow (C) space.

3. **Single GELU activation per block.** ResNet applies ReLU after every conv.
   ConvNeXt applies one GELU inside the expanded (4C) projection, following the
   Transformer FFN pattern. This gives the network more linear capacity across layers.

4. **Single LayerNorm per block, replacing BatchNorm.** ResNet uses BatchNorm after
   each conv, which depends on batch statistics and behaves differently at train vs.
   inference. ConvNeXt uses one LayerNorm (per-sample, stable, inference-identical)
   at the start of each block.

5. **Separate downsampling layers between stages.** ResNet performs spatial downsampling
   inside residual blocks via strided convolutions. ConvNeXt uses explicit 2×2 strided
   LayerNorm+Conv layers between stages, so each stage operates at a fixed resolution
   throughout.

Crucially: nearly half of the total +6.4pp gain came from the training recipe alone
(+2.7pp), with zero architectural changes. The paper's deepest finding is that the
ConvNet vs Transformer debate is partly a training recipe debate.

---

### Which model gives the best accuracy-per-parameter ratio

**EfficientNet-B0: 22.5 acc/param** (90.06% / 4.0M), the highest in the group by a
wide margin.

For context:
- ViT-Tiny achieves 14.7 with 5.5M backbone parameters — a higher ratio than
  ResNet-18, but 3pp lower accuracy than EfficientNet-B0 at 1.4× fewer parameters.
- ConvNeXt-Tiny achieves the highest absolute accuracy (95.08%) but at 27.8M
  parameters costs 7× more than EfficientNet-B0 for a 5pp gain — ratio of 3.4.
- ResNet-18 at 7.5 reflects the cost of depth-only scaling without EfficientNet's
  balanced compound approach.

The compound scaling result is not surprising in isolation, but the magnitude is:
EfficientNet-B0 was pretrained on ImageNet-1k and its features transfer to CIFAR-10
at 90% accuracy from 4M backbone parameters. This is a direct demonstration of φ
working as intended — balanced scaling extracts more representational capacity per
parameter than any single-dimension scaling strategy.

---

## 5. Observations Worth Keeping

**ViT-Tiny inference parity.** In the first run, ViT-Tiny appeared faster than
ResNet-18 (19.6ms vs 41.2ms). After fixing the params bug and re-running, both land
at ~27ms. The first run likely included model loading or JIT warmup overhead that
artificially inflated ResNet's time. The corrected result makes more physical sense:
attention and convolution at this scale have comparable CPU cost.

**ConvNeXt's strong transfer.** 95.08% from a linear probe on CIFAR-10 is high. This
reflects two things: the modern training recipe (RandAugment, Mixup, AdamW, 300
epochs) builds richer ImageNet representations than the original ResNet recipe, and
CIFAR-10's 10 classes are a strict subset of ImageNet-1k concepts, so transfer is
nearly zero-shot for many classes.

**What this benchmark does not measure.** Linear probe accuracy is not fine-tuned
accuracy. EfficientNet-B0's 90% would likely increase to 96–97% with full fine-tuning.
The benchmark answers "how good is the backbone's representation?" — the fine-tuning
comparison is a separate experiment conducted on a GPU (Kaggle T4).

---

## 6. References

- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for CNNs.
  *ICML 2019*. arXiv:1905.11946
- Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022).
  A ConvNet for the 2020s. *CVPR 2022*. arXiv:2201.03545
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
  recognition. *CVPR 2016*. arXiv:1512.03385