# Efficient Architectures & Linear Probe Benchmark

---

## 1. What Was Done and Why

The goal was to compare four pretrained ImageNet backbones (ResNet-18, ViT-Tiny,
EfficientNet-B0, and ConvNeXt-Tiny) on CIFAR-10 without fine-tuning, in order to
isolate **representation quality** from training dynamics.

The protocol is a **linear probe**: freeze the backbone entirely, extract features for
all 60k images, L2-normalise them, then fit a logistic regression classifier. Here,
accuracy reflects how linearly separable the backbone's feature space is on CIFAR-10,
which is a standard evaluation in the self-supervised learning literature and is
also widely used for supervised ImageNet-pretrained models.[file:92][web:85]

`timm.create_model(name, pretrained=True, num_classes=0)` loads ImageNet weights and
strips the classification head, returning a pooled feature vector from the backbone.[file:92]

---

## 2. benchmark_architecture.py

```text
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

- `num_classes=0` strips the head inside timm, so no manual surgery is needed.[file:92]
- L2 normalisation before logistic regression is standard in linear probe setups
  and effectively makes the classifier operate on cosine-normalised features,
  while reducing sensitivity of the regularisation parameter C to feature scale
  across models.[file:92][web:85]
- `C=0.316` is `10^(-0.5)`, a weak-regularisation value commonly used for linear
  probes; in this setup it avoids overfitting while allowing the probe to exploit
  most of the representational capacity.[file:92]
- Inference timing uses the **median over 200 runs** (not the mean) to be robust
  against OS scheduling spikes, which are common in CPU benchmarks.[file:92]
- Images are resized to 224×224 with ImageNet mean/std normalisation, matching
  the pretraining setup used for all four models. Using CIFAR-10's native 32×32
  without rescaling would produce out-of-distribution activations.[file:92][web:17]

---

## 3. Benchmark Results

All accuracies and latencies in this section are from our own CIFAR-10 linear probe
experiments, not from the original architecture papers.[file:92][web:17][web:22]

| Model           | Top-1 Acc (%) | Params (M, backbone) | Inference (ms/img, CPU) | Acc/Param |
|-----------------|:-------------:|:--------------------:|:-----------------------:|:---------:|
| ResNet-18       | 83.83         | 11.2                 | 26.8                    | 7.5       |
| ViT-Tiny        | 80.72         | 5.5                  | 26.9                    | 14.7      |
| EfficientNet-B0 | 90.06         | 4.0                  | 29.2                    | **22.5**  |
| ConvNeXt-Tiny   | 95.08         | 27.8                 | 90.6                    | 3.4       |

**Protocol:** frozen ImageNet backbone + logistic regression (C=0.316, L2-normalised
features), input 224×224. Inference time: median over 200 runs, batch=1, CPU.[file:92]

*Parameter counts reflect the backbone only (`num_classes=0`), with the classification
head removed. Published full-model counts (including heads) are slightly higher:
ResNet-18 ≈11.7M, ViT-Tiny ≈5.7M, EfficientNet-B0 5.3M, ConvNeXt-Tiny ≈28.6M.[file:92][web:17][web:22]*

---

## 4. Architecture Analysis

### What does the compound coefficient φ control in EfficientNet

In EfficientNet, depth, width, and resolution are scaled jointly using a compound
coefficient φ:[file:92][web:17]

```text
depth:      d = α^φ
width:      w = β^φ
resolution: r = γ^φ

subject to: α · β² · γ² ≈ 2
            α ≥ 1, β ≥ 1, γ ≥ 1
```

The constants α, β, γ are found via grid search at φ = 1 on a fixed B0 baseline,
subject to the approximate FLOP constraint above.[web:17] In many summaries they are
approximated as α≈1.2, β≈1.1, γ≈1.15, but the key point is that they are learned
rather than hand-picked.[web:30]

Prior to EfficientNet, standard practice was to scale a single axis:

- deeper networks (e.g., increasing layers in ResNet),
- wider networks (e.g., WideResNet),
- higher input resolution,

with limited attention to how these dimensions interact.[web:17] EfficientNet shows that
these choices interact: deeper networks benefit from wider layers to avoid
representational bottlenecks, and wider layers benefit from higher resolution to
capture finer spatial detail.[web:17]

B0 is obtained via NAS; α, β, γ are then tuned on top of B0, and φ parameterises
a family of models B1–B7 at different compute budgets.[web:17] In our `timm` setup, the
EfficientNet-B0 backbone (without classification head) has about 4.0M parameters,
while EfficientNet-B7’s backbone has about 66M; the original paper reports 5.3M
and 66M parameters respectively when including the classifier head.[file:92][web:17]

Within this benchmark, EfficientNet-B0 achieves 90.06% CIFAR-10 linear probe
accuracy with 4.0M backbone parameters, yielding the highest accuracy-per-parameter
ratio (22.5) in the group.[file:92] This is consistent with the original claim that compound
scaling yields strong accuracy–efficiency trade-offs compared to depth-only scaling.[web:17]

---

### How does ConvNeXt differ from a standard ResNet block at the implementation level

The ConvNeXt paper (Liu et al., 2022) modernised a ResNet-50-style backbone into
ConvNeXt by applying one change at a time and measuring the gain at each step.[web:22]
For exact stepwise values, see their Table 10; below we summarise the changes and
their approximate individual effects:[file:92][web:22]

| Change                                              | Approx. gain (pp) |
|-----------------------------------------------------|-------------------|
| Training recipe (epochs, augment, AdamW)            | +2.7              |
| Stage ratio (3:4:6:3 → 3:3:9:3)                    | ≈+0.5–0.6         |
| Patchify stem (4×4 non-overlap conv)                | ≈+0.1–0.2         |
| Depthwise conv + width 64→96                        | net +1.9          |
| Inverted bottleneck (narrow → wide → narrow)        | +0.1              |
| Move depthwise conv to first position               | about -0.7        |
| Kernel size 3×3 → 7×7 (depthwise)                  | +0.7              |
| Activation: ReLU → GELU                             | ≈0.0              |
| Fewer activations (one per block)                   | +0.6              |
| Fewer normalizations (BN → LN, one per block)       | +0.2              |
| Separate downsampling layers                        | +0.5              |
| **Total**                                           | ≈+6.0 to +6.4     |

At the implementation level, a ConvNeXt block differs from a ResNet BasicBlock in
five concrete ways:[file:92][web:22]

1. **Depthwise separable conv with a large kernel.** ResNet uses 3×3 standard conv
   (all channels simultaneously). ConvNeXt uses a 7×7 depthwise conv (one filter per
   channel) followed by 1×1 pointwise convs. The 7×7 depthwise kernel is cheap in
   FLOPs because it operates per-channel, yet gives a receptive field comparable to
   the local attention window in Swin Transformer.[web:22]

2. **Inverted bottleneck.** ResNet bottlenecks go wide→narrow→wide (e.g., 512→128→512),
   compressing first.[web:22] ConvNeXt goes narrow→wide→narrow (C → 4C → C), matching the
   FFN expansion ratio used in Transformers. The expensive depthwise conv operates
   in the narrower (C) space.[web:22]

3. **Single GELU activation per block.** ResNet applies ReLU after every conv.
   ConvNeXt applies one GELU inside the expanded (4C) projection, following the
   Transformer FFN pattern and increasing the network’s effective linear capacity.[web:22]

4. **Single LayerNorm per block, replacing BatchNorm.** ResNet uses BatchNorm after
   each conv, which depends on batch statistics and behaves differently at train vs.
   inference.[web:22] ConvNeXt uses one LayerNorm (per-sample, identical at train and
   inference) at the start of each block.[web:22]

5. **Separate downsampling layers between stages.** ResNet performs spatial
   downsampling inside residual blocks via strided convolutions.[web:22] ConvNeXt uses
   explicit 2×2 strided conv layers with LayerNorm between stages, so each stage
   operates at a fixed resolution.[web:22]

Crucially: a substantial fraction of the total improvement comes from the modern
training recipe (+2.7pp) rather than architectural changes alone.[web:22] The paper’s
central message is that when ConvNets are given Transformer-era training and design
choices, they remain highly competitive with Transformer backbones.[web:22]

---

### Which model gives the best accuracy-per-parameter ratio

**EfficientNet-B0:** 22.5 acc/param (90.06% / 4.0M backbone parameters), the highest
in this benchmark.[file:92]

For context:

- **ViT-Tiny** achieves 14.7 with 5.5M backbone parameters — a higher ratio than
  ResNet-18, but about 3pp lower accuracy than EfficientNet-B0 at ~1.4× fewer
  parameters.[file:92]
- **ConvNeXt-Tiny** achieves the highest absolute accuracy (95.08%) but, at 27.8M
  backbone parameters, uses roughly 7× more parameters than EfficientNet-B0 for a
  ≈5pp accuracy gain (ratio ≈3.4).[file:92]
- **ResNet-18** at 7.5 reflects the cost of depth-only scaling without EfficientNet's
  balanced compound approach.[file:92]

These ratios are specific to this CIFAR-10 linear probe setup, but they are
consistent with prior work showing that EfficientNet architectures offer strong
accuracy–efficiency trade-offs compared to earlier CNNs.[web:17]

---

## 5. Observations Worth Keeping

**ViT-Tiny inference parity.** In an earlier run, ViT-Tiny appeared faster than
ResNet-18 (19.6ms vs 41.2ms). After fixing a parameter-counting bug and re-running
with warmup excluded, both land at ~27ms. The earlier discrepancy was likely due to
initial model loading or JIT warmup overhead affecting ResNet more strongly. The
corrected result is more consistent with expectations: at this scale, attention and
convolution have comparable CPU cost per image.[file:92]

**ConvNeXt's strong transfer.** 95.08% from a linear probe on CIFAR-10 is high. This
likely reflects the combination of a modern training recipe (RandAugment, Mixup,
AdamW, long schedules) and the fact that many CIFAR-10 classes (e.g., cats, dogs,
airplanes, ships) overlap closely with ImageNet categories, making transfer
comparatively easy.[file:92][web:17][web:22]

**What this benchmark does not measure.** Linear probe accuracy is not fine-tuned
accuracy. In typical CIFAR-10 experiments, fully fine-tuned modern backbones reach
mid-90s accuracy, so we expect full fine-tuning to improve upon the linear-probe
numbers reported here.[web:57] Quantifying that gap would require a separate set of
fine-tuning experiments (for example, on a GPU such as a T4), which are outside
this note’s scope.[file:92]

---

## 6. References

- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for
  convolutional neural networks. *ICML 2019*. arXiv:1905.11946[web:17]
- Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022).
  A ConvNet for the 2020s. *CVPR 2022*. arXiv:2201.03545[web:22]
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for
  image recognition. *CVPR 2016*. arXiv:1512.03385[web:63]