# Efficient Architectures

---

## Compound Scaling

**Paper:** EfficientNet: Rethinking Model Scaling for CNNs — Tan & Le, 2019
**arXiv:** https://arxiv.org/abs/1905.11946

### Background

When you want a more powerful CNN, you have three independent axes to scale:

| Axis       | Meaning                                  | Example                        |
|------------|------------------------------------------|--------------------------------|
| **Width**  | More filters per layer                   | ResNet-50 → wider ResNet-50    |
| **Depth**  | More layers                              | ResNet-50 → ResNet-101         |
| **Resolution** | Larger input image               | 224×224 → 320×320              |

Prior work scaled these independently (ResNet scales depth; WideResNet scales
width). The question EfficientNet asks: is there a principled way to scale all
three together?

---

### The Compound Scaling Method

**Key observation:** Width, depth, and resolution are not independent.
- A deeper network needs higher resolution to benefit from the extra depth
  (more layers can capture finer patterns only if the input is detailed enough).
- A wider network needs more depth to integrate the richer per-layer features.

**Empirical finding:** If you increase FLOPs by φ (a scaling coefficient), the
optimal allocation of that budget is:

```
depth:      d = α^φ
width:      w = β^φ
resolution: r = γ^φ

subject to: α · β² · γ² ≈ 2
            α ≥ 1, β ≥ 1, γ ≥ 1
```

The `β²` and `γ²` terms reflect that FLOPs scale quadratically with width and
resolution (doubling width → 4× more multiplications per layer; doubling
resolution → 4× more spatial positions).

**The two-step recipe:**
1. Fix φ=1, grid-search α, β, γ subject to the constraint. This finds the best
   allocation of a 2× FLOP budget.
2. Scale α, β, γ by arbitrary φ to hit any target compute budget.

The baseline architecture (EfficientNet-B0) is found via NAS; then B1–B7 are
just compound-scaled versions of B0 with increasing φ.

---

### Why It Beats Naive Scaling

**Naive scaling** means increasing one axis alone until you hit diminishing
returns, then stopping (or arbitrarily picking a number):

```
naive depth scaling:      accuracy gains saturate — gradients degrade in very deep networks even with ResNets; 
                          adding more layers stops helping around 1000 layers on CIFAR.

naive width scaling:      wide-but-shallow networks fail to capture high-level
                          features requiring hierarchical composition across layers.

naive resolution scaling: at some point the network lacks the depth/width to
                          process the extra detail; pixels become wasted compute.
```

**Compound scaling keeps the axes balanced**, so each axis can do the work it
is suited for:
- Resolution provides fine detail for filters to detect.
- Width provides enough filters to model the rich patterns in that detail.
- Depth provides the hierarchy to compose those patterns into semantics.

**Concrete evidence from the paper:**

| Scaling strategy          | ImageNet Top-1 (at ~1.8B FLOPs) |
|---------------------------|----------------------------------|
| Depth only (d)            | ~79.6%                           |
| Width only (w)            | ~79.2%                           |
| Resolution only (r)       | ~79.0%                           |
| Compound (d + w + r)      | ~81.6%                           |

+2pp at the same FLOP budget purely from coordinated scaling.

**Intuition:** Scaling one axis is like buying a better camera lens while keeping
a bad sensor (resolution without width), or adding more processing cores while
starving them of data (depth without resolution). Compound scaling buys all three
in balance.

---

### EfficientNet Results

| Model           | ImageNet Top-1 | Params | FLOPs  |
|-----------------|---------------|--------|--------|
| ResNet-50       | 76.0%         | 26M    | 4.1B   |
| ResNet-152      | 77.8%         | 60M    | 11.3B  |
| EfficientNet-B0 | 77.1%         | 5.3M   | 0.39B  |
| EfficientNet-B4 | 82.9%         | 19M    | 4.2B   |
| EfficientNet-B7 | 84.3%         | 66M    | 37B    |

B0 matches ResNet-50 at 8.5× fewer parameters and 10× fewer FLOPs. B4 beats
ResNet-152 at the same FLOP budget with 3× fewer parameters.

---

## Why Does Compound Scaling Beat Naive Scaling?


> Naive scaling hits diminishing returns because the scaled axis becomes the
> bottleneck. Compound scaling avoids bottlenecks by keeping width, depth, and
> resolution in a fixed empirically-optimal ratio as compute scales.

---

## When Does ConvNeXt Beat ViT?

**Paper:** A ConvNet for the 2020s — Liu et al., 2022
**arXiv:** https://arxiv.org/abs/2201.03545

### Background

After ViT (2020) and Swin Transformer (2021), the field largely concluded that
attention-based architectures were fundamentally superior to CNNs for vision.
ConvNeXt challenged this directly: starting from ResNet-50, systematically
applying every training and design improvement associated with transformers,
and asking whether the resulting pure CNN could match them.

The answer was yes — and the specific conditions where it wins or matches ViT
are informative.

---

### The ConvNeXt Modernisation Roadmap

Starting from ResNet-50 → ResNet-50-A → ... → ConvNeXt, each step is ablated:

| Change                                | Accuracy gain |
|---------------------------------------|---------------|
| Training recipe (epochs, augment, AdamW) | +2.7pp    |
| Stage ratio (3:4:6:3 → 3:3:9:3)      | +0.6pp        |
| Patchify stem (4×4 non-overlap conv)  | +0.1pp        |
| Depthwise separable conv              | +0.3pp        |
| Inverted bottleneck (wide → narrow)   | +0.1pp        |
| Large kernel (3×3 → 7×7 depthwise)   | +0.5pp        |
| Activation: ReLU → GELU              | +0.0pp        |
| Fewer activations (one per block)     | +0.4pp        |
| Fewer normalizations (BN → LN, one)  | +0.5pp        |
| Separate downsampling layers          | +0.4pp        |
| **Total**                             | **+5.7pp**    |

The final ConvNeXt-T matches Swin-T at identical FLOPs and parameters.

This table documents how Liu et al. (2022) transformed a standard ResNet-50 into ConvNeXt-T by applying one change at a time, measuring the ImageNet accuracy gain from each modification in isolation. The baseline ResNet-50 sits around 76.1% top-1 accuracy; the final ConvNeXt-T reaches ~82%.

- **Training recipe (+2.7pp):** The biggest single gain comes from not changing the architecture at all, just training better: 300 epochs instead of 90, stronger augmentations (Mixup, CutMix, RandAugment), and AdamW instead of SGD. This alone closes most of the gap with Swin Transformers, which were trained with these modern recipes from the start.
- **Stage ratio 3:4:6:3 → 3:3:9:3 (+0.6pp):** — ResNet distributes blocks evenly across its four stages. Swin Transformers heavily front-load computation in stage 3. Mirroring that ratio improves accuracy because more capacity is allocated where feature maps are most informative.
- **Patchify stem (+0.1pp):** ResNet uses a 7×7 conv + maxpool to downsample early. ConvNeXt replaces this with a 4×4 non-overlapping convolution (borrowed from ViT's patch embedding), which is simpler and slightly more accurate.
- **Depthwise separable conv (+0.3pp):** Standard convolutions apply a filter across all channels simultaneously. Depthwise separable conv splits this into a per-channel spatial filter followed by a 1×1 pointwise conv, dramatically reducing FLOPs while maintaining expressivity.
- **Inverted bottleneck (+0.1pp):** Traditional ResNet bottlenecks go wide→narrow→wide (compress then expand). ConvNeXt flips this to narrow→wide→narrow, matching the MobileNetV2 and Transformer FFN design where the hidden dimension is expanded.
- **Large kernel 3×3 → 7×7 depthwise (+0.5pp):** Arguably the most conceptually important change. A 7×7 depthwise conv has a receptive field analogous to the self-attention window in Swin, and because it's depthwise (cheap), the extra kernel size costs almost nothing in FLOPs. This is what gives ConvNeXt its "long-range" spatial mixing without attention.
- **ReLU → GELU (+0.0pp):** Transformers use GELU; switching to it here makes zero difference in accuracy, but the authors keep it for architectural consistency with the Transformer family.
- **Fewer activations (+0.4pp):** ResNets apply ReLU after every conv. Transformers apply activation only once per block (inside the FFN). Reducing activations to one per block gives the network more linear capacity and improves accuracy non-trivially.
- **BN → LN, one normalization per block (+0.5pp):** Batch Normalization depends on batch statistics and behaves differently at train vs. inference. Layer Normalization (used in all Transformers) normalizes per-sample, is more stable, and here replaces multiple BN layers with a single LN per block.
- **Separate downsampling layers (+0.4pp):** ResNet performs downsampling inside residual blocks (via strided conv). ConvNeXt separates this into explicit 2×2 strided conv layers between stages, which is cleaner and lets each stage operate at a fixed resolution throughout.

- The key takeaway is that modern training recipes account for nearly half the total gain, and the architectural changes are each small and individually motivated by Transformer design principles — ConvNeXt is essentially "what if we trained a ResNet like a Transformer and borrowed its structural choices one by one."

---

### When ConvNeXt Wins (or Matches) ViT

#### 1. Limited Data / Small- to Medium-Scale Datasets

ViT's global self-attention has no spatial inductive bias, it must learn that
nearby pixels are more related than distant ones entirely from data. This hurts
on small datasets.

ConvNeXt's depthwise convolution is translation-equivariant by design. This
inductive bias is free supervision that reduces sample complexity.

**Rule of thumb:**
- Dataset < ~1M images → ConvNeXt ≥ ViT
- Dataset ~1M–14M (ImageNet-21k) → roughly equal with proper pretraining
- Dataset > 100M (JFT-300M) → ViT pulls ahead (more data, less need for priors)

#### 2. Dense Prediction Tasks (Detection, Segmentation)

ViT produces a single-scale feature map (all patches at one resolution). To use
ViT for dense tasks you need adaptations like feature pyramid networks or window
partitioning (Swin), which add complexity and hyperparameters.

ConvNeXt produces a natural multi-scale feature hierarchy (like ResNet), slots
directly into FPN-based detectors (Mask R-CNN, Cascade R-CNN) without
modification, and matches or exceeds Swin Transformer:

| Backbone     | COCO AP (box) | COCO AP (mask) |
|--------------|---------------|----------------|
| Swin-T       | 50.4          | 43.7           |
| ConvNeXt-T   | 50.4          | 43.7           |
| Swin-B       | 51.9          | 45.0           |
| ConvNeXt-B   | 52.7          | 45.6           |

ConvNeXt-B narrowly wins at the same scale. The hierarchy is a genuine advantage.

#### 3. Inference Efficiency (Throughput / Latency)

ViT's self-attention is O(n²) in sequence length (image tokens). For high-resolution
inputs or dense tasks, this quadratic cost dominates.

ConvNeXt's depthwise conv is O(n · k²) where k is kernel size (7 here) — linear
in sequence length. At high resolutions ConvNeXt is substantially faster:

- ConvNeXt-B: ~563 images/sec (A100)
- ViT-B/16: ~383 images/sec (A100, same resolution)

For deployment to edge / embedded hardware: attention is hard to accelerate;
convolution has decades of hardware support (cuDNN, ONNX, TFLite, CoreML).
ConvNeXt benefits from this entire ecosystem immediately.

#### 4. When You Cannot Use Large-Scale Pretraining

Swin and ViT-based models benefit enormously from pretraining on ImageNet-21k or
larger. For a domain where large pretraining sets do not exist
(medical imaging, satellite imagery, niche industrial data), ConvNeXt's inductive
biases give a head start that the data volume cannot compensate for.

#### 5. Fine-tuning Stability

ViT is sensitive to learning rate, weight decay, and augmentation choices, especially when fine-tuning on smaller datasets.
Suboptimal hyperparameters cause training instability. ConvNeXt, trained with
the same AdamW recipe as transformers, is more forgiving due to the BN→LN switch, which 
retains transformer-like stability while the convolutional structure provides
implicit regularisation via weight sharing.

---

### When ViT Wins

- **Very large pretraining data** (JFT-300M+, internet-scale): attention's ability
  to model arbitrary long-range dependencies dominates over convolutional locality.
- **Tasks requiring global reasoning** from the first layer: e.g., image-text
  matching, retrieval, where you want the image representation to integrate
  global context without the hierarchical bottleneck of a pyramid.
- **MAE / DINO-style self-supervised pretraining**: masked autoencoding works
  naturally on ViT's patch tokens; it requires architectural changes for
  hierarchical models.

---

### Summary Table

| Condition                                 | Winner       |
|-------------------------------------------|--------------|
| Small/medium dataset (< 1M images)        | ConvNeXt     |
| Large dataset (ImageNet-21k+)             | Roughly equal|
| Massive data (JFT-300M+)                  | ViT          |
| Dense prediction (detection/segmentation) | ConvNeXt     |
| High-resolution input                     | ConvNeXt     |
| Edge / embedded deployment                | ConvNeXt     |
| Global reasoning / cross-modal            | ViT          |
| MAE-style self-supervised pretraining     | ViT          |

---

### Conclusion

ConvNeXt's main contribution is that **most of ViT's gains over ResNet came from training improvements and design
decisions, not from self-attention itself**. When applying those same improvements to a CNN, the gap closes almost entirely.

This is directly relevant to your portfolio work: when you see a ViT beat a ResNet,
ask whether the comparison is fair (same training recipe, same augmentation, same
number of parameters). ConvNeXt was the paper that made this question unavoidable.

---

*References:*
- *EfficientNet: https://arxiv.org/abs/1905.11946*
- *ConvNeXt: https://arxiv.org/abs/2201.03545*
- *Swin Transformer: https://arxiv.org/abs/2103.14030*