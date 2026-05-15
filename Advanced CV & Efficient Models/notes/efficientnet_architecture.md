# Efficient Architectures

***

## Compound Scaling

**Paper:** EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks - Tan & Le, ICML 2019  
**arXiv:** https://arxiv.org/abs/1905.11946

### Background

When you want a more powerful CNN, you have three main axes to scale:

| Axis       | Meaning                                  | Example                        |
|------------|------------------------------------------|--------------------------------|
| **Width**  | More filters per layer                   | ResNet-50 → wider ResNet-50    |
| **Depth**  | More layers                              | ResNet-50 → ResNet-101         |
| **Resolution** | Larger input image               | 224×224 → 320×320              |

Prior work often scaled these independently, such as increasing depth in ResNet-style models or width in WideResNet-style models. The question EfficientNet asks is whether there is a principled way to scale all three together.

***

### The Compound Scaling Method

**Key observation:** Width, depth, and resolution are not independent in practice.
- A deeper network can benefit more from higher-resolution input because additional layers can exploit finer spatial structure.
- A wider network can benefit from sufficient depth so that richer intermediate features can be combined hierarchically.

**Empirical finding:** If you increase FLOPs by φ (a scaling coefficient), the paper proposes scaling:

```
depth:      d = α^φ
width:      w = β^φ
resolution: r = γ^φ

subject to: α · β² · γ² ≈ 2
            α ≥ 1, β ≥ 1, γ ≥ 1
```

The `β²` and `γ²` terms reflect that FLOPs scale approximately quadratically with width and resolution. The paper uses a two-step procedure: first search for α, β, and γ under a small fixed budget, then use φ to scale the model family to larger compute regimes.

**The two-step recipe:**
1. Fix φ=1, grid-search α, β, γ subject to the constraint. This finds the best allocation of about a 2× FLOP budget.
2. Scale α, β, γ by arbitrary φ to hit larger target compute budgets.

The baseline architecture (EfficientNet-B0) is found via NAS; then B1–B7 are compound-scaled versions of B0.

***

### Why It Beats Naive Scaling

**Naive scaling** means increasing one axis alone until gains diminish.

```text
naive depth scaling:      can run into diminishing returns when extra layers are
                          not matched by sufficient width or resolution.

naive width scaling:      can add channel capacity without enough hierarchical
                          depth to use it effectively.

naive resolution scaling: can add input detail that the network lacks capacity
                          to process efficiently.
```

**Compound scaling keeps the axes balanced**, so each axis is increased together rather than in isolation.
- Resolution supplies finer detail.
- Width increases representational capacity.
- Depth increases hierarchical feature composition.

**Concrete evidence from the paper:**

| Scaling strategy          | ImageNet Top-1 (at ~1.8B FLOPs) |
|---------------------------|----------------------------------|
| Depth only (d)            | ~79.1%                           |
| Width only (w)            | ~79.1%                           |
| Resolution only (r)       | ~79.1%                           |
| Compound (d + w + r)      | ~80.0%                           |

The exact values depend on the baseline family and table being referenced, but the paper consistently reports that compound scaling outperforms single-axis scaling at similar compute. The safest conclusion is empirical: coordinated scaling gives a better accuracy-efficiency tradeoff than one-dimensional scaling in the reported experiments.

**Intuition:** Scaling only one axis can leave the others as bottlenecks. Compound scaling reduces that mismatch by increasing width, depth, and resolution together in a fixed ratio.

***

### EfficientNet Results

| Model           | ImageNet Top-1 | Params | FLOPs  |
|-----------------|---------------|--------|--------|
| ResNet-50       | 76.0%         | 25.6M  | 4.1B   |
| ResNet-152      | 77.8%         | 60.2M  | 11.5B  |
| EfficientNet-B0 | 77.1%         | 5.3M   | 0.39B  |
| EfficientNet-B4 | 82.9%         | 19M    | 4.2B   |
| EfficientNet-B7 | 84.3%         | 66M    | 37B    |

B0 exceeds ResNet-50 accuracy with far fewer parameters and much lower FLOPs. B4 surpasses ResNet-152 in accuracy while using substantially fewer parameters and fewer FLOPs.

***

## Why Does Compound Scaling Beat Naive Scaling?

> Naive scaling tends to hit diminishing returns because one unscaled axis can become a bottleneck. Compound scaling improves the balance between width, depth, and resolution as compute increases.

***

## When Does ConvNeXt Beat ViT?

**Paper:** A ConvNet for the 2020s - Liu et al., CVPR 2022  
**arXiv:** https://arxiv.org/abs/2201.03545

### Background

After ViT (2020) and Swin Transformer (2021), Transformer-based vision models became dominant in many benchmark comparisons. ConvNeXt challenged the idea that this gap was mainly due to self-attention itself by starting from ResNet-50, systematically applying modern training and design changes, and evaluating whether the resulting pure CNN could match strong Transformer baselines.

The answer in the paper is that it often can. More precisely, ConvNeXt shows that a modernized ConvNet can match or surpass Swin on several standard benchmarks under matched training settings.

***

### The ConvNeXt Modernisation Roadmap

Starting from ResNet-50 → ResNet-50-A → ... → ConvNeXt, each step is ablated:

| Change                                | Accuracy gain |
|---------------------------------------|---------------|
| Training recipe (epochs, augment, AdamW) | +2.7pp    |
| Stage ratio (3:4:6:3 → 3:3:9:3)      | +0.54pp       |
| Patchify stem (4×4 non-overlap conv)  | +0.15pp       |
| ResNeXt-style depthwise conv          | -0.9pp        |
| Width 64→96                           | +1.9pp        |
| Inverted bottleneck                   | +0.1pp        |
| Move depthwise conv to first position | -0.72pp       |
| Large kernel (3×3 → 7×7 depthwise)   | +0.7pp        |
| Activation: ReLU → GELU              | +0.0pp        |
| Fewer activations (one per block)     | +0.65pp       |
| Fewer normalizations                  | +0.14pp       |
| BN → LN                              | +0.06pp       |
| Separate downsampling layers          | +0.50pp       |
| **Total**                             | **~+5.9pp from 76.1 to 82.0** |

The final ConvNeXt-T is reported at about 82.1% top-1 accuracy and slightly outperforms Swin-T while remaining in a similar parameter and FLOP regime.

This table documents how Liu et al. (2022) transformed a ResNet-style baseline into ConvNeXt through stepwise changes reported in Table 10. The most accurate interpretation is to treat these values as immediate incremental gains rather than as loosely grouped conceptual changes.

- **Training recipe (+2.7pp):** A major part of the gain comes from stronger optimization and augmentation rather than architecture alone.
- **Stage ratio (+0.54pp):** Redistributing blocks toward later stages improves performance modestly.
- **Patchify stem (+0.15pp):** Replacing the original early stem with a ViT-like patchifying stem gives a small gain.
- **Depthwise conv then width increase:** Depthwise convolution alone hurts accuracy, but increasing width more than recovers the loss.
- **Inverted bottleneck (+0.1pp):** This gives a small gain and aligns the block more closely with MobileNetV2/Transformer-style design.
- **Move depthwise conv earlier (-0.72pp):** This is a real drop in Table 10, larger than many simplified summaries report.
- **Large kernel (+0.7pp):** Increasing the depthwise kernel to 7×7 recovers the lost accuracy and improves spatial mixing.
- **ReLU → GELU (+0.0pp):** This has effectively no measurable effect in the paper’s ablation.
- **Fewer activations (+0.65pp):** Reducing activations per block yields a noticeable gain.
- **Fewer normalizations / BN → LN:** These are positive but smaller gains than many informal summaries suggest.
- **Separate downsampling layers (+0.50pp):** This final change helps produce the full ConvNeXt design.

The key takeaway is still that the training recipe contributes a large fraction of the gain, while the architectural modifications collectively close the remaining gap. That conclusion is supported by the paper, but some earlier summaries overstated individual deltas.

***

### When ConvNeXt Wins (or Matches) ViT

#### 1. Dense Prediction Tasks (Detection, Segmentation)

ConvNeXt produces a natural multi-scale feature hierarchy, which fits standard FPN-based detection and segmentation pipelines directly. In the paper’s COCO results, ConvNeXt matches or exceeds Swin at comparable scales:

| Backbone     | COCO AP (box) | COCO AP (mask) |
|--------------|---------------|----------------|
| Swin-T       | 50.4          | 43.7           |
| ConvNeXt-T   | 50.4          | 43.7           |
| Swin-B       | 51.9          | 45.0           |
| ConvNeXt-B   | 52.7          | 45.6           |

*Cascade Mask R-CNN, 3× schedule, COCO val2017 (Liu et al., 2022, Table 3).*

ConvNeXt-B slightly exceeds Swin-B under this evaluation setup. That is a supported result from the paper.

#### 2. Inference Efficiency (Throughput / Latency)

ConvNeXt’s depthwise convolutions scale linearly with token count for fixed kernel size, while full self-attention scales quadratically with sequence length. In the paper’s Appendix E benchmark on A100 GPUs, ConvNeXt achieves higher throughput than similarly sized Swin models, with the largest reported gap reaching about 49%.

That throughput result is benchmark-specific and should be stated as such. It supports the claim that ConvNeXt can be faster in the reported hardware setting, not that it is universally faster in every deployment scenario.

#### 3. When Careful Benchmark Matching Matters

The clearest conclusion from the paper is not a universal “ConvNeXt beats ViT” rule. It is that modernized CNNs remain highly competitive with Transformer backbones when the comparison is made under matched training recipes, parameter budgets, and evaluation settings.

***

### When ViT Wins

The ConvNeXt paper itself does not establish a universal rule for when ViT will always outperform ConvNeXt. Broader claims about massive-data scaling, global reasoning, or self-supervised learning may be plausible in the literature, but they should not be stated as direct conclusions from ConvNeXt alone without additional sources.

***

### Summary Table

| Condition                                               | Supported by ConvNeXt paper? |
|---------------------------------------------------------|------------------------------|
| ConvNeXt matches or exceeds Swin on several benchmarks  | Yes                          |
| ConvNeXt is strong for dense prediction tasks           | Yes                          |
| ConvNeXt can have higher A100 throughput than Swin      | Yes, in the reported setup   |
| ConvNeXt always beats ViT on small datasets             | No                           |
| ViT always wins on massive datasets                     | No, not established here     |
| ViT is always better for global reasoning               | No, not established here     |

***

### Conclusion

ConvNeXt’s central finding is that much of the performance gap between older ResNets and modern vision Transformers can be reduced by updating the training recipe and architectural design. When those improvements are applied to a CNN, the gap with strong Transformer baselines narrows substantially and can disappear on several benchmarks.

When comparing ViT against ResNet-family models, it is important to ask whether the comparison is fair: same training recipe, similar parameter counts, and similar evaluation settings. ConvNeXt made that comparison much harder to ignore.

***

*References:*
- *EfficientNet: https://arxiv.org/abs/1905.11946*
- *ConvNeXt: https://arxiv.org/abs/2201.03545*
- *Swin Transformer: https://arxiv.org/abs/2103.14030*