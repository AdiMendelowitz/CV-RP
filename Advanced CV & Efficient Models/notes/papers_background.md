# Background Paper Notes: Efficiency, Detection & Compression

> **Purpose:** Skim-level reference notes for four foundational papers.
> Each section covers motivation, core ideas, key mechanics, results, limitations,
> and what to carry forward into implementation work.

---

## Table of Contents

1. [MobileNetV3 (2019) — Howard et al.](#1-mobilenetv3-2019)
2. [YOLOv1 (2016) — Redmon et al.](#2-yolov1-2016)
3. [DETR (2020) — Carion et al.](#3-detr-2020)
4. [Deep Compression (2016) — Han et al.](#4-deep-compression-2016)

---

## 1. MobileNetV3 (2019)

**Full title:** Searching for MobileNetV3
**Authors:** Howard et al. (Google)
**Venue:** ICCV 2019
**arXiv:** https://arxiv.org/abs/1905.02244

---

### Motivation

MobileNetV1/V2 introduced depthwise separable convolutions and inverted residuals
to dramatically shrink model size for mobile deployment. V3 asks: instead of
hand-designing the next improvement, can we use **automated search** to find the
optimal architecture given real hardware latency constraints?

The key tension being solved: accuracy vs. latency on mobile CPUs — not FLOPs,
actual wall-clock milliseconds on an ARM chip.

---

### Core Contributions

#### 1. Two-Stage Architecture Search

**Stage 1 — Platform-Aware NAS (MnasNet-style):**
- Searches for the macro-level block structure (layer types, kernel sizes, expansion ratios).
- Optimises for a multi-objective reward: `ACC(m) × [LAT(m) / TAR]^w` where
  `TAR` is target latency and `w` is a weighting factor.
- Produces the baseline skeleton of MobileNetV3.

**Stage 2 — NetAdapt:**
- Fine-grained, layer-by-layer adaptation of the NAS skeleton.
- Iteratively proposes small changes (reduce filter count, remove a layer) that
  satisfy a latency budget, then retrains briefly to measure accuracy impact.
- Complements NAS by tuning widths without re-running the full search.

The combination lets global structure and local widths be optimised jointly,
something neither algorithm achieves alone.

#### 2. Hard-Swish Activation (h-swish)

The Swish activation `x · σ(x)` empirically outperforms ReLU but is expensive
due to the sigmoid. The authors approximate it as:

```
h-swish(x) = x · ReLU6(x + 3) / 6
```

This is piecewise linear, easy to quantise, and fast on fixed-point hardware.
Applied only in the deeper layers (where channel count is high and the per-op
cost is amortised across more computation).

Swish/h-swish benefits: smoother loss landscape, better gradient flow in deep
networks compared to ReLU. Related to GELU used in transformers.

#### 3. Squeeze-and-Excitation (SE) Blocks

Borrowed from SENet and selectively added to certain bottleneck blocks.

```
SE block: Global Average Pool → FC → ReLU → FC → h-sigmoid → scale channels
```

Each channel gets a scalar gate learned from global context, letting the network
recalibrate feature importance at negligible parameter cost.
MobileNetV3 uses SE in the expansion phase of the bottleneck (after the pointwise
expand convolution) to keep latency low.

#### 4. Redesigned Head and Stem

**Stem:** First convolution (16 filters, stride 2) kept but subsequent layers
thinned using NetAdapt — the early stem was over-parameterised in V2.

**Head (classifier):** V2 used a 1×1 conv → global average pool → 1×1 conv
sequence. V3 moves the expensive 1×1 conv *after* pooling, reducing the spatial
resolution it operates on from 7×7 to 1×1. Saves latency with no accuracy cost.

#### 5. Two Variants

| Variant             | Target Use Case           |
|---------------------|---------------------------|
| MobileNetV3-Large   | High-resource mobile (e.g. flagship phones) |
| MobileNetV3-Small   | Low-resource (e.g. wearables, IoT)          |

#### 6. LR-ASPP Segmentation Decoder

For dense prediction tasks (semantic segmentation), the authors propose
**Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP)**:
- Replaces the full ASPP used in DeepLabV3+ with a lightweight alternative.
- Uses a single global average pooling branch + 1×1 conv instead of multiple
  atrous rates.
- 30% faster than MobileNetV2's R-ASPP at similar Cityscapes accuracy.

---

### Key Results

| Model                 | ImageNet Top-1 | vs V2 (acc) | vs V2 (latency) |
|-----------------------|---------------|-------------|-----------------|
| MobileNetV3-Large     | ~75.2%        | +3.2%       | −15%            |
| MobileNetV3-Small     | ~67.4%        | +4.6%       | −5%             |

- COCO detection: MobileNetV3-Large is 25% faster at comparable mAP to V2.
- Cityscapes segmentation: LR-ASPP is 30% faster than V2's R-ASPP.

---

### Limitations

- NAS and NetAdapt search is expensive to run (requires TPU time); practitioners
  use the released checkpoints rather than re-searching.
- The architecture is highly hardware-specific — latency gains on ARM CPUs may
  not transfer to GPUs or DSPs without re-running search on the target platform.
- SE blocks add memory bandwidth cost that partially cancels latency savings on
  some hardware.

---

### What to Take Forward

- **h-swish** is a practical activation for any deployment-focused model.
- **Separating NAS (structure) + NetAdapt (widths)** is a reusable search strategy.
- **SE blocks in bottlenecks** are a cheap, reliable accuracy booster.
- When targeting mobile deployment: optimise for *measured latency*, not FLOPs.

---

### Lineage

```
MobileNetV1 (depthwise separable convolutions)
    ↓
MobileNetV2 (inverted residuals + linear bottlenecks)
    ↓
MobileNetV3 (NAS + NetAdapt + h-swish + SE)
    ↓
EfficientNet (compound scaling, same era, different design philosophy)
```

---

---

## 2. YOLOv1 (2016)

**Full title:** You Only Look Once: Unified, Real-Time Object Detection
**Authors:** Redmon, Divvala, Girshick, Farhadi (UW / Allen AI)
**Venue:** CVPR 2016
**arXiv:** https://arxiv.org/abs/1506.02640

---

### Motivation

Prior detectors (R-CNN, DPM) decompose detection into a pipeline of separate
stages: region proposal → feature extraction → classification → bounding box
regression. Each stage is trained independently and the full pipeline is slow
(R-CNN: ~50s/image at the time).

**The YOLO insight:** Frame object detection as a **single regression problem**
over the image. One network, one pass, one loss, trained end-to-end.
"You Only Look Once" at the image.

---

### Architecture Overview

```
Input image (448×448×3)
    ↓
24 Convolutional layers (alternating 1×1 and 3×3)
    ↓
2 Fully connected layers
    ↓
Output tensor: S × S × (B × 5 + C)
```

- `S = 7`: The image is divided into a 7×7 grid.
- `B = 2`: Each cell predicts 2 bounding boxes.
- `C = 20`: 20 class probabilities (Pascal VOC).
- Output shape: `7 × 7 × 30`.

Backbone is inspired by GoogLeNet; pretraining on ImageNet at 224×224 then
fine-tuning detection at 448×448 (doubled resolution for better small-object
localisation).

---

### Core Detection Mechanism

**Grid cell responsibility:** Each grid cell is responsible for detecting objects
whose *centre* falls inside that cell. This is the key spatial assignment rule.

**Each bounding box prediction:**
```
(x, y, w, h, confidence)
```
- `(x, y)`: centre offset relative to the grid cell, normalised to [0, 1].
- `(w, h)`: relative to full image width/height, also in [0, 1].
- `confidence = P(object) × IoU(pred, gt)`: combines objectness and localisation quality.

**Class prediction:** Each cell predicts a *single* set of C class probabilities,
shared across both predicted boxes. This is a key simplification that creates the
"one object per cell" limitation.

**At test time:** Final class-specific confidence = `P(class | object) × confidence`.

---

### Loss Function

The loss is a hand-crafted sum of squared errors (SSE) with careful weighting:

```
L = λ_coord · Σ_cell Σ_box [box responsible] · [(x̂−x)² + (ŷ−y)²]
  + λ_coord · Σ [box responsible] · [(√ŵ − √w)² + (√ĥ − √h)²]   ← sqrt for scale invariance
  + Σ [box responsible] · (Ĉ − C)²
  + λ_noobj · Σ [box NOT responsible] · (Ĉ − C)²
  + Σ_cell Σ_class (p̂_i − p_i)²

where λ_coord = 5, λ_noobj = 0.5
```

**Design choices explained:**
- `λ_coord = 5`: Upweights localisation loss since most cells contain no object.
- `λ_noobj = 0.5`: Downweights confidence loss from background cells to prevent
  them dominating.
- Square-root of width/height: Makes the loss less sensitive to absolute box scale
  — a 10px error on a large box should penalise less than on a small one.

---

### Why It Works (and Why It Fails)

**Strengths:**
- End-to-end training means the whole network learns detection jointly.
- Global context: the fully connected layers see the entire image, reducing false
  positives from background patches (YOLO produces far fewer "ghost" detections
  than region-proposal methods).
- Fast: base model runs at 45fps; Fast YOLO (9 conv layers) at 155fps.

**Weaknesses:**
- **One object per cell:** Two objects whose centres fall in the same cell can only
  have one detected. Struggles with dense, small objects (e.g. a flock of birds).
- **Fixed prior:** No anchor priors — boxes are predicted from scratch, making
  localisation harder for unusual aspect ratios.
- **Scale inflexibility:** Grid resolution is fixed at 7×7, limiting spatial
  granularity.
- **Generalisation:** Performs worse on small objects than region-proposal methods
  at the time.

---

### Results (Pascal VOC 2007)

| Model       | mAP   | FPS  |
|-------------|-------|------|
| DPM v5      | 33.7% | ~0.07|
| R-CNN       | 66.0% | ~0.02|
| Fast R-CNN  | 70.0% | 0.5  |
| YOLO        | 63.4% | 45   |
| Fast YOLO   | 52.7% | 155  |

YOLO trades ~7pp mAP for a 90× speedup over Fast R-CNN. First detector to
demonstrate real-time performance with reasonable accuracy.

---

### Historical Impact

YOLOv1 started a lineage that dominates real-time detection to this day:

```
YOLOv1 (2016) → YOLOv2/YOLO9000 (anchor priors, multi-scale)
    → YOLOv3 (Darknet-53, multi-scale prediction heads)
    → YOLOv4 (Bag of Freebies/Specials)
    → YOLOv5 / YOLOv8 (Ultralytics, production-grade)
    → RT-DETR (transformer-based real-time, 2023)
```

Each successor addressed specific YOLOv1 limitations. The one-shot regression
paradigm and grid-cell concept remain in every version.

---

### What to Take Forward

- The "detection as regression" paradigm is the foundation for all single-stage
  detectors.
- The multi-component weighted loss is a template for detection loss design.
- Understanding *why* YOLO struggles with small objects and grid limitations
  motivates anchor-based detectors (YOLOv2+), feature pyramids (FPN), and
  eventually anchor-free transformers (DETR).

---

---

## 3. DETR (2020)

**Full title:** End-to-End Object Detection with Transformers
**Authors:** Carion, Massa, Synnaeve, Usunier, Kirillov, Zagoruyko (Facebook AI)
**Venue:** ECCV 2020
**arXiv:** https://arxiv.org/abs/2005.12872

---

### Motivation

Every major object detector before DETR relies on **hand-engineered inductive
biases**:
- Anchor boxes encoding prior beliefs about object shapes/scales.
- Non-Maximum Suppression (NMS) to deduplicate predictions.
- Region proposal networks or grid-cell assignment rules.

These components require domain expertise to tune and prevent fully end-to-end
training. DETR asks: can transformers eliminate all of this, replacing heuristics
with learned attention over global context?

---

### Architecture

```
Image
  ↓
CNN Backbone (ResNet-50 or ResNet-101)  → feature map H/32 × W/32 × C
  ↓
Positional Encoding (fixed 2D sine)
  ↓
Transformer Encoder (6 layers, standard MHSA + FFN)
  ↓
Transformer Decoder (6 layers)
  ← Object Queries (N=100 learned embeddings, one per "slot")
  ↓
FFN per query → (class logits + bounding box)
  ↓
N predictions (most → "no object")
```

---

### Key Mechanisms

#### 1. Object Queries

`N = 100` learned position embeddings fed as the decoder's target sequence.
Each query learns to specialise in detecting objects at certain positions or scales.
Parallel decoding: all queries are processed simultaneously, not autoregressively.

The queries have no explicit spatial prior — the model learns what they attend to
entirely from data. This is what eliminates the need for anchor design.

#### 2. Bipartite Matching Loss

The critical innovation. Given `N` predictions and `M` ground-truth objects (`M ≤ N`):

1. Compute a cost matrix: `C[i,j] = class_cost + bbox_cost` for every
   (prediction `i`, ground truth `j`) pair.
2. Find the optimal **one-to-one** assignment using the **Hungarian algorithm**
   (polynomial-time optimal matching).
3. Compute loss only on matched pairs; all unmatched predictions are supervised
   as the "no object" class.

This guarantees unique predictions — each ground-truth box is matched to exactly
one query. **NMS becomes unnecessary** because duplicate detections of the same
object would compete and only one would win the match.

**Loss components:**

```
L_match = −log(p̂_σ(i)(c_i))  +  L_box(b_i, b̂_σ(i))
L_box   = λ_iou · L_GIoU + λ_L1 · ||b_i − b̂_σ(i)||_1
```

GIoU loss is scale-invariant and handles the case where boxes do not overlap.
L1 on normalised box coordinates handles absolute size.

#### 3. Transformer Encoder

Standard multi-head self-attention over the flattened feature map. Every spatial
position attends to every other position — this gives global context before
the decoder runs. The encoder learns to group features corresponding to the same
object across spatial locations.

#### 4. Transformer Decoder (Cross-Attention)

Each object query attends to the encoder output via cross-attention, then refines
via self-attention with other queries (allowing them to model object-object
relationships). The decoder produces one set prediction per query.

---

### Panoptic Segmentation Extension

DETR naturally extends to panoptic segmentation by adding a pixel-level FFN over
the decoder's attention maps. For each detected box, the corresponding attention
maps (multi-head, multi-scale) are upsampled and merged to produce a mask.
No separate mask head design needed — a clean generalisation of the same framework.

---

### Results (COCO 2017 val)

| Model                | AP    | AP_S  | AP_M  | AP_L  | GFLOPs | FPS  |
|----------------------|-------|-------|-------|-------|--------|------|
| Faster R-CNN R50 FPN | 42.0  | 26.6  | 45.4  | 53.4  | 180    | ~15  |
| DETR R50             | 42.0  | 20.5  | 45.8  | 61.1  | 86     | 28   |
| DETR R101            | 43.5  | 21.9  | 48.0  | 61.8  | 152    | 20   |

**Reading the table:**
- On overall AP, DETR matches Faster R-CNN — remarkable given far less engineering.
- DETR is significantly better on large objects (`AP_L`) due to global self-attention.
- DETR is significantly worse on small objects (`AP_S`) — a known weakness.

---

### Limitations

- **Training is slow:** Requires 500 epochs to converge on COCO (Faster R-CNN
  converges in ~36). The Hungarian matching provides weak gradient signal at the
  start when predictions are random.
- **Small object detection is weak:** The backbone downsamples by 32×; the
  transformer lacks built-in multiscale processing (addressed by Deformable DETR).
- **Quadratic attention cost:** Standard MHSA is O(n²) in sequence length; the
  feature map is large, making the encoder slow.
- **N=100 fixed slots:** Cannot detect more than 100 objects per image.

These limitations led directly to Deformable DETR, DAB-DETR, DN-DETR, and DINO
(the detection model, not the SSL method), which progressively fixed each issue.

---

### Why DETR Matters Beyond Its Numbers

DETR demonstrated that:
1. Detection pipelines can be reduced to a pure transformer architecture.
2. Set prediction + bipartite matching is a general, elegant way to handle variable
   numbers of outputs without hand-crafted post-processing.
3. Transformers can reason about global object relationships in a way CNNs cannot.

The paradigm became the foundation for most modern open-vocabulary and grounded
detection models (Grounding DINO, OWL-ViT, etc.).

---

### Lineage

```
Transformer ("Attention Is All You Need", 2017)
    ↓
DETR (2020) — set prediction, bipartite matching
    ↓
Deformable DETR (2021) — deformable attention, 10× faster convergence
    ↓
DAB-DETR / DN-DETR (2022) — anchor-based queries, denoising training
    ↓
DINO-Det (2022) — contrastive denoising, SOTA on COCO
    ↓
Grounding DINO / OWL-ViT (2023) — open-vocabulary
```

---

### What to Take Forward

- **Bipartite matching / Hungarian algorithm** is the key algorithmic idea —
  understand it deeply; it reappears in tracking, pose estimation, and generative
  matching (e.g. DDPM alignment).
- **Object queries as learned slots** is a general pattern for variable-cardinality
  outputs without NMS.
- When DETR underperforms: it is almost always `AP_S`; the fix is multi-scale
  features (FPN equivalent in transformer land = deformable attention).

---

---

## 4. Deep Compression (2016)

**Full title:** Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding
**Authors:** Han, Mao, Dally (Stanford / NVIDIA)
**Venue:** ICLR 2016 (Best Paper / Oral)
**arXiv:** https://arxiv.org/abs/1510.00149

---

### Motivation

In 2016, deploying neural networks on mobile and embedded hardware (phones, IoT,
FPGAs) was severely limited by:
- **Memory:** Models like AlexNet (240MB) and VGG-16 (552MB) exceed on-chip SRAM.
  Accessing off-chip DRAM is 100× more energy-expensive.
- **Compute:** Embedded CPUs have limited FP32 throughput.

The goal: compress models to fit in on-chip SRAM, enabling both faster inference
and dramatically lower energy consumption — without any accuracy loss.

---

### The Three-Stage Pipeline

```
Pretrained Network
        ↓
  [Stage 1: Pruning]
        ↓
  Sparse Network (retrain to recover accuracy)
        ↓
  [Stage 2: Trained Quantization + Weight Sharing]
        ↓
  Quantized Sparse Network (retrain centroids)
        ↓
  [Stage 3: Huffman Coding]
        ↓
  Compressed Binary File
```

Each stage is applied sequentially and each enables the next to be more effective.

---

### Stage 1: Pruning

**Mechanism:**
1. Train the network normally.
2. Remove all connections whose absolute weight falls below a threshold τ.
3. Retrain the remaining sparse network (the surviving weights can grow to
   compensate for removed connections).
4. Optionally iterate (prune → retrain → prune → ...).

**Result:** Connections are pruned but surviving weights retain full precision.

The sparse network is stored as a **Compressed Sparse Row (CSR)** matrix, with
indices coded using relative (delta) offsets to compress the index representation.

**Compression achieved by pruning alone:**

| Network  | Original connections | After pruning | Reduction |
|----------|----------------------|---------------|-----------|
| AlexNet  | 61M                  | 6.7M          | 9×        |
| VGG-16   | 138M                 | 10.3M         | 13×       |

Key insight: most weight *mass* is in FC layers (which can be pruned heavily),
while most *computation* is in conv layers (which are pruned more conservatively
to preserve accuracy).

---

### Stage 2: Trained Quantization (Weight Sharing)

After pruning, each weight is stored as a full 32-bit float. Stage 2 reduces
precision by clustering weights and sharing cluster centroids.

**Mechanism — k-means clustering:**
1. Cluster all weights in each layer into `k` clusters (e.g. k=256 → 8-bit index).
2. Replace each weight with the index of its nearest centroid.
3. **Retrain (fine-tune) only the centroids** using gradient accumulation:
   - Gradients of all weights in a cluster are summed and applied to that cluster's centroid.
   - Individual weights remain fixed as cluster indices.

**Storage layout:**

```
Before: each weight = 32 bits
After:  each weight = log2(k) bits for cluster index
        + (k × 32 bits) for centroid table (amortised over all weights in the layer)
```

For k=256: 32 bits → 8 bits per weight = 4× reduction.
For k=32: 32 bits → 5 bits per weight.

**Centroid initialisation matters:**
- Random init: poor, many weights fall in low-density regions.
- Density-based init: more centroids where weights are dense (near zero). Better
  but still misses large weights.
- **Linear init** (paper's recommendation): equally spaced centroids between
  min and max weight. Ensures large weights, which have outsized gradient impact,
  each get their own centroid. Empirically best.

**Precision per layer type:**
- Convolutional layers: 8 bits (more sensitive to quantisation).
- Fully-connected layers: 4–5 bits (over-parameterised, very robust).

---

### Stage 3: Huffman Coding

After weight sharing, the distribution of cluster indices is non-uniform — most
weights fall in a small number of clusters (near-zero values dominate).
**Huffman coding** assigns shorter codes to more frequent indices, achieving
lossless compression of the index stream.

Additional: the sparse CSR indices (from pruning) are also Huffman-coded.

**Compression from Huffman alone:** ~20–30% additional reduction on top of
quantization.

---

### Overall Results

| Network  | Original | After Compression | Ratio | Accuracy loss |
|----------|----------|-------------------|-------|---------------|
| AlexNet  | 240MB    | 6.9MB             | 35×   | ~0%           |
| VGG-16   | 552MB    | 11.3MB            | 49×   | ~0%           |

Both models now fit in a typical on-chip SRAM cache (8–16MB), enabling inference
without DRAM access.

**Inference speedup (layerwise):** 3×–4× on CPU, GPU, mobile GPU.
**Energy efficiency:** 3×–7× better (dominated by memory access savings).

---

### Why This Works: The Redundancy Argument

Neural networks, especially FC layers, are massively over-parameterised. The
pruning step reveals that most connections carry near-zero weights and contribute
negligibly to the output. Quantisation works because the remaining weights
cluster naturally around a small number of values. Huffman coding works because
those clusters are not uniformly used.

The staged approach is crucial: pruning first makes quantisation easier (fewer
weights, simpler distribution); quantisation reduces precision before Huffman,
which then efficiently encodes the skewed distribution.

---

### Limitations

- **Irregular sparsity:** Unstructured pruning (removing individual connections)
  creates sparse matrices that are hard to accelerate on standard GPU hardware
  (which prefers dense tensor operations). Speedup requires sparse BLAS support
  or specialised hardware.
- **Structured alternatives:** Later work (channel pruning, filter pruning) removes
  entire filters to produce dense models that accelerate naturally on any GPU —
  at the cost of higher accuracy loss for the same compression ratio.
- **Retraining required:** Each stage requires fine-tuning, adding training cost.
- **Quantisation-Aware Training (QAT)** has largely superseded the k-means
  approach for deployment; modern frameworks (PyTorch, TFLite) support QAT natively
  with straight-through estimators for gradients.

---

### Relation to Modern Compression

```
Deep Compression (2016)
  ├── Pruning branch → Lottery Ticket Hypothesis (2019), magnitude pruning, L0
  ├── Quantization branch → QAT (PyTorch), INT8 inference (TensorRT, ONNX Runtime)
  │                      → Post-Training Quantization (PTQ), GPTQ (2022)
  └── Coding branch → Entropy coding in model compression (less used now)

Knowledge Distillation (Hinton, 2015) — orthogonal but complementary approach
MobileNets / EfficientNet — architecture-level efficiency instead of post-hoc compression
```

---

### What to Take Forward

- **Pruning → Quantization → Coding** as a three-stage pipeline is still the
  conceptual framework used in production model optimisation.
- For GPU deployment: prefer **structured pruning** (channel/filter) over
  unstructured to get real speedup without specialised hardware.
- **INT8 quantisation with QAT** is the modern standard; `torch.quantization` and
  TensorRT both implement this. The centroid/k-means approach is mainly historical.
- The core insight — *most weights are redundant and can be represented with far
  fewer bits without accuracy loss* — underpins all of modern model compression
  and is directly relevant to LLM quantisation (INT4, GGUF, AWQ, etc.).

---

---

## Cross-Paper Themes

| Theme                        | MobileNetV3 | YOLOv1 | DETR | Deep Compression |
|------------------------------|:-----------:|:------:|:----:|:----------------:|
| Efficiency / deployment      | ✓           |        |      | ✓                |
| End-to-end training          |             | ✓      | ✓    |                  |
| Eliminating hand-engineering |             |        | ✓    |                  |
| Architecture search / NAS    | ✓           |        |      |                  |
| Quantization                 | ✓ (h-swish) |        |      | ✓                |
| Real-time inference          | ✓           | ✓      |      | ✓                |
| Set prediction / global ctx  |             |        | ✓    |                  |

**Progression arc to understand:**

```
YOLOv1 (detection as regression, fast but limited)
    → DETR (detection as set prediction, no heuristics, transformer-native)
    → MobileNetV3 / Deep Compression (efficiency, deploy either on edge hardware)
```

These four papers together cover: how to detect objects efficiently (YOLO),
how to detect them without hand-engineering (DETR), how to build efficient
backbones (MobileNetV3), and how to compress any model for deployment (Deep
Compression).

---

*Sources: all facts verified against paper abstracts (arXiv) and published results.*
*arXiv links: [1905.02244](https://arxiv.org/abs/1905.02244) · [1506.02640](https://arxiv.org/abs/1506.02640) · [2005.12872](https://arxiv.org/abs/2005.12872) · [1510.00149](https://arxiv.org/abs/1510.00149)*