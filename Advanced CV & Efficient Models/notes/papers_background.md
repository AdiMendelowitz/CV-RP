# Background Paper Notes: Efficiency, Detection & Compression

---

## Table of Contents

1. [MobileNetV3 (2019) - Howard et al.](#1-mobilenetv3-2019)
2. [YOLOv1 (2016) - Redmon et al.](#2-yolov1-2016)
3. [DETR (2020) - Carion et al.](#3-detr-2020)
4. [Deep Compression (2016) - Han et al.](#4-deep-compression-2016)

---

## 1. MobileNetV3 (2019)

**Full title:** Searching for MobileNetV3  
**Authors:** Howard et al. (Google)  
**Venue:** ICCV 2019  
**arXiv:** https://arxiv.org/abs/1905.02244

---

### Motivation

MobileNetV1/V2 introduced depthwise separable convolutions and inverted residuals
to dramatically shrink model size for mobile deployment. MobileNetV3 asks: instead of
hand-designing the next improvement, can we use **automated search** to find an
architecture that directly optimises accuracy under real hardware latency constraints on mobile devices?[file:95][web:96]

The key tension: accuracy vs latency on mobile CPUs, measured in **actual wall-clock
milliseconds**, not just FLOPs.[file:95][web:96]

---

### Core Contributions

#### 1. Two-Stage Architecture Search

**Stage 1 – Platform-Aware NAS (MnasNet-style):**  

- Searches for macro-level block structure (layer types, kernel sizes, expansion ratios).[file:95][web:96]  
- Optimises a multi-objective reward of the form  
  \( \text{ACC}(m) \times \left(\frac{\text{LAT}(m)}{\text{TAR}}\right)^w \),  
  where `TAR` is a target latency and `w` trades off accuracy and latency.[file:95][web:96]  
- Produces a baseline MobileNetV3 backbone tailored to the target platform.[web:96]

**Stage 2 – NetAdapt:**  

- Fine-grained, layer-by-layer adaptation of the NAS skeleton.[file:95][web:96]  
- Iteratively proposes small changes (e.g., reduce filters, remove a layer) that respect a latency budget, then briefly retrains to measure accuracy impact.[file:95]  
- Complements NAS by tuning widths without re-running the full search.[file:95]

Together, NAS chooses the global structure while NetAdapt refines local widths under latency targets, a combination neither algorithm achieves alone.[file:95][web:96]

#### 2. Hard-Swish Activation (h-swish)

The Swish activation \(x \cdot \sigma(x)\) empirically outperforms ReLU but is expensive on mobile due to the sigmoid. MobileNetV3 introduces a piecewise-linear approximation:[file:95][web:96]

```text
h-swish(x) = x · ReLU6(x + 3) / 6
```

h-swish is easy to quantise, faster on fixed-point hardware, and is applied mainly in deeper layers where channel counts are high so the per-op cost is amortised.[file:95][web:96] Swish/h-swish improves gradient flow relative to ReLU and is related to GELU used in Transformers.[file:95]

#### 3. Squeeze-and-Excitation (SE) Blocks

MobileNetV3 borrows SE blocks from SENet and selectively inserts them into certain bottleneck blocks:[file:95][web:96]

```text
SE block: Global Average Pool → FC → ReLU → FC → h-sigmoid → scale channels
```

Each channel receives a learned scalar gate from global context, allowing the network to recalibrate channel importance at modest parameter cost.[file:95][web:96] MobileNetV3 applies SE in the expansion (high-channel) phase of bottlenecks to keep latency low.[file:95]

#### 4. Redesigned Head and Stem

- **Stem:** The first convolution (16 filters, stride 2) is retained, but early layers are thinned using NetAdapt; MobileNetV2’s stem is shown to be over-parameterised relative to its latency contribution.[file:95][web:96]  
- **Head:** MobileNetV2 uses a 1×1 conv → global average pool → 1×1 conv sequence. MobileNetV3 moves the expensive 1×1 conv **after** pooling, so it operates on 1×1 rather than 7×7 feature maps, saving latency with negligible accuracy loss.[file:95][web:96]

#### 5. Two Variants

| Variant           | Target Use Case                                  |
|-------------------|--------------------------------------------------|
| MobileNetV3-Large | Higher-resource mobile (e.g., flagship phones)   |
| MobileNetV3-Small | Lower-resource devices (e.g., wearables, IoT)    |

[file:95][web:96]

#### 6. LR-ASPP Segmentation Decoder

For dense prediction tasks, the authors propose **Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP)**:[file:95][web:96]

- Replaces the full ASPP of DeepLabV3+ with a lightweight alternative.  
- Uses a single global average pooling branch + 1×1 conv instead of multiple atrous rates.[file:95]  
- Achieves about 30% lower latency than MobileNetV2’s R-ASPP at similar Cityscapes accuracy.[file:95][web:96]

---

### Key Results

Representative ImageNet and downstream numbers from the paper:[file:95][web:96][web:99]

| Model               | ImageNet Top-1 | vs V2 (acc)      | vs V2 (latency)      |
|---------------------|---------------:|-----------------:|---------------------:|
| MobileNetV3-Large   | ~75.2%         | +3.2 percentage points | ~20% lower latency |
| MobileNetV3-Small   | ~67.4%         | +6.6 percentage points | similar / slightly lower latency |

- COCO detection: MobileNetV3-Large is reported as over 25% faster than MobileNetV2 at comparable mAP when using channel reduction.[web:96][web:99]  
- Cityscapes segmentation: LR-ASPP is about 30% faster than MobileNetV2’s R-ASPP at similar accuracy.[file:95][web:96]

(Exact percentages vary slightly across configurations; numbers above follow the main ImageNet-1.0× width models.)

---

### Limitations

- NAS and NetAdapt search are computationally expensive (TPU time); practitioners typically use the released MobileNetV3 checkpoints rather than re-running search.[file:95][web:96]  
- The architecture is **hardware-specific**: latency gains are measured on ARM CPUs; benefits may not transfer directly to GPUs or DSPs without re-searching on the target platform.[file:95][web:96]  
- SE blocks improve accuracy but add memory bandwidth overhead, which can partially offset latency savings on some hardware.[file:95][web:96]

---

### What to Take Forward

- **h-swish** is a practical activation choice in deployment-focused CNNs.  
- Combining **NAS (macro structure)** with **NetAdapt (per-layer widths)** is a reusable search strategy.  
- **SE blocks in bottlenecks** provide a cheap and generally reliable accuracy boost.  
- When targeting mobile deployment, optimise for **measured latency**, not just FLOPs.[file:95][web:96]

---

### Lineage

```text
MobileNetV1 (depthwise separable convolutions)
    ↓
MobileNetV2 (inverted residuals + linear bottlenecks)
    ↓
MobileNetV3 (NAS + NetAdapt + h-swish + SE)
    ↓
EfficientNet (compound scaling, similar era but different design philosophy)
```

[file:95][web:96]

---

---

## 2. YOLOv1 (2016)

**Full title:** You Only Look Once: Unified, Real-Time Object Detection  
**Authors:** Redmon, Divvala, Girshick, Farhadi (UW / Allen AI)  
**Venue:** CVPR 2016  
**arXiv:** https://arxiv.org/abs/1506.02640

---

### Motivation

Prior detectors (R-CNN, DPM, and later Fast R-CNN) decomposed detection into separate stages: region proposal → feature extraction → classification → bounding box regression.[file:95][web:106] These stages were trained separately and were relatively slow; R-CNN required tens of seconds per image, and even Fast R-CNN with Selective Search achieved only around 0.5 fps on VOC.[web:106]

**YOLO’s insight:** Frame object detection as a **single regression problem** over the image: one network, one pass, one loss, trained end-to-end.[file:95][web:106] Hence “You Only Look Once.”

---

### Architecture Overview

```text
Input image (448×448×3)
    ↓
24 convolutional layers (alternating 1×1 and 3×3)
    ↓
2 fully connected layers
    ↓
Output tensor: S × S × (B × 5 + C)
```

- `S = 7`: image divided into a 7×7 grid.  
- `B = 2`: each cell predicts 2 bounding boxes.  
- `C = 20`: 20 classes for PASCAL VOC.  
- Output shape: `7 × 7 × 30`.[file:95][web:106]

The backbone is a custom CNN inspired by GoogLeNet. It is first pretrained on ImageNet classification at 224×224, then fine-tuned for detection at 448×448 to improve localisation of small objects.[file:95][web:106]

---

### Core Detection Mechanism

**Grid cell responsibility:** Each grid cell is responsible for detecting objects whose **centre** falls inside that cell.[file:95][web:106]

**Bounding box prediction per cell:**  

```text
(x, y, w, h, confidence)
```

- (x, y): box centre relative to the grid cell, in [0, 1].  
- (w, h): width and height relative to the full image, in [0, 1].  
- confidence = \(P(\text{object}) \times \text{IoU}(\text{pred}, \text{gt})\).[file:95][web:106]

**Class prediction:** Each cell predicts a **single** set of C class probabilities shared by both predicted boxes in that cell, which creates a “one class per cell” limitation.[file:95][web:106]

**At test time:** The final score for class c in a box is:

```text
score_c = P(class_c | object) × confidence = P(class_c) × IoU
```

[file:95][web:106]

---

### Loss Function

YOLOv1 uses a hand-crafted sum of squared errors (SSE) with task-specific weighting.[file:95][web:106] A concise form (using indices i for cells, j for boxes, c for classes):

```text
L = λ_coord Σ_i Σ_j 1_obj_ij [(x_i - x*_i)^2 + (y_i - y*_i)^2]
  + λ_coord Σ_i Σ_j 1_obj_ij [(sqrt(w_i) - sqrt(w*_i))^2
                             + (sqrt(h_i) - sqrt(h*_i))^2]
  + Σ_i Σ_j 1_obj_ij (C_i - C*_i)^2
  + λ_noobj Σ_i Σ_j 1_noobj_ij (C_i - C*_i)^2
  + Σ_i 1_obj_i Σ_c (p_i(c) - p*_i(c))^2

λ_coord = 5,  λ_noobj = 0.5
```

where:[file:95][web:106]

- 1_obj_ij: box j in cell i is responsible for an object.  
- 1_noobj_ij: box j in cell i is not responsible for any object.  
- 1_obj_i: any object is present in cell i.  
- C*_i is IoU when 1_obj_ij = 1, otherwise 0.

**Design choices:**[file:95][web:106]

- λ_coord = 5: upweights localisation loss, since most cells are empty.  
- λ_noobj = 0.5: downweights confidence loss for background boxes so it doesn’t dominate.  
- sqrt(w), sqrt(h): makes localisation loss less sensitive to absolute box scale (small objects are penalised more for the same absolute pixel error).

---

### Why It Works (and Why It Fails)

**Strengths:**[file:95][web:106]

- Fully end-to-end training: the whole network learns detection jointly.  
- Global context: fully connected layers see the entire image, reducing some background false positives.  
- Real-time speed: base YOLO runs at ~45 fps; Fast YOLO runs at ~155 fps on a Titan X (as reported in the paper).[web:106]

**Weaknesses:**[file:95][web:106]

- **One object per cell:** Two objects whose centres fall in the same cell cannot both be correctly classified and localised. This hurts performance on small, crowded scenes.  
- **No anchor priors:** Boxes are predicted directly from the grid, making unusual aspect ratios harder to model.  
- **Coarse grid:** The 7×7 grid limits spatial resolution and localisation accuracy.  
- **Small object performance:** Worse mAP on small objects compared to region-proposal methods at the time.

---

### Results (PASCAL VOC 2007)

Representative results from Redmon et al. and related literature:[file:95][web:100][web:106]

| Model       | mAP   | FPS   |
|-------------|-------|-------|
| DPM v5      | 33.7% | ~0.07 |
| R-CNN       | 66.0% | ≪1    |
| Fast R-CNN  | 70.0% | 0.5   |
| YOLO        | 63.4% | 45    |
| Fast YOLO   | 52.7% | 155   |

YOLO trades roughly 6–7 mAP points vs. Fast R-CNN on VOC 2007 for two orders of magnitude higher throughput (0.5 fps → 45 fps), making it the first widely adopted real-time detector with competitive accuracy.[web:100][web:106]

---

### Historical Impact

YOLOv1 started a lineage that dominates real-time detection:

```text
YOLOv1 (2016) → YOLOv2 / YOLO9000 (anchors, multi-scale)
    → YOLOv3 (Darknet-53, multi-scale heads)
    → YOLOv4 (Bag of Freebies/Specials)
    → YOLOv5 / YOLOv8 (Ultralytics, production-grade)
    → RT-DETR (real-time transformer detectors, 2023+)
```

[file:95][web:103]

Each successor addresses specific YOLOv1 limitations (anchors, FPN, better backbones, focusing on small objects), but the “detection as regression on a grid” idea remains central.

---

### What to Take Forward

- The “detection as regression” paradigm underpins all single-stage detectors.  
- The multi-term weighted loss is a template for designing detection objectives.  
- Understanding YOLOv1’s struggles with small, dense objects motivates anchor-based detectors (YOLOv2+), feature pyramids, and eventually anchor-free and transformer-based detectors.[file:95][web:103]

---

---

## 3. DETR (2020)

**Full title:** End-to-End Object Detection with Transformers  
**Authors:** Carion, Massa, Synnaeve, Usunier, Kirillov, Zagoruyko (Facebook AI)  
**Venue:** ECCV 2020  
**arXiv:** https://arxiv.org/abs/2005.12872

---

### Motivation

Before DETR, major detectors relied on hand-engineered components:[file:95][web:101]

- Anchor boxes encoding prior shapes and scales.  
- Non-Maximum Suppression (NMS) to deduplicate overlapping predictions.  
- Region proposal networks or grid-cell assignment rules.[file:95]

These components require manual design and prevent fully end-to-end set prediction. DETR asks whether transformers can eliminate such heuristics by learning object detection as a **set prediction** problem with global self-attention.[file:95][web:101]

---

### Architecture

```text
Image
  ↓
CNN Backbone (ResNet-50 / ResNet-101) → feature map H/32 × W/32 × C
  ↓
Positional Encoding (2D sine/cosine)
  ↓
Transformer Encoder (L layers, MHSA + FFN)
  ↓
Transformer Decoder (L layers)
  ← Object Queries (N = 100 learned embeddings)
  ↓
FFN per query → (class logits + bounding box)
  ↓
N predictions (most assigned to "no object")
```

[file:95][web:101]

---

### Key Mechanisms

#### 1. Object Queries

DETR uses N=100 learned embeddings, called **object queries**, fed to the transformer decoder as the target sequence.[file:95][web:101]

- Each query learns to specialise in certain positions or object types.  
- Queries are processed in parallel (non-autoregressively).  
- Queries have no explicit spatial prior; their behaviour emerges from training, removing the need for hand-crafted anchors.[file:95][web:101]

#### 2. Bipartite Matching Loss

Given N predictions and M ground-truth objects (M ≤ N):

1. Compute a cost matrix between predictions and ground-truth boxes: classification cost plus box cost (e.g., L1 + GIoU).[file:95][web:101]  
2. Use the **Hungarian algorithm** to find a one-to-one matching between predictions and ground truth (bipartite matching).  
3. Compute loss only on matched pairs; unmatched predictions are trained as “no object.”[file:95][web:101]

This one-to-one supervision removes the need for NMS; duplicate predictions of the same object are penalised via the matching.[file:95][web:101]

**Box loss:** combines L1 on normalised coordinates and Generalised IoU (GIoU) loss, weighted by λ terms.[file:95][web:101]

#### 3. Transformer Encoder

The encoder applies multi-head self-attention over the flattened feature map, allowing each location to attend to all others and aggregating global context; this replaces hand-designed context modules and multi-stage feature aggregation.[file:95][web:101]

#### 4. Transformer Decoder (Cross-Attention)

Each object query attends to encoder outputs via cross-attention, then interacts with other queries via self-attention. This lets the model capture object–object relations and resolve which queries take responsibility for which objects.[file:95][web:101]

---

### Panoptic Segmentation Extension

DETR extends to panoptic segmentation by adding a simple pixel-wise FFN over upsampled decoder attention maps. For each detected object, corresponding attention maps are combined to form a mask, without designing a new mask head.[file:95][web:101]

---

### Results (COCO 2017 val)

Representative results (DETR-R50, R101) as reported in the original and follow-up implementations:[file:95][web:101]

| Model                | AP   | AP_S | AP_M | AP_L | FLOPs (G) | Epochs |
|----------------------|------|------|------|------|-----------|--------|
| Faster R-CNN R50 FPN | ~42  | 26.6 | 45.4 | 53.4 | ~180      | 36     |
| DETR R50             | 42.0 | 20.5 | 45.8 | 61.1 | 86        | 500    |
| DETR R101            | 43.5 | 21.0 | 48.0 | 61.8 | 152       | 500    |

**Interpretation:**[file:95][web:101]

- DETR matches Faster R-CNN in overall AP with a simpler, more generic architecture.  
- DETR is significantly better on large objects (AP_L) due to global self-attention.  
- DETR is significantly worse on small objects (AP_S), a well-known limitation addressed by Deformable DETR and FPN-like extensions.

---

### Limitations

- **Slow convergence:** Original DETR requires ~500 training epochs on COCO to reach its best AP, vs ~36 epochs for the 3× Faster R-CNN schedule.[file:95][web:101][web:107] Matching-based supervision yields relatively weak gradients early on when predictions are random.  
- **Small object performance:** The backbone’s 32× downsampling and lack of explicit multi-scale features hurt AP_S; Deformable DETR and later variants add multi-scale and deformable attention to address this.[file:95][web:101]  
- **Quadratic attention cost:** Standard MHSA is O(n²) in sequence length, making the encoder relatively expensive on high-resolution feature maps.[file:95]  
- **Fixed slots (N=100):** DETR cannot predict more than N objects per image; this is usually sufficient for COCO but is conceptually a capacity limit.[file:95][web:101]

These limitations directly motivated Deformable DETR, Conditional/Anchor-based DETRs, DN-DETR and DINO (detector), which progressively improve convergence and small-object performance.[file:95][web:101]

---

### Why DETR Matters Beyond Its Numbers

DETR showed that:[file:95][web:101]

1. Detection can be framed as set prediction with transformers, eliminating hand-engineered anchors and NMS.  
2. Bipartite matching and Hungarian loss provide a general way to supervise variable-cardinality outputs.  
3. Global self-attention allows modelling object relationships and context beyond local receptive fields.

DETR’s ideas underpin many modern open-vocabulary and grounded detection models (e.g., Grounding DINO, OWL-ViT), which combine language-conditioning with DETR-style set prediction.[file:95]

---

### Lineage

```text
Transformer ("Attention Is All You Need", 2017)
    ↓
DETR (2020) – set prediction, bipartite matching for detection
    ↓
Deformable DETR (2021) – multi-scale deformable attention, faster convergence
    ↓
DAB-DETR / DN-DETR (2022) – anchor-based queries, denoising training
    ↓
DINO-Det (2022) – contrastive denoising, strong COCO benchmarks
    ↓
Grounding DINO / OWL-ViT (2023+) – open-vocabulary, text-conditioned detection
```

[file:95][web:101]

---

### What to Take Forward

- **Bipartite matching (Hungarian algorithm)** is a core idea for supervising sets of predictions and reappears in tracking, pose estimation, and some generative models.  
- **Object queries as learned slots** provide a general mechanism for variable-cardinality outputs without NMS.[file:95][web:101]  
- When DETR underperforms, it is usually on AP_S; multi-scale or deformable attention is the natural fix.

---

---

## 4. Deep Compression (2016)

**Full title:** Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding  
**Authors:** Han, Mao, Dally (Stanford / NVIDIA)  
**Venue:** ICLR 2016 (Best Paper)  
**arXiv:** https://arxiv.org/abs/1510.00149

---

### Motivation

Around 2016, deployment on mobile/embedded hardware was constrained by:[file:95][web:1510.00149]

- **Memory:** Models like AlexNet (~240 MB) and VGG-16 (~552 MB) exceeded on-chip SRAM capacities; off-chip DRAM access is roughly two orders of magnitude more energy-expensive than on-chip SRAM.[file:95]  
- **Compute:** Embedded CPUs/GPUs had limited FP32 throughput.

Goal: compress networks so they fit entirely in on-chip SRAM, enabling faster and more energy-efficient inference with minimal or no accuracy loss.[file:95][web:1510.00149]

---

### The Three-Stage Pipeline

```text
Pretrained Network
        ↓
  [Stage 1: Pruning]
        ↓
  Sparse Network (retrain)
        ↓
  [Stage 2: Trained Quantization + Weight Sharing]
        ↓
  Quantized Sparse Network (retrain centroids)
        ↓
  [Stage 3: Huffman Coding]
        ↓
  Compressed Binary File
```

Each stage enables the next to be more effective: pruning exposes sparsity, quantization reduces bit-width, and Huffman coding exploits non-uniform symbol distributions.[file:95][web:1510.00149]

---

### Stage 1: Pruning

**Mechanism:**[file:95][web:1510.00149]

1. Train the network normally.  
2. Prune connections whose absolute weight is below a threshold τ.  
3. Retrain the sparse network to recover accuracy.  
4. Optionally iterate prune → retrain cycles.

Weights for surviving connections remain full precision; only connectivity changes. Sparse weights are stored in a **Compressed Sparse Row (CSR)**-like format with relative indices to reduce index storage.[file:95][web:1510.00149]

**Compression from pruning alone (paper values):**[file:95][web:1510.00149]

| Network | Original parameters | After pruning | Reduction |
|---------|--------------------:|--------------:|----------:|
| AlexNet | 61M                 | 6.7M          | ~9×       |
| VGG-16  | 138M                | 10.3M         | ~13×      |

Most pruned parameters are in fully connected layers; convolutional layers are pruned more conservatively to preserve accuracy.[file:95][web:1510.00149]

---

### Stage 2: Trained Quantization (Weight Sharing)

After pruning, each remaining weight is still stored as a 32-bit float. Stage 2 reduces precision by **weight sharing** via k-means clustering.[file:95][web:1510.00149]

**Mechanism – k-means over weights:**  

1. For each layer, cluster its weights into k clusters (e.g., k=256 → 8-bit indices).[file:95]  
2. Replace each weight by an index into the centroid table.  
3. Retrain only the centroids: gradients from all weights in a cluster are aggregated and applied to that centroid.[file:95][web:1510.00149]

**Storage layout:**[file:95][web:1510.00149]

```text
Before: each weight = 32 bits
After:  each weight = log2(k) bits (cluster index)
        + k × 32 bits for centroid table (amortised)
```

For k=256: 32 bits → 8 bits (4× reduction per weight).  
For k=32: 32 bits → 5 bits, etc.[file:95][web:1510.00149]

**Centroid initialisation:**[file:95][web:1510.00149]

- Random: poor.  
- Density-based: more centroids near zero.  
- **Linear** (recommended): centroids equally spaced between min and max weight; works best because large-magnitude weights (most influential) get dedicated centroids.

Typical precision in the paper: 8 bits for convolutional layers, 4–5 bits for fully connected layers (more tolerant to quantisation).[file:95][web:1510.00149]

---

### Stage 3: Huffman Coding

After weight sharing, cluster indices follow a skewed distribution (many weights near zero). Huffman coding assigns shorter codes to frequent indices, giving an additional compression factor.[file:95][web:1510.00149]

Sparse CSR indices are also Huffman-coded. This yields an extra ~20–30% reduction on top of pruning + quantization, depending on layer.[file:95][web:1510.00149]

---

### Overall Results

Representative numbers from the paper:[file:95][web:1510.00149]

| Network | Original size | After compression | Ratio | Accuracy loss |
|---------|--------------:|------------------:|------:|--------------:|
| AlexNet | 240 MB        | 6.9 MB            | 35×   | ~0%           |
| VGG-16  | 552 MB        | 11.3 MB           | 49×   | ~0%           |

Both models then fit into 8–16 MB SRAM budgets typical of embedded systems, avoiding DRAM access.[file:95][web:1510.00149]

Reported improvements:[file:95][web:1510.00149]

- Inference speedups of roughly 3–4× on CPU/GPU due largely to better cache utilisation.  
- Energy efficiency improvements of 3–7×, dominated by reduced memory traffic.

---

### Why This Works: Redundancy in Deep Nets

Deep networks, especially fully connected layers, are heavily over-parameterised. Magnitude-based pruning shows many weights contribute negligibly to the output. Remaining weights naturally cluster, enabling aggressive quantisation with minimal accuracy loss. Huffman coding then exploits the non-uniform distribution of cluster indices and sparse indices.[file:95][web:1510.00149]

The staged approach (pruning → quantization → coding) is key:  
- Pruning reduces parameter count and simplifies distributions.  
- Quantisation reduces bits-per-weight.  
- Coding compresses the skewed symbol distribution.[file:95][web:1510.00149]

---

### Limitations

- **Irregular sparsity:** Unstructured connection pruning yields sparse matrices that are hard to accelerate efficiently on standard GPUs (which prefer dense GEMMs). Real speedups require sparse kernels or specialised hardware.[file:95]  
- **Structured pruning alternatives:** Later work explores filter/channel pruning to produce dense but smaller models that accelerate on commodity GPUs, at the cost of somewhat lower compression ratios for the same accuracy.[file:95]  
- **Retraining cost:** Each stage requires fine-tuning, which adds training overhead.  
- **Outdated quantisation method:** Modern **quantisation-aware training (QAT)** and post-training quantisation (PTQ) methods largely supersede k-means weight sharing; frameworks like PyTorch and TensorRT provide QAT primitives natively.[file:95]

---

### Relation to Modern Compression

```text
Deep Compression (2016)
  ├─ Pruning branch → Lottery Ticket Hypothesis (2019), magnitude pruning, L0 regularisation
  ├─ Quantisation branch → QAT (PyTorch/TensorRT), INT8 inference, PTQ (e.g., GPTQ)
  └─ Coding branch → Entropy coding in model packaging (less critical today)

Orthogonal:
  Knowledge Distillation (Hinton, 2015) – teacher–student compression via soft targets
  MobileNet / EfficientNet – architecture-level efficiency (design-time), rather than post-hoc compression
```

[file:95]

---

### What to Take Forward

- **Pruning → Quantisation → Coding** remains a useful conceptual framework for model optimisation.  
- For GPU deployment, **structured pruning** (channels/filters) often yields more practical speedups than unstructured sparsity.  
- **INT8 QAT** is now the standard production technique; weight-sharing via k-means is historically important but less used in modern toolchains.  
- The core insight—that most weights are redundant and can be compressed with little or no accuracy loss—carries directly into LLM compression (INT4, INT8, GGUF, AWQ, etc.).[file:95]

---

---

## Cross-Paper Themes

| Theme                         | MobileNetV3 | YOLOv1 | DETR | Deep Compression |
|-------------------------------|:-----------:|:------:|:----:|:----------------:|
| Efficiency / deployment       | ✓           |        |      | ✓                |
| End-to-end detection training |             | ✓      | ✓    |                  |
| Eliminating hand-engineering  |             |        | ✓    |                  |
| Architecture search / NAS     | ✓           |        |      |                  |
| Quantisation / weight sharing |             |        |      | ✓                |
| Real-time inference           | ✓           | ✓      |      | ✓ (via memory)   |
| Set prediction / global ctx   |             |        | ✓    |                  |

[file:95][web:96][web:100][web:101][web:1510.00149]

**Progression arc to understand:**

```text
YOLOv1 – detection as regression, real-time single-stage
    → DETR – detection as set prediction, transformer-native, no NMS/anchors
    → MobileNetV3 / Deep Compression – efficient backbones and post-hoc compression for deployment
```

These four papers collectively cover: how to detect objects efficiently (YOLO), how to detect them without hand-crafted heuristics (DETR), how to build mobile-efficient backbones (MobileNetV3), and how to compress networks aggressively for deployment (Deep Compression).[file:95][web:96][web:100][web:101][web:1510.00149]

---

*All factual statements above are cross-checked against the cited papers (arXiv / CVF) and derived tables. Key references: [1905.02244](https://arxiv.org/abs/1905.02244), [1506.02640](https://arxiv.org/abs/1506.02640), [2005.12872](https://arxiv.org/abs/2005.12872), [1510.00149](https://arxiv.org/abs/1510.00149).*