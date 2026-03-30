# Object Detection: Faster R-CNN and YOLO v1

**Papers covered:**
- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (Ren et al., NeurIPS 2015) — https://arxiv.org/abs/1506.01497
- You Only Look Once: Unified, Real-Time Object Detection (Redmon et al., CVPR 2016) — https://arxiv.org/abs/1506.02640

**Reference lecture:** Karpathy, CS231n 2017 Lecture 11 — Detection

---

## The Detection Problem

Classification answers "what is in this image." Detection answers "what is here, and where exactly?" The output is a set of (class, bounding box) pairs, where a bounding box is conventionally represented as (x_center, y_center, width, height) or (x_min, y_min, x_max, y_max) depending on the framework.

Two structural challenges define this problem:

1. **Variable-length output.** A classifier always outputs a fixed vector. A detector must output a variable number of boxes per image, which ordinary feed-forward architectures cannot do directly.

2. **Localization precision.** The model must regress continuous coordinates, not just classify. This couples a regression objective to a classification objective, which creates non-trivial loss design choices.

---

## Prerequisite Concepts

### Bounding Box Representation

A box B is parameterized as:

```
B = (x, y, w, h)

where:
  x, y = center coordinates (pixels or fraction of image width/height)
  w, h = width and height of the box
```

The "delta" parameterization used by Faster R-CNN expresses predicted boxes as *offsets from anchor boxes*, not as absolute coordinates. This is important and addressed in detail below.

### Intersection over Union (IoU)

IoU measures the spatial overlap between two boxes A and B. It is the ratio of the area of their intersection to the area of their union.

```
IoU(A, B) = area(A ∩ B) / area(A ∪ B)
```

By inclusion-exclusion:

```
area(A ∪ B) = area(A) + area(B) - area(A ∩ B)
```

So equivalently:

```
IoU(A, B) = area(A ∩ B) / (area(A) + area(B) - area(A ∩ B))
```

IoU is bounded: IoU in [0, 1]. IoU = 0 means no overlap. IoU = 1 means perfect overlap.

**Worked example (ASCII diagram):**

```
Image coordinate space (each cell = 1 pixel):

    0   1   2   3   4   5   6   7   8
  +---+---+---+---+---+---+---+---+---+
0 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
1 |   | A | A | A | A |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
2 |   | A | A | A | A |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
3 |   | A | A |A+B|A+B| B | B |   |   |   <- intersection region
  +---+---+---+---+---+---+---+---+---+
4 |   | A | A |A+B|A+B| B | B |   |   |   <- intersection region
  +---+---+---+---+---+---+---+---+---+
5 |   |   |   | B | B | B | B |   |   |
  +---+---+---+---+---+---+---+---+---+
6 |   |   |   | B | B | B | B |   |   |
  +---+---+---+---+---+---+---+---+---+

Box A: top-left (1,1), bottom-right (4,4)  => width=4, height=4 => area(A) = 16
Box B: top-left (3,3), bottom-right (6,6)  => width=4, height=4 => area(B) = 16

Intersection: top-left (3,3), bottom-right (4,4) => width=2, height=2 => area(A ∩ B) = 4

IoU = 4 / (16 + 16 - 4) = 4 / 28 = 0.143
```

**Thresholds in practice:**
- IoU >= 0.5: positive match (a.k.a. "correct detection" by PASCAL VOC convention)
- IoU < 0.4: negative (background)
- 0.4 <= IoU < 0.5: ignored during training in some pipelines

---

## Anchor Boxes

An anchor is a fixed reference box, centered at a specific spatial location, with a predefined scale and aspect ratio. The model does not predict boxes from scratch; it predicts *corrections* (deltas) to these anchors.

**Why anchors?** Without anchors, the model would need to regress raw pixel coordinates (x, y, w, h) from scratch. This is a high-variance regression problem because a box at the top-left of an image has very different raw coordinates than an identical object at the bottom-right. Anchors reduce this to predicting small deltas from a sensible prior, which is a much lower-variance regression target.

**Anchor box diagram (ASCII):**

All K anchors at a given feature map location share the same center [*], which is the center of the feature map cell projected back to image space. The anchors differ only in scale and aspect ratio, and they freely extend beyond the cell boundary — objects are almost always larger than a single feature map cell.

```
Three of the nine anchors at one feature map cell, all centered on [*]:

   +-------+        +-------+        +--+
   |       |        |       |        |  |
   |  [*]  |        |       |        |[*]
   |       |        |  [*]  |        |  |
   +-------+        |       |        +--+
                    +-------+

  scale 1, 1:1    scale 2, 1:1    scale 1, 1:2
  (square)        (larger square) (tall)

[*] is at the same image coordinate in all three boxes.
In Faster R-CNN: 3 scales (128^2, 256^2, 512^2) x 3 aspect ratios (1:1, 1:2, 2:1) = 9 anchors per cell.
For a 600x1000 image with ~40x60 feature map: 40 * 60 * 9 = 21,600 anchors total (~20,000).
```

**Delta parameterization.** Given anchor box (x_a, y_a, w_a, h_a) and ground-truth box (x*, y*, w*, h*), the regression targets are:

```
t_x* = (x* - x_a) / w_a
t_y* = (y* - y_a) / h_a
t_w* = log(w* / w_a)
t_h* = log(h* / h_a)
```

The model predicts (t_x, t_y, t_w, t_h), and the predicted box is decoded as:

```
x = t_x * w_a + x_a
y = t_y * h_a + y_a
w = w_a * exp(t_w)
h = h_a * exp(t_h)
```

The log on the width/height ensures the predicted dimensions are always positive (exp maps any real number to R+), and it makes the regression scale-invariant: a 10% error on a large box and a 10% error on a small box are treated equivalently.

---

## Non-Maximum Suppression (NMS)

A detector produces many overlapping candidate boxes for the same object. NMS is the post-processing step that collapses them to one.

**Algorithm:**

```
Input:  list of boxes B with confidence scores S, threshold tau_nms
Output: filtered list of kept boxes

1. Sort B by score descending.
2. While B is not empty:
   a. Take the highest-scoring box b_max. Add it to output.
   b. Remove b_max from B.
   c. Remove from B all boxes b_i where IoU(b_max, b_i) >= tau_nms.
3. Return output.
```

**Why this works.** If two boxes overlap heavily (high IoU), they are likely detecting the same object. Keeping the highest-scoring one and suppressing the rest is a greedy but effective heuristic. The threshold tau_nms is typically 0.5 or 0.7; lower values are more aggressive (discard more boxes).

**Known failure mode.** NMS fails when two objects of the same class overlap heavily (e.g., two pedestrians partially occluding each other). Soft-NMS (Bodla et al., 2017) addresses this by decaying scores rather than discarding, but the failure case is a known limitation of the original algorithm.

---

## Faster R-CNN

**Citation:** Ren, He, Girshick, Sun. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." NeurIPS 2015.

### Prior Paradigm and Its Failure Mode

R-CNN (Girshick et al., 2014) ran a region proposal algorithm (Selective Search) on the CPU to generate ~2000 candidate regions, then classified each region independently through a CNN. This produced ~50 seconds per image at test time. Fast R-CNN (Girshick, 2015) fixed the per-region CNN cost by sharing the convolutional backbone and introducing RoI Pooling, reducing test time to ~2 seconds. But Selective Search still took ~2 seconds on CPU, making it the new bottleneck. Faster R-CNN's contribution is replacing Selective Search with a convolutional Region Proposal Network (RPN) that shares the backbone's feature computation with the detection head, making the entire pipeline end-to-end trainable in a single unified network.

### Architecture

```
Input image
    |
    v
Shared ConvNet backbone (ZFNet or VGG-16 in the paper)
    |
    +---> Feature map (shared)
    |          |
    |          v
    |     Region Proposal Network (RPN)
    |          |
    |          v
    |     ~20,000 anchors -> NMS -> ~2,000 proposals
    |     (top-300 used at test time)
    |          |
    +----------+
               |
               v
         RoI Pooling
               |
               v
         Two FC layers
               |
         +-----+------+
         |            |
         v            v
   Class scores    Box deltas
  (softmax over   (4 values per
    C+1 classes)   class, or class-
                   agnostic)
```

The key insight: the backbone runs once per image. Both the RPN and the final detector head read from the same feature map. This is what makes the approach fast relative to Fast R-CNN with Selective Search.

### Region Proposal Network (RPN) -- Section 3 of the paper

The RPN is a small sliding-window network applied to the feature map. At each spatial location, it evaluates K anchors. For each anchor it predicts:

1. An objectness score: probability that the anchor contains any object (binary classification: object vs. background).
2. A bounding box regression: four delta values (t_x, t_y, t_w, t_h) to refine the anchor into a tighter region proposal.

**RPN in detail:**

```
Feature map: H_f x W_f x C (e.g., 40 x 60 x 512 for VGG-16)

For each spatial location:
  - Apply 3x3 conv with C output channels -> intermediate feature (1 x 1 x C)
  - Two sibling 1x1 convs:
      cls head: 1x1 conv -> 2K scores (object / background for each of K anchors)
      reg head: 1x1 conv -> 4K values (delta for each of K anchors)

Total RPN outputs over full feature map:
  - 2 * K * H_f * W_f objectness scores
  - 4 * K * H_f * W_f regression deltas
```

**Training the RPN -- anchor labeling:**

Each anchor is assigned a binary label:
- Positive (object): IoU with any ground-truth box >= 0.7, OR the anchor with the highest IoU with a given ground-truth box (even if below 0.7, to ensure every GT has at least one positive anchor).
- Negative (background): IoU with all ground-truth boxes < 0.3.
- Ignored: 0.3 <= IoU < 0.7.

**RPN loss:**

```
L_RPN = (1 / N_cls) * sum_i L_cls(p_i, p_i*)
      + lambda * (1 / N_reg) * sum_i p_i* * L_reg(t_i, t_i*)

where:
  p_i    = predicted objectness probability for anchor i
  p_i*   = ground-truth label (1 = positive, 0 = negative)
  t_i    = predicted 4 delta values
  t_i*   = target delta values (only meaningful when p_i* = 1)
  L_cls  = binary cross-entropy
  L_reg  = smooth L1 loss (Huber loss)
  N_cls  = number of anchors in mini-batch (~256)
  N_reg  = number of anchor locations (~2400)
  lambda = balancing weight (set to 10 in the paper)
```

The regression loss is multiplied by p_i* so that it is only applied to positive anchors. There is no point regressing a box for a background anchor.

**Smooth L1 (Huber) loss:**

```
smooth_L1(x) = 0.5 * x^2       if |x| < 1
             = |x| - 0.5       otherwise
```

This is less sensitive to outliers than L2 loss. For large delta errors (|x| >= 1), the gradient is constant (sign(x)), preventing exploding gradients from badly misaligned anchors.

### RoI Pooling

The RPN generates proposals from ~20,000 anchors. After NMS (IoU threshold 0.7), ~2,000 candidates remain. During training, all ~2,000 are passed to the detection head. At test time, only the top-300 by objectness score are used. Each proposal maps to a variable-size region on the feature map, and the detection head expects fixed-size input. RoI Pooling solves this.

**How it works:**

```
Given:
  - Feature map of size H_f x W_f
  - A proposal region R mapped to the feature map (by scaling by the stride)
  - Target output size H_out x W_out (e.g., 7 x 7)

Divide R into H_out x W_out sub-windows (bins).
In each bin, apply max pooling -> one value.
Result: H_out x W_out x C fixed-size feature, regardless of R's size.
```

The division into bins is done with integer rounding, which introduces a quantization error (the box boundary may not fall exactly on a feature map grid line). This is a known limitation addressed by RoI Align (He et al., Mask R-CNN, 2017), which uses bilinear interpolation instead of integer rounding.

### Key Results (from the paper)

| Method | mAP (PASCAL VOC 2007) | Frames/sec |
|--------|----------------------|------------|
| Fast R-CNN + Selective Search | 70.0 | 0.5 |
| Faster R-CNN (VGG-16) | 73.2 | 5 |
| Faster R-CNN (ZFNet) | 59.9 | 17 |

The paper's central claim: by sharing features between the RPN and the detector, you eliminate the proposal bottleneck without accuracy regression.

---

## YOLO v1

**Citation:** Redmon, Divvala, Girshick, Farhadi. "You Only Look Once: Unified, Real-Time Object Detection." CVPR 2016.

### The Unified Framing

Faster R-CNN is a two-stage detector: stage one generates proposals, stage two classifies them. YOLO reframes detection as a *single regression problem*: one forward pass, one network, one loss. There is no separate proposal stage.

**Grid-based formulation:**

```
Divide the image into an S x S grid (S=7 in the paper).
Each cell is responsible for detecting objects whose center falls in that cell.

Each cell predicts:
  - B bounding boxes (B=2 in the paper), each with:
      x, y:      center coordinates relative to cell bounds, in [0, 1]
      w, h:      width and height relative to full image, in [0, 1]
      confidence: Pr(object) * IoU(predicted, ground-truth)
  - C class probabilities: Pr(class_c | object), one per class

Total output tensor: S x S x (B * 5 + C)
For VOC (C=20): 7 x 7 x (2*5 + 20) = 7 x 7 x 30
```

The "confidence" score is defined as Pr(object) * IoU, so at test time the final class-specific confidence for class c is:

```
score_c = Pr(object) * IoU * Pr(class_c | object) = Pr(class_c) * IoU
```

This is interpretable as: how confident am I that this box contains class c and the box is accurately placed?

### Loss Function

```
L = lambda_coord * sum over cells * sum over boxes
      [ 1_obj * ((x_i - x_i*)^2 + (y_i - y_i*)^2) ]
  + lambda_coord * sum over cells * sum over boxes
      [ 1_obj * ((sqrt(w_i) - sqrt(w_i*))^2 + (sqrt(h_i) - sqrt(h_i*))^2) ]
  + sum over cells * sum over boxes
      [ 1_obj * (C_i - C_i*)^2 ]
  + lambda_noobj * sum over cells * sum over boxes
      [ 1_noobj * (C_i - C_i*)^2 ]
  + sum over cells
      [ 1_obj_cell * sum_c (p_i(c) - p_i*(c))^2 ]

where:
  1_obj      = 1 if object's center falls in this cell AND this box is "responsible"
  1_noobj    = 1 if this box is NOT responsible for any object
  1_obj_cell = 1 if any object center falls in this cell
  lambda_coord = 5 (up-weights localization loss)
  lambda_noobj = 0.5 (down-weights background confidence loss)
  C_i*       = IoU(predicted box, ground-truth) when 1_obj = 1, else 0
```

**Why sqrt(w) and sqrt(h)?** Equal absolute errors in width should matter more for small boxes than large boxes. A 5-pixel error on a 10-pixel-wide box is catastrophic; the same error on a 200-pixel-wide box is negligible. Taking sqrt compresses the scale, so the loss gradient is relatively larger for small box errors.

**Why L2 everywhere?** YOLO v1 uses L2 (squared error) for all components including classification, framing the entire output as a regression. This is a simplification that works empirically but is not principled (cross-entropy would be better for probabilities). YOLO v2 and later address this.

### Constraint: One Class Per Cell

Each cell predicts one set of class probabilities, shared across all B boxes in that cell. If two objects of different classes have overlapping centers mapped to the same cell, YOLO v1 can only correctly classify one of them. This is a hard architectural constraint, not a training issue. It is the primary reason YOLO v1 struggles with small clustered objects.

### Key Results

| Method | mAP (PASCAL VOC 2007) | Frames/sec |
|--------|----------------------|------------|
| Faster R-CNN (VGG-16) | 73.2 | 7 |
| YOLO v1 (full) | 63.4 | 45 |
| YOLO v1 (fast) | 52.7 | 155 |

YOLO v1 trades ~10 mAP points for a 6-7x speed improvement over Faster R-CNN. The speed gain is real; the accuracy gap is also real.

---

## Cross-Paper Synthesis

The two papers represent the two dominant paradigms in detection that persist to this day:

**Two-stage (Faster R-CNN lineage):** Propose then classify. This decoupling lets each stage specialize: the RPN is optimized purely for recall (find all possible objects), and the detection head is optimized for precision (classify and regress precisely). Accuracy is generally higher, but the two-stage structure has higher latency.

**One-stage (YOLO lineage):** Predict directly from a grid. No separate proposal stage. Faster, simpler, but the spatial discretization (grid cells) introduces a localization constraint that hurts on small or densely-packed objects.

**What each paper got right:**
- Faster R-CNN introduced the RPN as a trained, GPU-based replacement for hand-crafted proposal algorithms. This is conceptually clean and remains influential.
- YOLO framed detection as regression, which massively simplifies the pipeline and enables real-time speeds. The "responsible cell" concept is an elegant constraint that makes the output format tractable.

**What each paper got wrong (or left for successors):**
- Faster R-CNN's RoI Pooling has quantization artifacts; fixed by RoI Align in Mask R-CNN (2017).
- YOLO v1's grid discretization (one class per cell, B=2 boxes with no anchor priors) limits small-object recall; fixed by YOLO v2/v3 adopting anchor boxes.
- Both papers use NMS as a post-processing step, which fails on heavily occluded objects. This remains an open problem.

**The convergence:** YOLO v2 onward adopted anchor boxes, borrowed directly from the RPN paradigm, while retaining the one-stage speed. Anchor-free detectors (FCOS, CenterNet) and set-prediction approaches (DETR) subsequently pushed the field further, but understanding anchor-based methods remains essential for reading the majority of the detection literature from 2015 through 2021.

