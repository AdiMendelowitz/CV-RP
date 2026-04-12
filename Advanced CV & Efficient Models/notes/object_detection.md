# Object Detection

**Papers covered:**
- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (Ren, He, Girshick, Sun — NeurIPS 2015) — https://arxiv.org/abs/1506.01497
- You Only Look Once: Unified, Real-Time Object Detection (Redmon, Divvala, Girshick, Farhadi — CVPR 2016) — https://arxiv.org/abs/1506.02640

**Reference lecture:** Fei-Fei Li, Justin Johnson, Serena Yeung — CS231n Lecture 11, Stanford, May 2017

---

## The Detection Problem

Classification answers "what is in this image." Detection answers "what is here, and where exactly?" The output is a set of (class, bounding box) pairs, where a bounding box is conventionally represented as (x_center, y_center, width, height).

Two structural challenges define this problem:

**Variable-length output.** A classifier always outputs a fixed vector. A detector must output a variable number of boxes per image. Pure regression over raw coordinates fails because different images need different numbers of outputs, and a fixed output head cannot accommodate this.

**Coupled objectives.** The model must simultaneously classify regions and regress continuous coordinates. This requires a multi-task loss with non-trivial design choices around balancing the two objectives.

---

## The Detection Task Taxonomy

It is useful to keep these four tasks distinct, as they increase in difficulty and output complexity:

| Task | Output | Key constraint |
|------|--------|----------------|
| Classification | single class label | one label per image |
| Classification + Localization | class label + one box | single object assumed |
| Object Detection | variable number of (class, box) pairs | multiple objects, unknown count |
| Instance Segmentation | per-pixel mask per instance | extends detection with a mask head |

Detection is harder than localization precisely because the output cardinality is unknown at inference time. Semantic segmentation assigns a class to every pixel but does not distinguish instances of the same class (two adjacent cows are both labeled "cow"). Instance segmentation additionally separates individual objects, giving each a distinct mask.

---

## Prerequisite: Bounding Box Parameterization

A box B is stored as (x, y, w, h), where (x, y) is the center and (w, h) are width and height. Regressing raw pixel coordinates directly is poorly conditioned: a box at the top-left corner of an image has very different absolute coordinate values than an identical object at the bottom-right, even though the prediction task is the same. The standard fix is to predict offsets relative to a reference box called an anchor:

```
t_x* = (x* - x_a) / w_a        t_y* = (y* - y_a) / h_a
t_w* = log(w* / w_a)            t_h* = log(h* / h_a)
```

where subscript `a` denotes the anchor and `*` denotes the ground-truth box. The model predicts (t_x, t_y, t_w, t_h) and the predicted box is decoded as:

```
x = t_x * w_a + x_a            y = t_y * h_a + y_a
w = w_a * exp(t_w)              h = h_a * exp(t_h)
```

The log on width and height enforces positivity (exp of any real number is positive) and makes the regression scale-invariant: t_w = 0 means "keep the anchor's width." A 10% size error on a large box and a 10% size error on a small box produce the same loss magnitude. This parameterization is shared by R-CNN, Fast R-CNN, Faster R-CNN, and later YOLO versions.

---

## Prerequisite: Intersection over Union (IoU)

IoU measures spatial overlap between two boxes A and B:

```
IoU(A, B) = area(A ∩ B) / area(A ∪ B)
```

By inclusion-exclusion, the union area equals area(A) + area(B) - area(A ∩ B), so:

```
IoU(A, B) = area(A ∩ B) / (area(A) + area(B) - area(A ∩ B))
```

IoU is bounded in [0, 1] and is scale-invariant: a small predicted box perfectly covering a small ground-truth object scores the same as a large box covering a large one.

**Worked example:**

```
    0   1   2   3   4   5   6   7   8
  +---+---+---+---+---+---+---+---+---+
0 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
1 |   | A | A | A | A |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
2 |   | A | A | A | A |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
3 |   | A | A |A+B|A+B| B | B |   |   |
  +---+---+---+---+---+---+---+---+---+
4 |   | A | A |A+B|A+B| B | B |   |   |
  +---+---+---+---+---+---+---+---+---+
5 |   |   |   | B | B | B | B |   |   |
  +---+---+---+---+---+---+---+---+---+
6 |   |   |   | B | B | B | B |   |   |
  +---+---+---+---+---+---+---+---+---+

Box A: top-left (1,1), bottom-right (4,4)  =>  area(A) = 16
Box B: top-left (3,3), bottom-right (6,6)  =>  area(B) = 16
Intersection: top-left (3,3), bottom-right (4,4)  =>  area(A ∩ B) = 4

IoU = 4 / (16 + 16 - 4) = 4 / 28 ≈ 0.14
```

IoU serves two distinct roles: assigning ground-truth labels to anchors during training, and evaluating detection quality at test time. The standard PASCAL VOC evaluation protocol considers a detection correct if IoU with the ground-truth box is at least 0.5.

---

## Prerequisite: Non-Maximum Suppression (NMS)

A detector produces many overlapping candidate boxes for the same object. NMS collapses them to one per object.

```
Input:  boxes B with confidence scores S, overlap threshold tau
Output: filtered list of kept boxes

1. Sort B by score descending.
2. While B is not empty:
   a. Take the highest-scoring box b_max. Add it to output.
   b. Remove b_max from B.
   c. Remove all b_i from B where IoU(b_max, b_i) >= tau.
3. Return output.
```

If two boxes overlap heavily, they are likely detecting the same object. Keeping the highest-scoring one and suppressing the rest is a greedy but effective heuristic. The threshold tau is typically 0.5 to 0.7 depending on the pipeline.

**Known failure mode.** NMS fails when two objects of the same class overlap heavily, for example two pedestrians partially occluding each other. Both fire on the same region, and the lower-scoring one gets suppressed even though it corresponds to a real, distinct object. Soft-NMS (Bodla et al., 2017) addresses this by decaying scores rather than discarding boxes outright, but the failure case is a known limitation of standard NMS.

---

## Anchor Boxes

An anchor is a fixed reference box centered at a specific spatial location in the original image, with a predefined scale and aspect ratio. The model predicts corrections (deltas) to these anchors rather than boxes from scratch.

All k anchors at a given feature map location share the same center, projected back from the feature map to image coordinates. They differ only in scale and aspect ratio, and freely extend beyond their cell boundary, since objects are almost always larger than a single feature map cell.

```
Three of the nine anchors at one feature map location, all centered on [*]:

   +-------+        +-------+        +--+
   |       |        |       |        |  |
   |  [*]  |        |       |        |[*]
   |       |        |  [*]  |        |  |
   +-------+        |       |        +--+
                    +-------+

  scale 1, 1:1    scale 2, 1:1    scale 1, 1:2
  (square)        (larger square) (tall)
```

In Faster R-CNN: 3 scales (128^2, 256^2, 512^2 pixels) x 3 aspect ratios (1:1, 1:2, 2:1) = 9 anchors per spatial location. For a VGG-16 backbone with stride 16 on a 600x1000 image, the feature map is approximately 40x60, giving 40 * 60 * 9 = 21,600 anchors total.

---

## The R-CNN Family: Evolution

### R-CNN (Girshick et al., CVPR 2014)

1. Run Selective Search on CPU to generate approximately 2000 region proposals per image.
2. Warp each proposal to a fixed size and run a CNN independently on each one.
3. Classify with a linear SVM; regress box coordinates separately with a different model.

The pipeline is multi-stage and disconnected: the CNN, SVM, and regressor are trained independently with different loss functions. Inference requires 47 seconds per image with VGG-16 because the CNN runs once per proposal with no feature reuse across overlapping regions.

### Fast R-CNN (Girshick, ICCV 2015)

The backbone CNN runs once on the full image, producing a shared feature map. Each proposal is projected onto this feature map, and a fixed-size representation is extracted via RoI Pooling. Classification and box regression are trained jointly with a single multi-task loss. Inference drops to roughly 2 seconds per image with VGG-16. The remaining bottleneck is Selective Search, which still runs on CPU and takes 1-2 seconds per image, making it the dominant cost.

### Faster R-CNN (Ren, He, Girshick, Sun — NeurIPS 2015)

Replaces Selective Search with a learned Region Proposal Network that shares the convolutional backbone with the detector. Proposals become nearly free because features are computed once and reused for both proposal generation and detection. Total inference time drops to approximately 198ms per image with VGG-16 (5 fps), of which the RPN contributes only ~10ms.

---

## Faster R-CNN: Architecture

```
Input image
    |
    v
Shared ConvNet backbone (VGG-16 in the paper's primary experiments)
    |
    +---> Feature map (computed once, shared by both heads)
    |          |
    |          v
    |     Region Proposal Network (RPN)
    |          |
    |          v
    |     ~21,600 anchors -> NMS (tau=0.7) -> ~2,000 proposals
    |     (top-300 used at test time)
    |          |
    +----------+
               |
               v
         RoI Pooling (variable proposal size -> fixed 7x7)
               |
               v
         Two FC layers
               |
         +-----+------+
         |            |
         v            v
   Class scores    Box deltas
  (softmax over   (4 values per
    C+1 classes)   class)
```

The backbone runs once per image. Both the RPN and the detection head read from the same feature map. This is what makes the proposal step nearly free relative to Selective Search.

---

## Region Proposal Network (RPN)

The RPN is a small fully-convolutional network applied to the feature map. At each of the H_f x W_f spatial locations, it evaluates k anchors. For each anchor it predicts two things:

1. An objectness score: binary probability that the anchor contains any foreground object.
2. Four box regression deltas (t_x, t_y, t_w, t_h): how to refine the anchor into a tighter proposal.

**Implementation:**

```
Feature map: H_f x W_f x C  (e.g., ~40x60x512 for VGG-16)

At each spatial location:
  3x3 conv -> intermediate feature (1x1xC)
  Two sibling 1x1 convs:
    cls head: 2k scores (object / background per anchor)
    reg head: 4k values (delta per anchor)

Total RPN outputs:
  2 * k * H_f * W_f objectness scores
  4 * k * H_f * W_f regression deltas
```

**Anchor labeling during training:**

- Positive (IoU >= 0.7 with any ground-truth box): the anchor is treated as containing an object.
- Also positive: the single highest-IoU anchor for each ground-truth box, even if its IoU falls below 0.7. This ensures every ground-truth object has at least one positive anchor.
- Negative (IoU < 0.3 with all ground-truth boxes): background.
- Ignored (0.3 <= IoU < 0.7): excluded from the loss.

**RPN loss:**

```
L = (1/N_cls) · Σ_i  L_cls(p_i, p*_i)
  + (λ/N_reg) · Σ_i  p*_i · L_reg(t_i, t*_i)
```

- `p_i`: predicted objectness probability for anchor i
- `p*_i`: binary ground-truth label (1 = positive, 0 = negative)
- `t_i`: predicted 4-vector of regression deltas
- `t*_i`: target deltas computed from the ground-truth box and the anchor (only meaningful when p*_i = 1)
- `L_cls`: binary cross-entropy
- `L_reg`: smooth L1 loss
- `N_cls` = mini-batch size (~256), `N_reg` = total anchor locations (~2400)
- `lambda` = 10, chosen to balance the two terms to roughly equal magnitude

The `p*_i` multiplier on the regression term means regression loss is only computed for positive anchors. There is no meaningful regression target for a background anchor.

**Smooth L1 (Huber) loss:**

```
smooth_L1(x) = 0.5 * x^2       if |x| < 1
             = |x| - 0.5       otherwise
```

For large errors (|x| >= 1), the gradient is constant at sign(x), preventing exploding gradients from badly misaligned anchors early in training. For small errors it recovers quadratic behavior, giving smooth convergence near the optimum.

---

## RoI Pooling

The detection head's fully connected layers require fixed-size input. Proposals are variable-size rectangles on the feature map. RoI Pooling resolves this:

```
Given:
  Feature map of size H_f x W_f x C
  Proposal region R projected onto the feature map (divide pixel coords by backbone stride)
  Target output size H_out x W_out  (7x7 in the paper)

1. Divide R into H_out x W_out bins with integer rounding.
2. Max-pool within each bin independently.

Output: H_out x W_out x C, regardless of R's original shape.
```

The integer rounding introduces a small spatial misalignment: the bin boundary may not align exactly with a feature map grid line. This is a known limitation addressed by RoI Align in Mask R-CNN (He et al., 2017), which uses bilinear interpolation instead of snapping to integer coordinates. For bounding box detection the quantization error is negligible; it matters more for instance segmentation, where pixel-level mask accuracy is required.

---

## Training

Four losses are optimized in the full system:

1. RPN objectness classification
2. RPN box regression
3. Detection head classification (C classes + background)
4. Detection head box regression

The paper proposes a 4-step alternating training procedure as the default: train the RPN from scratch, use its proposals to train the detector, fine-tune the RPN with the fixed shared backbone, then fine-tune the detector head with the same fixed backbone. After this procedure both heads share the same backbone weights.

The paper also proposes approximate joint training, where both losses are summed in a single forward-backward pass. In joint training, the gradient through the proposal coordinates is ignored: proposals are treated as fixed inputs to the detection head. This approximation works well in practice because the ignored gradient is second-order small near convergence.

---

## Faster R-CNN Results (PASCAL VOC 2007 test set)

Results are taken directly from the paper. mAP uses the standard VOC metric (IoU threshold 0.5). Timing is on a K40 GPU.

| Method | mAP | fps |
|--------|-----|-----|
| Fast R-CNN + Selective Search | 70.0% | 0.5 |
| Faster R-CNN (ZFNet) | 59.9% | 17 |
| Faster R-CNN (VGG-16) | 73.2% | 5 |

The mAP improvement over Fast R-CNN + Selective Search reflects that RPN proposals are better calibrated to the shared features than Selective Search, which is class-agnostic and untrained. The speed improvement comes from replacing the 1-2 second CPU-side Selective Search with a ~10ms GPU-resident RPN.

---

## YOLO v1: Single-Stage Detection

**Redmon, Divvala, Girshick, Farhadi — CVPR 2016**

### The Unified Framing

Faster R-CNN is a two-stage detector: stage one generates proposals, stage two classifies them. YOLO reframes detection as a single regression problem: one forward pass, one network, one loss. There is no separate proposal stage.

**Grid-based formulation:**

```
Divide the image into an S x S grid  (S = 7)
Each cell is responsible for objects whose center falls within it.

Each cell predicts:
  B bounding boxes  (B = 2)
  Each box has 5 values:
    x, y:        center coords relative to cell bounds, in [0, 1]
    w, h:        width/height relative to full image, in [0, 1]
    confidence:  Pr(object) * IoU(predicted box, ground-truth box)
  C class probabilities: Pr(class_c | object), one per class  (C = 20 for VOC)

Output tensor: S x S x (B * 5 + C)  =  7 x 7 x 30
```

The confidence score encodes both whether an object exists and how accurately the box is placed. At test time, the final class-specific confidence for class c is:

```
score_c = Pr(object) * IoU * Pr(class_c | object) = Pr(class_c) * IoU
```

This is the joint probability that the box contains class c and is accurately placed.

### Loss Function

YOLO v1 uses sum-squared error for all components. Let 1_obj_ij indicate that an object's center falls in cell i and box j is the "responsible" predictor (highest IoU with the ground truth among the B boxes in that cell), and 1_noobj_ij the complement. Let 1_obj_i indicate that any object center falls in cell i.

```
L = lambda_coord * Σ_i Σ_j 1_obj_ij * [(x_i - x*_i)^2 + (y_i - y*_i)^2]

  + lambda_coord * Σ_i Σ_j 1_obj_ij * [(sqrt(w_i) - sqrt(w*_i))^2
                                      + (sqrt(h_i) - sqrt(h*_i))^2]

  + Σ_i Σ_j 1_obj_ij  * (C_i - C*_i)^2

  + lambda_noobj * Σ_i Σ_j 1_noobj_ij * (C_i - C*_i)^2

  + Σ_i 1_obj_i * Σ_c (p_i(c) - p*_i(c))^2

where:
  x_i, y_i, w_i, h_i    = predicted box center and dimensions for box j in cell i
  x*_i, y*_i, w*_i, h*_i = ground-truth box center and dimensions
  C_i                    = predicted confidence score
  C*_i                   = IoU(predicted, ground-truth) when 1_obj = 1, else 0
  p_i(c)                 = predicted class probability for class c in cell i
  p*_i(c)                = ground-truth class label (1 for true class, 0 otherwise)
  lambda_coord = 5
  lambda_noobj = 0.5
```

**Why sqrt(w) and sqrt(h)?** Equal absolute errors should matter more for small boxes than large ones. A 5-pixel error on a 10-pixel-wide object is catastrophic; the same error on a 200-pixel-wide object is negligible. Taking the square root compresses the scale so that the loss gradient is relatively larger for small-box errors. The authors report trying to optimize IoU directly and finding the sqrt formulation easier to train.

**Why lambda_coord = 5 and lambda_noobj = 0.5?** The vast majority of grid cells contain no object, so without reweighting, the confidence loss for background cells would dominate and push the network toward predicting low confidence everywhere. lambda_noobj = 0.5 down-weights these cells. lambda_coord = 5 compensates by up-weighting localization loss so it is not drowned out.

**Why L2 everywhere?** Using squared error for class probabilities frames the entire prediction as regression. Cross-entropy is better calibrated for probability outputs, and YOLO v2 and later adopt it, but the v1 formulation works empirically.

### Architectural Constraint: One Class Per Cell

Each cell predicts a single set of C class probabilities, shared across all B boxes in that cell. If two objects of different classes have centers that fall in the same grid cell, YOLO v1 can only correctly classify one of them. This is a hard architectural constraint, not a training issue, and is the primary reason YOLO v1 struggles with small, densely packed objects.

### YOLO v1 Results (PASCAL VOC 2007 test set)

| Method | mAP | fps |
|--------|-----|-----|
| Faster R-CNN (VGG-16) | 73.2% | 5 |
| YOLO v1 (full) | 63.4% | 45 |
| YOLO v1 (Fast YOLO) | 52.7% | 155 |

YOLO v1 trades roughly 10 mAP points for a 9x speed improvement over Faster R-CNN. The speed gain comes from eliminating the proposal stage entirely; the accuracy gap reflects the coarser spatial resolution imposed by the 7x7 grid and the one-class-per-cell constraint.

---

## Cross-Paper Synthesis

The two papers represent the two dominant detection paradigms that persist through the present day.

**Two-stage detectors (Faster R-CNN lineage):** Propose then classify. The RPN is optimized for recall (surface all candidate regions); the detection head is optimized for precision (classify and regress accurately). This decoupling allows each stage to specialize. Accuracy is generally higher; latency is also higher.

**Single-stage detectors (YOLO lineage):** Predict boxes and classes directly from a fixed grid in one forward pass. No proposal stage means lower latency and simpler pipelines, but the grid discretization imposes a localization constraint that hurts small and densely-packed objects.

**What successors corrected:**
Faster R-CNN's RoI Pooling introduces quantization artifacts from integer rounding; Mask R-CNN (He et al., 2017) replaces it with RoI Align. YOLO v1's grid discretization and one-class-per-cell constraint limit small-object recall; YOLO v2 (Redmon and Farhadi, CVPR 2017) adopts anchor boxes and per-box class predictions, borrowing directly from the RPN paradigm while retaining single-stage speed. Both papers rely on NMS, which fails on heavily occluded objects of the same class; DETR (Carion et al., ECCV 2020) eliminates NMS by framing detection as a set prediction problem with bipartite matching.

**The convergence:** YOLO v2 onward adopted anchor boxes from Faster R-CNN's RPN while retaining the one-stage architecture. Anchor-free detectors (FCOS, CenterNet) and set-prediction approaches (DETR) subsequently pushed the field further, but anchor-based methods remain essential context for the majority of the detection literature from 2015 through 2021.