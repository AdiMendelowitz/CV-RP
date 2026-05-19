# Object Detection

**Papers covered:**
- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (Ren, He, Girshick, Sun — NeurIPS 2015) — https://arxiv.org/abs/1506.01497
- You Only Look Once: Unified, Real-Time Object Detection (Redmon, Divvala, Girshick, Farhadi — CVPR 2016) — https://arxiv.org/abs/1506.02640

**Reference lecture:** Fei-Fei Li, Justin Johnson, Serena Yeung — CS231n Lecture 11, Stanford, May 2017

---

## The Detection Problem

Classification answers “what is in this image.” Detection answers “what is here, and where exactly?” The output is a set of (class, bounding box) pairs, where a bounding box is commonly represented as (x_center, y_center, width, height).

Two structural challenges define this problem:

**Variable-length output.** A classifier outputs a fixed-length vector. A detector must output a variable number of boxes per image. Using a single fixed-size regression head to emit all boxes is awkward because different images require different numbers of predictions.

**Coupled objectives.** The model must simultaneously classify regions and regress continuous coordinates. This motivates multi-task losses with explicit choices for how to balance classification and localization terms.

---

## The Detection Task Taxonomy

It is useful to keep these four tasks distinct, as they increase in difficulty and output complexity:

| Task                        | Output                          | Key constraint              |
|-----------------------------|---------------------------------|-----------------------------|
| Classification              | single class label              | one label per image         |
| Classification + Localization | class label + one box       | single object assumed       |
| Object Detection            | variable number of (class, box) pairs | multiple objects, unknown count |
| Instance Segmentation       | per-pixel mask per instance     | extends detection with a mask head |

Detection is harder than localization precisely because the output cardinality is unknown at inference time. Semantic segmentation assigns a class to every pixel but does not distinguish instances of the same class (two adjacent cows are both labeled “cow”). Instance segmentation additionally separates individual objects, giving each a distinct mask.

---

## Prerequisite: Bounding Box Parameterization

A box B is stored as (x, y, w, h), where (x, y) is the center and (w, h) are width and height. Regressing raw pixel coordinates directly is poorly conditioned: a box at the top-left corner of an image has very different absolute coordinate values than an identical object at the bottom-right, even though the prediction task is the same.

The standard fix in R-CNN, Fast R-CNN, Faster R-CNN, and later YOLO versions is to predict offsets relative to a reference box called an anchor:

```text
t_x* = (x* - x_a) / w_a        t_y* = (y* - y_a) / h_a
t_w* = log(w* / w_a)           t_h* = log(h* / h_a)
```

where subscript `a` denotes the anchor and `*` denotes the ground-truth box. The model predicts (t_x, t_y, t_w, t_h) and the predicted box is decoded as:

```text
x = t_x * w_a + x_a            y = t_y * h_a + y_a
w = w_a * exp(t_w)             h = h_a * exp(t_h)
```

The log on width and height enforces positivity and makes the regression approximately scale-invariant: t_w = 0 means “keep the anchor’s width.” A 10% size error on a large box and a 10% size error on a small box produce similar loss magnitudes under this parameterization.

---

## Prerequisite: Intersection over Union (IoU)

IoU measures spatial overlap between two boxes A and B:

```text
IoU(A, B) = area(A ∩ B) / area(A ∪ B)
```

By inclusion–exclusion, the union area equals area(A) + area(B) − area(A ∩ B), so:

```text
IoU(A, B) = area(A ∩ B) / (area(A) + area(B) - area(A ∩ B))
```

IoU lies in [0, 1] and is scale-invariant: a small predicted box perfectly covering a small ground-truth object scores the same as a large box covering a large one.

**Worked example:**

```text
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

IoU serves two distinct roles: assigning ground-truth labels to anchors during training, and evaluating detection quality at test time. The standard PASCAL VOC protocol counts a detection as correct if IoU with the ground-truth box is at least 0.5.

---

## Prerequisite: Non-Maximum Suppression (NMS)

A detector produces many overlapping candidate boxes for the same object. NMS collapses them to one per object.

```text
Input:  boxes B with confidence scores S, overlap threshold tau
Output: filtered list of kept boxes

1. Sort B by score descending.
2. While B is not empty:
   a. Take the highest-scoring box b_max. Add it to output.
   b. Remove b_max from B.
   c. Remove all b_i from B where IoU(b_max, b_i) >= tau.
3. Return output.
```

If two boxes overlap heavily, they are likely detecting the same object. Keeping the highest-scoring one and suppressing the rest is a greedy but effective heuristic. The threshold τ is typically 0.5–0.7 depending on the detector.

**Known failure mode.** NMS fails when two objects of the same class overlap heavily, for example two pedestrians partially occluding each other. Both fire on the same region, and the lower-scoring one may be suppressed even though it corresponds to a real, distinct object. Soft-NMS (Bodla et al., 2017) addresses this by decaying scores rather than discarding boxes outright.

---

## Anchor Boxes

An anchor is a fixed reference box centered at a specific spatial location in the original image, with a predefined scale and aspect ratio. The model predicts corrections (deltas) to these anchors rather than boxes from scratch.

All k anchors at a given feature map location share the same center, projected back from the feature map to image coordinates. They differ only in scale and aspect ratio and may extend beyond a single feature-map cell, since objects typically span multiple cells.

```text
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

In Faster R-CNN, the default configuration uses 3 scales (areas 128², 256², 512² pixels) × 3 aspect ratios (1:1, 1:2, 2:1), giving k = 9 anchors per spatial location. For a VGG-16 backbone with stride 16 on an image resized to approximately 600×1000 pixels, the feature map is about 40×60, giving 40 × 60 × 9 ≈ 21,600 anchors total.

---

## The R-CNN Family: Evolution

### R-CNN (Girshick et al., CVPR 2014)

1. Run Selective Search on CPU to generate roughly 2000 region proposals per image.
2. Warp each proposal to a fixed size and run a CNN independently on each one.
3. Classify with a linear SVM; regress box coordinates separately with a different model.

The pipeline is multi-stage and disconnected: the CNN, SVM, and regressor are trained independently. Inference requires tens of seconds per image with VGG-sized networks because the CNN runs once per proposal with no feature reuse across overlapping regions.

### Fast R-CNN (Girshick, ICCV 2015)

The backbone CNN runs once on the full image, producing a shared feature map. Each proposal is projected onto this feature map, and a fixed-size representation is extracted via RoI Pooling. Classification and box regression are trained jointly with a single multi-task loss.

Inference drops to roughly 2 seconds per image with VGG-16 when using Selective Search proposals; the remaining bottleneck is Selective Search itself, which still runs on CPU and takes on the order of 1–2 seconds per image.

### Faster R-CNN (Ren, He, Girshick, Sun — NeurIPS 2015)

Faster R-CNN replaces Selective Search with a learned Region Proposal Network that shares the convolutional backbone with the detector, making proposals much cheaper because features are computed once and reused. Total inference time drops to approximately 200 ms per image with VGG-16 (about 5 fps) on a K40, of which the RPN contributes only around 10 ms.

---

## Faster R-CNN: Architecture

```text
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

The backbone runs once per image. Both the RPN and the detection head read from the same feature map. This shared computation is what makes the proposal step relatively cheap compared to external proposal methods.

---

## Region Proposal Network (RPN)

The RPN is a small fully convolutional network applied to the shared feature map. At each of the H_f × W_f spatial locations, it evaluates k anchors. For each anchor it predicts:

1. An objectness score: probability that the anchor contains any foreground object.
2. Four box regression deltas (t_x, t_y, t_w, t_h): how to refine the anchor into a tighter proposal.

**Implementation:**

```text
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

- Positive (IoU ≥ 0.7 with any ground-truth box).
- Also positive: the single highest-IoU anchor for each ground-truth box, even if its IoU falls below 0.7, ensuring every object has at least one positive anchor.
- Negative (IoU ≤ 0.3 with all ground-truth boxes).
- Ignored (0.3 < IoU < 0.7): excluded from the loss.

Mini-batches are formed by sampling 256 anchors per image with a 1:1 positive–negative ratio when possible.

**RPN loss:**

```text
L = (1/N_cls) · Σ_i  L_cls(p_i, p*_i)
  + (λ/N_reg) · Σ_i  p*_i · L_reg(t_i, t*_i)
```

- `p_i`: predicted objectness probability for anchor i  
- `p*_i`: binary ground-truth label (1 = positive, 0 = negative)  
- `t_i`: predicted 4-vector of regression deltas  
- `t*_i`: target deltas computed from the ground-truth box and the anchor (only defined when p*_i = 1)  
- `L_cls`: log loss over two classes  
- `L_reg`: smooth L1 loss  
- `N_cls` = 256 (mini-batch size), `N_reg` ≈ number of anchor locations (~2400)  
- `λ` = 10, chosen so that the two terms are roughly balanced in magnitude.

The `p*_i` multiplier on the regression term means regression loss is only computed for positive anchors. There is no meaningful regression target for a background anchor.

**Smooth L1 (Huber) loss:**

```text
smooth_L1(x) = 0.5 * x^2       if |x| < 1
             = |x| - 0.5       otherwise
```

For large errors (|x| ≥ 1), the gradient is constant at sign(x), preventing very large gradients from badly misaligned anchors early in training. For small errors it recovers quadratic behavior, giving smooth convergence near the optimum.

---

## RoI Pooling

The detection head’s fully connected layers require fixed-size input. Proposals are variable-size rectangles on the feature map. RoI Pooling resolves this:

```text
Given:
  Feature map of size H_f x W_f x C
  Proposal region R projected onto the feature map (divide pixel coords by backbone stride)
  Target output size H_out x W_out  (7x7 in the paper)

1. Divide R into H_out x W_out bins with integer rounding.
2. Max-pool within each bin independently.

Output: H_out x W_out x C, regardless of R's original shape.
```

The integer rounding introduces a small spatial misalignment: the bin boundary may not align exactly with a feature map grid line. Mask R-CNN (He et al., 2017) replaces RoI Pooling with RoI Align, which uses bilinear interpolation instead of snapping to integer coordinates. For bounding box detection the quantization error is usually negligible; it matters more for instance segmentation.

---

## Training

Four losses are optimized in the full system:

1. RPN objectness classification  
2. RPN box regression  
3. Detection head classification (C classes + background)  
4. Detection head box regression  

The paper proposes a 4-step alternating training procedure as the default: train the RPN, use its proposals to train the detector, fine-tune the RPN with the shared backbone fixed, then fine-tune the detection head with the same fixed backbone. After this procedure both heads share the same backbone weights.

The paper also proposes approximate joint training, where both losses are summed in a single forward–backward pass. In joint training, the gradient through proposal coordinates is ignored: proposals are treated as fixed when backpropagating the detection loss. This approximation works well empirically.

---

## Faster R-CNN Results (PASCAL VOC 2007 test set)

Results below are taken directly from the Faster R-CNN paper. mAP uses the standard VOC metric (IoU threshold 0.5). Timing is on an NVIDIA K40 GPU.

| Method                       | mAP   | fps |
|-----------------------------|-------|-----|
| Fast R-CNN + Selective Search | 70.0% | 0.5 |
| Faster R-CNN (ZFNet)        | 59.9% | 17  |
| Faster R-CNN (VGG-16)       | 73.2% | 5   |

The mAP improvement over Fast R-CNN + Selective Search reflects that RPN proposals are better aligned with the shared features than Selective Search, which is class-agnostic and untrained. The speed improvement comes from replacing a CPU-bound 1–2 s proposal stage with a ~10 ms GPU RPN.

---

## YOLO v1: Single-Stage Detection

**Redmon, Divvala, Girshick, Farhadi — CVPR 2016**

### The Unified Framing

Faster R-CNN is a two-stage detector: stage one generates proposals, stage two classifies them. YOLO reframes detection as a single regression problem: one forward pass, one network, one loss. There is no separate proposal stage.

**Grid-based formulation:**

```text
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

```text
score_c = Pr(object) * IoU * Pr(class_c | object) = Pr(class_c) * IoU
```

This approximates the joint probability that the box contains class c and is accurately placed.

### Loss Function

YOLO v1 uses sum-squared error for all components. Let 1_obj_ij indicate that an object’s center falls in cell i and box j is the “responsible” predictor (highest IoU among the B boxes in that cell), and 1_noobj_ij its complement. Let 1_obj_i indicate that any object center falls in cell i.

```text
L = lambda_coord * Σ_i Σ_j 1_obj_ij * [(x_i - x*_i)^2 + (y_i - y*_i)^2]

  + lambda_coord * Σ_i Σ_j 1_obj_ij * [(sqrt(w_i) - sqrt(w*_i))^2
                                      + (sqrt(h_i) - sqrt(h*_i))^2]

  + Σ_i Σ_j 1_obj_ij  * (C_i - C*_i)^2

  + lambda_noobj * Σ_i Σ_j 1_noobj_ij * (C_i - C*_i)^2

  + Σ_i 1_obj_i * Σ_c (p_i(c) - p*_i(c))^2

where:
  x_i, y_i, w_i, h_i    = predicted box center and dimensions
  x*_i, y*_i, w*_i, h*_i = ground-truth center and dimensions
  C_i                    = predicted confidence score
  C*_i                   = IoU(predicted, ground-truth) when 1_obj_ij = 1, else 0
  p_i(c)                 = predicted class probability for class c in cell i
  p*_i(c)                = 1 for the true class, 0 otherwise
  lambda_coord = 5
  lambda_noobj = 0.5
```

**Why sqrt(w) and sqrt(h)?** Equal absolute errors should matter more for small boxes than large ones. Taking the square root compresses scale so that the loss is relatively larger for small-box errors. The authors report that optimizing this form is more stable than using raw w, h or IoU directly.

**Why λ_coord = 5 and λ_noobj = 0.5?** Most grid cells contain no object. Without reweighting, the confidence loss for background cells would dominate and push the network toward predicting low confidence everywhere. λ_noobj = 0.5 downweights these cells. λ_coord = 5 upweights localization loss so that it is not overwhelmed by confidence and classification terms.

**Why L2 everywhere?** Using squared error for class probabilities frames the entire prediction as regression. Later versions (YOLOv2 and beyond) move toward cross-entropy for classification and more specialized localization losses, but the v1 formulation works empirically.

### Architectural Constraint: One Class Per Cell

Each cell predicts a single set of C class probabilities shared across all B boxes in that cell. If two objects of different classes have centers that fall in the same grid cell, YOLO v1 can only correctly classify one of them. This hard architectural constraint is a major reason YOLO v1 struggles with small, densely packed objects.

### YOLO v1 Results (PASCAL VOC 2007 test set)

Representative results from Redmon et al. (2016), measured on a Titan X GPU:

| Method                  | mAP   | fps  |
|-------------------------|-------|------|
| Faster R-CNN (VGG-16)   | 73.2% | 7    |
| YOLO v1 (full)          | 63.4% | 45   |
| Fast YOLO v1            | 52.7% | 155  |

Note: Faster R-CNN is listed at 7 fps here as reported in Redmon et al. (2016) on a Titan X. The Faster R-CNN paper itself reports ~5 fps on a K40 GPU (see the Faster R-CNN Results section above). YOLO v1 trades about 10 mAP points versus Faster R-CNN for roughly a 6-9x throughput gain depending on hardware.

---

## Cross-Paper Synthesis

The two papers represent the two dominant detection paradigms that persisted for much of the 2015–2020 period.

**Two-stage detectors (Faster R-CNN lineage):** Propose then classify. The RPN is optimised for recall (surface candidate regions); the detection head is optimised for precision (classify and regress accurately). This decoupling allows each stage to specialise; accuracy is generally higher at the cost of increased latency.

**Single-stage detectors (YOLO lineage):** Predict boxes and classes directly from a fixed grid in one forward pass. Eliminating the proposal stage yields lower latency and simpler pipelines, but the grid discretisation in YOLO v1 imposes constraints that hurt small and densely packed objects.

**What successors corrected:** Faster R-CNN’s RoI Pooling introduces quantisation artefacts from integer rounding; Mask R-CNN (He et al., 2017) replaces it with RoI Align. YOLO v1’s coarse 7×7 grid and one-class-per-cell constraint limit small-object recall; YOLOv2 (Redmon and Farhadi, 2017) adopts anchor boxes and per-box class predictions, borrowing from the RPN paradigm while retaining single-stage speed. Both papers use NMS, which struggles on heavily occluded instances of the same class; DETR (Carion et al., 2020) eliminates NMS by framing detection as a set prediction problem with bipartite matching.

**The convergence:** YOLOv2 onward adopted anchor boxes from Faster R-CNN’s RPN while retaining single-stage inference. Anchor-free detectors (e.g., FCOS, CenterNet) and set-prediction approaches (DETR) subsequently pushed the field further, but anchor-based methods and RPN-style logic remain central to much of the 2015–2021 detection literature.

---

## References

- Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. NeurIPS 2015. https://arxiv.org/abs/1506.01497  
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR 2016. https://arxiv.org/abs/1506.02640  
- Girshick, R. (2015). *Fast R-CNN*. ICCV 2015. https://arxiv.org/abs/1504.08083  
- Bodla, N., Singh, B., Chellappa, R., & Davis, L. (2017). *Soft-NMS — Improving Object Detection with One Line of Code*. ICCV 2017.  
- He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). *Mask R-CNN*. ICCV 2017.  
- Carion, N., et al. (2020). *End-to-End Object Detection with Transformers (DETR)*. ECCV 2020.