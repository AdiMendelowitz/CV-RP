# DETR and the Hungarian Matching Algorithm

**Paper:** "End-to-End Object Detection with Transformers"  
**Authors:** Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko  
**Venue:** ECCV 2020  
**arXiv:** https://arxiv.org/abs/2005.12872

---

## Background: Why Standard Detectors Produce Duplicates

Every anchor-based or grid-based detector (Faster R-CNN, YOLO, SSD) partitions the image into a grid and assigns one or more anchor boxes to each cell. Multiple cells or anchors routinely fire on the same object, producing a cluster of overlapping high-confidence predictions for a single ground-truth instance. This is not a model failure; it is a structural consequence of how these detectors frame prediction as a dense, per-location classification problem.

The standard remedy is non-maximum suppression (NMS): sort predictions by confidence, keep the highest-scoring one, and suppress all predictions whose IoU with the kept box exceeds a threshold. NMS is effective in practice but introduces two problems. First, it is a post-processing heuristic with its own hyperparameters (the IoU threshold) that must be tuned per dataset and task. Second, its greedy suppression fails in dense scenes where two legitimate objects are close enough that one is incorrectly suppressed.

DETR eliminates both problems by changing the prediction structure entirely: it predicts exactly N candidate boxes in one forward pass and forces a one-to-one assignment between predictions and ground-truth objects during training. No redundancy is ever generated, so no suppression is needed.

---

## The Matching Cost Matrix

### Setup

Let N be the fixed number of object queries (predictions) output by DETR's decoder, and let M be the number of ground-truth objects in a given image. DETR always predicts exactly N boxes regardless of how many objects are present; images with fewer than N objects treat the unmatched predictions as "no object."

The cost matrix C has dimensions N x M. Entry C[i, j] encodes how expensive it is to assign prediction i to ground-truth object j. A low cost means prediction i is a plausible match for ground-truth j; a high cost means they are incompatible.

### Cost composition

Each entry C[i, j] is a weighted sum of three terms:

```
C[i, j] = -lambda_cls * p_i(c_j)
         + lambda_L1  * ||b_i - b_j||_1
         + lambda_GIoU * GIoU_loss(b_i, b_j)
```

where:
- `p_i(c_j)` is the predicted probability of prediction i having the class label of ground-truth j
- `b_i` is the predicted box (normalised centre-x, centre-y, width, height)
- `b_j` is the ground-truth box in the same format
- `lambda_cls`, `lambda_L1`, `lambda_GIoU` are scalar weights

The classification term uses a negative probability (not a log probability) so that Hungarian matching does not need to evaluate a full softmax before the matching step is known. The L1 and GIoU terms measure geometric similarity between the predicted and target boxes.

### The Hungarian algorithm

Given C, the Hungarian algorithm finds the permutation sigma of size M (a one-to-one assignment of M ground-truth objects to M of the N predictions) that minimises the total cost:

```
sigma* = argmin_{sigma in S_M}  sum_{j=1}^{M} C[sigma(j), j]
```

The remaining N - M predictions are assigned to a special "no object" class. The algorithm runs in O((N + M)^3) time in the worst case; since N is fixed at 100 in the published model and M is typically small (fewer than 100 objects per image), this is computationally negligible relative to the forward pass.

---

## Why the Set Prediction Loss Requires a Fixed Matching

Object detection is intrinsically a set prediction problem: a ground-truth annotation is a set of (class, box) pairs with no canonical ordering. A naive training objective might try to compute loss over all possible assignments between predictions and ground-truth objects and sum or average them, but this approach is ill-defined for gradient-based optimisation for two reasons.

First, the optimal assignment changes during training as weights update. If the loss averages over all N! permutations (or even the M! permutations of the matched subset), gradients from contradictory assignments cancel partially, producing a noisy and unstable signal. A prediction that is spatially close to object A but far from object B receives a gradient that simultaneously pulls it toward A and toward B, with the net direction depending on which force happens to dominate at that weight configuration.

Second, averaging over all permutations is computationally intractable for realistic N (100 queries yields 100! terms) and conceptually wrong: the model should not be rewarded for being a mediocre match to every object simultaneously. It should be penalised for failing to match each object precisely.

The Hungarian matching resolves both problems by finding, before any loss is computed, the single lowest-cost bijection between predictions and ground-truth objects. Once sigma* is fixed for the current image, the loss is computed over that assignment alone:

```
L_DETR = sum_{j=1}^{M} [ -log p_{sigma*(j)}(c_j)
                         + 1[c_j != no_obj] * L_box(b_{sigma*(j)}, b_j) ]
         + sum_{i not in sigma*} [-log p_i(no_obj)]
```

This loss is a standard sum of per-object cross-entropy and box regression terms, which is differentiable and well-conditioned. The matching step itself is not differentiable, but it does not need to be: sigma* is treated as a fixed index into the predictions, and gradients flow only through the loss terms evaluated at those indices.

The key insight is that matching and loss computation are separated in time. Matching finds the best assignment given current weights; the loss then optimises those weights to make that assignment better. This alternation is stable because the Hungarian solution changes smoothly as weights change.

---

## Why DETR Does Not Need NMS

The root cause of duplicate predictions in standard detectors is the many-to-one structure of their assignment: multiple anchors compete to predict the same object, and all of them may win independently. NMS is the post-hoc fix for this structural redundancy.

DETR's one-to-one matching during training eliminates the structural redundancy at its source. Each ground-truth object is assigned to exactly one prediction by the Hungarian algorithm, and only that prediction receives a positive training signal for that object. Every other prediction is trained toward "no object" for that ground-truth instance. After sufficient training, the model learns to produce at most one confident prediction per object, because producing two confident predictions for the same object would mean one of them was trained with a "no object" label and is therefore inconsistent with what the model learned to predict.

There is a subtlety here: this argument holds at the level of the training set distribution. At inference on a novel image, the model might in principle produce two confident predictions for one object if the image is far from the training distribution. In practice, the one-to-one matching constraint is strong enough that this is rare, and DETR runs inference without any suppression step.

YOLO's grid-based design does not have this property. Each grid cell independently predicts boxes for whatever objects fall within it, and neighbouring cells covering the same large object all generate predictions. There is no mechanism in the training objective that discourages two adjacent cells from both predicting the same object confidently, because they are never penalised against each other during training.

---

## GIoU as a Box Regression Loss

### The problem with IoU as a loss

Intersection over Union (IoU) measures the overlap between two boxes as:

```
IoU(A, B) = |A ∩ B| / |A ∪ B|
```

IoU is geometrically meaningful and invariant to box scale, making it the standard evaluation metric. However, it has two properties that make it a poor training loss.

The first property is the non-overlapping case: when two boxes do not overlap, their intersection is zero and IoU is exactly zero regardless of how far apart they are. The gradient of IoU with respect to the predicted box coordinates is also zero in this case, so no learning signal is produced for non-overlapping predictions. Early in training, when predictions are initialised randomly, many predictions will not overlap their targets, and IoU loss provides no gradient to move them toward the target.

The second property is the scale-invariance trap: because IoU normalises by union area, a small error in a small box produces the same IoU penalty as a proportionally equivalent error in a large box. This is often desirable for evaluation, but during training it means the loss surface is flat in directions that move the box proportionally rather than absolutely, which can slow convergence for small objects.

### GIoU

Generalized IoU (GIoU) was introduced by Rezatofighi et al. (CVPR 2019) to fix the zero-gradient problem. It augments IoU with a penalty based on the smallest enclosing box of the two boxes:

```
GIoU(A, B) = IoU(A, B) - (|C \ (A ∪ B)|) / |C|
```

where C is the smallest axis-aligned box that contains both A and B, and `|C \ (A ∪ B)|` is the area of C not covered by the union of A and B.

The correction term is always non-negative (it is zero only when A and B have the same bounding box) and is strictly positive whenever A and B do not overlap. When the two boxes are far apart, C is large, the union is small, and the correction term approaches 1, so GIoU approaches -1. As the boxes converge, the correction term shrinks to zero and GIoU approaches IoU. The range of GIoU is [-1, 1], compared to [0, 1] for IoU.

The GIoU loss used in DETR is:

```
L_GIoU(b_i, b_j) = 1 - GIoU(b_i, b_j)
```

This is zero when the boxes perfectly overlap and increases as they diverge, providing a non-zero gradient in all configurations including the non-overlapping case.

### Why DETR combines L1 and GIoU

DETR uses both L1 loss and GIoU loss on the box coordinates, which addresses a complementary weakness. GIoU is scale-invariant in the same way IoU is, so it does not distinguish between a large absolute error on a small box and a small absolute error on a large box. L1 loss on the normalised coordinates provides absolute positional feedback, penalising large displacements in image-space coordinates regardless of box size. The two losses together provide both geometric overlap feedback (GIoU) and absolute displacement feedback (L1), which empirically produces faster and more stable convergence than either alone. This combination is also used in the Hungarian cost matrix, meaning the matching step and the loss step are aligned in what they penalise.

---

## Summary of Key Mechanisms

| Mechanism | Role in DETR |
|-----------|-------------|
| N object queries | Fixed-size prediction set; enables set prediction framing |
| N x M cost matrix | Encodes class and geometry compatibility for every prediction-target pair |
| Hungarian algorithm | Finds the unique minimum-cost one-to-one assignment |
| Fixed matching before loss | Makes the loss well-defined and differentiable |
| One-to-one training signal | Prevents duplicate predictions; eliminates the need for NMS |
| GIoU loss | Provides non-zero gradient even when predicted and target boxes do not overlap |
| L1 + GIoU combination | Covers both absolute displacement and geometric overlap during training |

---

## References

- Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., and Zagoruyko, S. "End-to-End Object Detection with Transformers." ECCV 2020. https://arxiv.org/abs/2005.12872
- Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., and Savarese, S. "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression." CVPR 2019. https://arxiv.org/abs/1902.09630
- Kuhn, H. W. "The Hungarian Method for the Assignment Problem." Naval Research Logistics Quarterly, 1955.