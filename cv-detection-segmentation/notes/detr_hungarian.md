# DETR and the Hungarian Matching Algorithm

**Paper:** “End-to-End Object Detection with Transformers”  
**Authors:** Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko  
**Venue:** ECCV 2020  
**arXiv:** https://arxiv.org/abs/2005.12872[file:208][web:198]

---

## Background: Why standard detectors produce duplicates

Anchor‑based and grid‑based detectors (Faster R‑CNN, YOLO, SSD) partition the image into a grid and assign one or more anchor boxes to each cell.[file:208][web:198] Multiple cells or anchors often fire on the same object, producing clusters of overlapping high‑confidence boxes for a single ground‑truth instance. This is a structural consequence of framing detection as dense, per‑location classification plus regression: there is no explicit mechanism enforcing a one‑to‑one mapping between predictions and objects.[file:208][web:198]

The standard remedy is **non‑maximum suppression (NMS)**: sort predictions by confidence, keep the highest‑scoring one, and suppress any prediction whose IoU with a kept box exceeds a threshold.[file:208] NMS works well in practice but:

- Introduces task‑specific hyperparameters (IoU threshold) that must be tuned.  
- Is greedy and can fail in crowded scenes, suppressing valid boxes when objects are close together.[file:208][web:198]

DETR removes the need for NMS by changing the prediction structure itself: it predicts a fixed‑size set of **N object queries** and uses a **one‑to‑one assignment** between predictions and ground‑truth objects during training.[file:208][web:198][web:204] Redundancy is discouraged at its source, so post‑hoc suppression is unnecessary.

---

## The matching cost matrix

### Setup

Let:

- \(N\): fixed number of object queries (predictions) output by DETR’s decoder.  
- \(M\): number of ground‑truth objects in a particular image.

DETR always outputs exactly \(N\) predictions per image, independent of \(M\).[file:208][web:198] For images with fewer than \(N\) objects, the extra predictions are assigned to a special “no object” class during training.

Define a cost matrix \(C \in \mathbb{R}^{N \times M}\), where entry \(C[i, j]\) measures how “expensive” it is to match prediction \(i\) with ground‑truth object \(j\).[file:208][web:198] Low cost indicates a good match; high cost indicates incompatibility.

### Cost composition

DETR uses a weighted sum of classification and box compatibility terms:[file:208][web:198]

```text
C[i, j] = - λ_cls  · p_i(c_j)
          + λ_L1   · ||b_i - b_j||_1
          + λ_GIoU · L_GIoU(b_i, b_j)
```

where:

- \(p_i(c_j)\): predicted probability that prediction \(i\) has class \(c_j\) (the class of ground‑truth \(j\)).  
- \(b_i\): predicted box (normalized center‑x, center‑y, width, height).  
- \(b_j\): ground‑truth box in the same parameterization.  
- \(L_{\text{GIoU}}(b_i, b_j)\): GIoU‑based box loss (defined later).  
- \(\lambda_{\text{cls}}, \lambda_{\text{L1}}, \lambda_{\text{GIoU}}\): scalar weights.[file:208][web:198]

The classification term uses a negative probability (rather than negative log‑probability) in the matching cost to avoid computing a full softmax over classes for each potential matching at this stage.[file:208][web:198] The L1 and GIoU terms encourage geometric alignment between predicted and ground‑truth boxes.

### The Hungarian algorithm

Given \(C\), DETR uses the **Hungarian algorithm** to find the minimum‑cost one‑to‑one assignment between predictions and ground‑truth boxes.[file:208][web:198]

Formally, over all permutations \(\sigma\) of \(\{1, \dots, N\}\), we solve:

```text
σ* = argmin_{σ ∈ S_N} Σ_{j=1}^{M} C[σ(j), j]
```

Only the first \(M\) entries of the permutation are used; the remaining \(N - M\) predictions are treated as “no object” for that image.[file:208][web:198]

The Hungarian algorithm has worst‑case complexity \(O((\max(N, M))^3)\).[file:208] In DETR, \(N\) is typically fixed at 100 and \(M\) (objects per image) is much smaller than \(N\), so the cost of matching is negligible compared to the forward/backward passes.[web:198][web:204]

---

## Why the set prediction loss requires a fixed matching

Object detection is fundamentally a **set prediction** task: labels are an unordered set of (class, box) pairs.[file:208][web:198] A naive approach might define a loss that averages over all possible assignments between predictions and ground‑truth sets, but this is both computationally and conceptually problematic.

1. **Assignment instability and gradient conflict.**  
   If you average loss over many possible permutations, the “best” assignment changes as model weights evolve. A single predicted box might be partially rewarded for matching several different ground‑truth boxes across permutations, leading to conflicting gradients that pull it in inconsistent directions.[file:208][web:198]

2. **Combinatorial explosion.**  
   For \(N\) predictions, there are \(N!\) possible permutations; even if you restrict to the \(M!\) assignments over the ground‑truth subset, this is infeasible for realistic N (e.g., 100 queries → 100! permutations).[file:208]

DETR resolves this by **decoupling matching from loss computation**:[file:208][web:198]

1. First, Hungarian matching finds a single optimal assignment \(\sigma^\*\) given the current predictions and cost matrix.  
2. Then, **given this fixed assignment**, DETR computes a standard detection loss over matched pairs.

Formally, the DETR loss for an image is:[file:208][web:198]

```text
L_DETR = Σ_{j=1}^{M} [ - log p_{σ*(j)}(c_j)
                       + 1[c_j ≠ no_obj] · L_box(b_{σ*(j)}, b_j) ]
         + Σ_{i ∉ {σ*(1..M)}} [ - log p_i(no_obj) ]
```

where \(L_{\text{box}}\) combines L1 and GIoU loss:[file:208][web:198]

```text
L_box(b_pred, b_gt) = λ_L1 · ||b_pred - b_gt||_1
                      + λ_GIoU · L_GIoU(b_pred, b_gt)
```

Key points:

- The matching step is **non‑differentiable**, but it is treated as a fixed combinatorial operation; gradients are taken only with respect to the loss terms once σ* is chosen.[file:208][web:198]  
- The Hungarian solution depends smoothly on predictions, so assignments change gradually during training in practice, leading to stable optimisation.[file:208][web:198]

---

## Why DETR does not need NMS

Standard detector duplicates arise from a **many‑to‑one** training structure: many anchors can be assigned to the same ground‑truth object, and all of them are encouraged to predict that object, so at test time multiple high‑confidence boxes can survive per object.[file:208][web:198]

In DETR:

- Hungarian matching enforces a **one‑to‑one assignment**: each ground‑truth object is matched to exactly one prediction, and that prediction receives the positive training signal for that object.[file:208][web:198]  
- All other predictions are encouraged towards the “no object” class for that image, which discourages them from producing high‑confidence object predictions for the same region.[file:208]

As training converges, the distribution of predictions adapts to this constraint:

- Producing two high‑confidence predictions for the same object would mean one of them has been repeatedly trained as “no object” in similar configurations, which the model learns to avoid.  
- Consequently, DETR typically produces at most one confident prediction per object, and inference can be run without any NMS step.[file:208][web:198][web:204]

This property is specific to DETR’s combination of fixed‑size output set, one‑to‑one matching, and loss design; grid‑based detectors like YOLO do not share it, since each grid cell optimises independently and is not penalised for redundant predictions in neighbouring cells.[file:208][web:198]

---

## GIoU as a box regression loss

### The problem with IoU as a loss

Intersection over Union (IoU) for boxes A and B is:

```text
IoU(A, B) = |A ∩ B| / |A ∪ B|
```

IoU is an excellent **evaluation metric** but problematic as a direct **training loss**:[file:208][web:198]

- If boxes do **not overlap**, \(|A ∩ B| = 0\) and IoU = 0, regardless of how far apart they are. The gradient of IoU w.r.t. the predicted box coordinates is also zero almost everywhere in the non‑overlapping regime, so IoU loss provides no signal to move the prediction towards the target.[file:208][web:191]  
- IoU’s scale invariance means that proportional errors yield the same IoU penalty for small and large boxes; this can hurt convergence for small objects which need stronger localisation signals.[file:208]

### GIoU

Generalized IoU (GIoU) was proposed by Rezatofighi et al. (CVPR 2019) to address IoU’s zero‑gradient issue for non‑overlapping boxes.[file:208][web:191]

Let C be the smallest axis‑aligned box that encloses A and B. Then:

```text
GIoU(A, B) = IoU(A, B) - |C \ (A ∪ B)| / |C|
```

- The second term penalises the area of C not covered by A ∪ B. It is zero when A and B have identical boxes and strictly positive whenever they do not fully overlap.[file:208][web:191]  
- GIoU ranges from −1 (no overlap, boxes far apart) to 1 (perfect overlap), whereas IoU ranges from 0 to 1.[file:208][web:191]

DETR uses the standard GIoU loss:[file:208][web:198]

```text
L_GIoU(b_i, b_j) = 1 - GIoU(b_i, b_j)
```

This loss is:

- Zero for perfectly overlapping boxes.  
- Positive with non‑zero gradients even when boxes do not overlap, providing a useful signal early in training when predictions are crude.[file:208][web:191]

### Why DETR combines L1 and GIoU

DETR combines an **L1 loss on normalized box coordinates** with the GIoU loss:[file:208][web:198]

- GIoU captures **geometric overlap quality** and is scale‑invariant.  
- L1 penalises **absolute coordinate differences**, ensuring that large displacements are penalised even when IoU/GIoU gradients might be small.

Using both gives complementary feedback: L1 drives boxes toward correct positions and sizes in coordinate space, while GIoU refines overlap and shape alignment, especially when boxes start non‑overlapping.[file:208][web:198] This combination is used **both** in the box regression loss and in the Hungarian matching cost, aligning the training objective with the assignment metric.[file:208][web:198]

---

## Summary of key mechanisms

| Mechanism                 | Role in DETR                                                                 |
|---------------------------|------------------------------------------------------------------------------|
| Fixed N object queries    | Frames detection as fixed‑size set prediction per image                      |
| N × M cost matrix         | Encodes class and geometric compatibility for every prediction–target pair   |
| Hungarian algorithm       | Finds minimum‑cost one‑to‑one matching between predictions and ground truth  |
| Fixed matching before loss| Makes the set loss well‑defined and differentiable                           |
| One‑to‑one supervision    | Prevents duplicates; removes structural need for NMS                         |
| GIoU loss                 | Provides non‑zero gradients even for non‑overlapping boxes                   |
| L1 + GIoU combination     | Covers both absolute displacement and overlap quality                         |

[file:208][web:198][web:191]

---

## References

- Carion et al., “End-to-End Object Detection with Transformers,” ECCV 2020.[web:198][web:204]  
- Rezatofighi et al., “Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression,” CVPR 2019.[web:191]  
- Kuhn, “The Hungarian Method for the Assignment Problem,” Naval Research Logistics Quarterly, 1955.[file:208]

