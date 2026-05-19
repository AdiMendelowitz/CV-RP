# Deformable DETR: Deformable Transformers for End-to-End Object Detection

**Reference:** Zhu et al., “Deformable DETR: Deformable Transformers for End-to-End Object Detection,” ICLR 2021 (Oral), arXiv:2010.04159.  
**Focus:** Section 3 – revisiting Transformers and DETR; deformable attention module.

---

## Why is vanilla DETR slow to train?

DETR (Carion et al., ECCV 2020) applies standard Transformer self‑attention directly over image feature map elements. In the encoder, both queries and keys are spatial positions in the flattened feature map, so the self‑attention complexity is `O(N_q N_k C)`, where `N_q` and `N_k` are the numbers of query and key positions, and `C` is the channel dimension. When queries and keys are all pixels on a feature map (so `N_q = N_k = HW`), this becomes quadratic in spatial resolution: `O((HW)^2 C)`.

Two main consequences:

- **Uniform, uninformative attention at initialization.**  
  With dense attention over many positions, randomly initialized query–key dot products tend to produce near‑uniform attention distributions over all `N_k` spatial locations (i.e. each weight `~= 1/N_k`), which provides little guidance about which locations matter. The model must learn to sharpen these weights over many epochs before strong, sparse correspondences emerge.

- **Quadratic complexity prevents high‑resolution features.**  
  Because cost grows as `O((HW)^2)`, using high‑resolution feature maps is computationally and memory‑prohibitive. In practice, DETR works on a single low‑resolution feature map (e.g. from a downsampled stage of ResNet), which limits performance on small objects that benefit from finer spatial detail.

On COCO, vanilla DETR typically requires **400–500 training epochs** to match Faster R‑CNN performance—roughly **10–20× more epochs** than standard detector schedules.

---

## What does deformable attention do differently?

Deformable attention replaces dense all‑pairs attention with a **learned sparse sampling scheme** around a small set of reference points.

For each query element `q` with feature `z_q` and a reference point `p_q` in (normalized) spatial coordinates, deformable attention predicts:

- For each attention head `m`, a set of **K sampling offsets** `Delta p_mqk` relative to `p_q`.  
- Corresponding attention weights `A_mqk`, predicted from the query feature (and head) rather than from pairwise query–key dot products.

The output for head `m` can be written (conceptually) as:

```text
DeformAttn(z_q, p_q, x) =
    Σ_{m=1}^M W_m · Σ_{k=1}^K A_{mqk} · W_m' x(p_q + Δp_{mqk})
```

where:

- `x(*)` samples the feature map(s) at (potentially fractional) locations via bilinear interpolation.  
- `K` is a small fixed number of sampling points per head (e.g., K = 4 or 8 in the paper).  
- The weights per head satisfy `sum_k A_mqk = 1`.

**Complexity.**  

- Standard multi‑head attention over N queries and N keys has complexity `O(2 N C^2 + N^2 C)` in the formulation of Zhu et al.  
- Deformable attention reduces this to `O(2 N_q C^2 + K N_q C M)` when each query attends to only K sampled positions per head, independent of feature map resolution.

Thus, deformable attention is **linear in the number of queries** and in K, rather than quadratic in spatial positions, making it much more scalable to high‑resolution features.

**Convergence behaviour.**  

Because attention weights are predicted directly from each query feature over a small set of locations, attention is not forced to be uniform over thousands of keys at initialization. The architecture structurally encourages sparse, localised focus around reference points from the beginning, which improves gradient signal and greatly speeds up training convergence.

**Multi‑scale extension.**  

Deformable DETR generalises the attention to **multi‑scale feature maps**:\

- It uses multiple backbone feature levels (e.g., from different ResNet stages).  
- Queries attend to a small number of sampling points across *all* levels simultaneously (multi‑scale deformable attention), without requiring an explicit FPN neck.  

This multi‑scale design directly addresses DETR’s small‑object weakness by allowing the model to aggregate information from higher‑resolution feature maps without incurring a quadratic attention cost.

---

## Practical training speedup

Empirically, Deformable DETR:

- Achieves **better COCO AP** than vanilla DETR, particularly on small objects.  
- Converges in **≈50 epochs**, roughly a **10× reduction** compared to a 500‑epoch DETR schedule used to match Faster R‑CNN.  
- Offers substantial wall‑clock savings; reported experiments show up to **10× fewer epochs, ≈20× less training time, and ~1.6× faster inference** for comparable backbones and settings.

The improvement is especially pronounced on small objects because multi‑scale deformable attention lets the detector leverage high‑resolution features that standard DETR could not afford.

---

## Summary

The core insight is that **global dense attention is a poor inductive bias for image feature maps** in detection:

- Most spatial locations are irrelevant to any given query; dense attention forces the model to score all of them, causing initial attention to be nearly uniform and gradients to be ambiguous.  
- Quadratic complexity in spatial resolution makes it impractical to use multi‑scale, high‑resolution features, hurting small‑object detection.

Deformable attention resolves both issues by:

- Letting each query attend to a **small, learned set of sampling points** near a reference location, predicted directly from query features.  
- Making attention complexity **linear in the number of queries and sampling points**, rather than quadratic in the number of pixels.  
- Extending naturally to **multi‑scale** feature maps, improving small‑object performance without an explicit FPN.

This yields a DETR‑style detector that is both faster to train and better‑behaved computationally, while preserving the elegance of end‑to‑end set prediction.
