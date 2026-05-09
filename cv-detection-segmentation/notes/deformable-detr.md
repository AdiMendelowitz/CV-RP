# Deformable DETR: Deformable Transformers for End-to-End Object Detection

**Reference:** Zhu et al., ICLR 2021 (Oral). arXiv:2010.04159.  
**Focus:** Section 3 -- Revisiting Transformers and DETR; deformable attention module.

---

## Why is vanilla DETR slow to train?

DETR (Carion et al., ECCV 2020) applies standard Transformer self-attention directly
over image feature map pixels. In the encoder, both query and key elements are pixels,
so the self-attention complexity is O(H^2 * W^2 * C) -- quadratic in the number of
spatial positions. On a typical ResNet feature map this means attending to every pixel
against every other pixel in the same map.

The consequence is two-fold. First, at initialization the attention weights are nearly
uniform across all Nk pixel positions (Amqk ~ 1/Nk when Nk is large), producing
ambiguous gradients that give no useful learning signal about which spatial locations
matter. The model must learn from scratch, over many epochs, to concentrate attention
on sparse meaningful regions. Second, the quadratic complexity makes high-resolution
feature maps computationally infeasible, which forces DETR to operate on a single
low-resolution feature map and limits its ability to detect small objects.

On COCO, vanilla DETR requires 500 training epochs to converge -- approximately
10 to 20 times longer than Faster R-CNN.

---

## What does deformable attention do differently?

Deformable attention replaces the dense all-pairs attention with a learned sparse
sampling scheme. For each query element q with feature z_q and a reference point p_q
on the feature map, the module predicts K sampling offsets Delta p_mqk relative to
that reference point. Attention weights A_mqk are predicted from z_q alone (not from
query-key dot products). The output for attention head m is then:

    DeformAttn(z_q, p_q, x) =
        sum_m W_m * sum_{k=1}^{K} A_mqk * W_m' * x(p_q + Delta p_mqk)

where K is a small fixed number of sampling points (K=4 in the paper), independent
of feature map resolution. The attention weights satisfy sum_k A_mqk = 1.

This eliminates the quadratic dependence on spatial size: complexity is now O(2*Nq*C^2 + K*Nq*C*M), linear in both query count and sampling points. Because the model learns
which K locations to attend to rather than computing compatibility against all pixels,
the attention weights are no longer initialized to a near-uniform distribution, and
the convergence problem is resolved structurally rather than through extended training.

The module extends naturally to multi-scale feature maps by attending across all
scales simultaneously, aggregating features from {l=1..L} levels without requiring
a separate FPN. This directly addresses DETR's small-object weakness.

---

## Practical training speedup

Deformable DETR matches or exceeds DETR's COCO detection performance using 10x
fewer training epochs. Where DETR requires 500 epochs, Deformable DETR converges
in 50 epochs. The improvement is particularly pronounced on small objects, where
multi-scale deformable attention enables high-resolution feature processing that was
computationally infeasible for vanilla DETR.

---

## Summary

The core insight is that global dense attention is the wrong prior for image feature
maps. Most spatial locations are irrelevant to any given query; forcing the model to
attend to all of them creates an initialization problem (uniform weights, ambiguous
gradients) and a complexity problem (quadratic scaling). Deformable attention sidesteps
both by learning which sparse locations to attend to, directly from the query feature,
producing a module that is both faster and better-behaved during optimization.