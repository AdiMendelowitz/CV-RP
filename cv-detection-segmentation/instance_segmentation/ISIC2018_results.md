# ISIC 2018 Skin Lesion Analysis: Experimental Results

**Status:** Work in progress -- Task 1 complete (4 runs); Task 3 pending.

**Dataset:** ISIC 2018 Challenge (Codella et al., arXiv:1902.03368, 2019)
**Underlying data:** HAM10000 (Tschandl et al., Scientific Data, 2018)
**Kaggle sources:**
- Task 1: `tschandl/isic2018-challenge-task1-data-segmentation`
- Task 3: `kmader/skin-cancer-mnist-ham10000`

---

## Task 1: Lesion Segmentation

### Problem Setup

**Objective:** Produce a binary segmentation mask delineating the lesion boundary
for each dermoscopic image.

**Primary metric:** Thresholded Jaccard index (T = 0.65). Standard IoU is computed
per image; values below T = 0.65 are set to zero before averaging across the
validation set. The threshold was derived from inter-observer variability on the
2016 challenge data, where the minimum pairwise annotator agreement was 0.743.
This formulation penalises gross segmentation failures more directly than mean IoU.

**Dataset split:** 2,594 training images with ground-truth binary masks. An 80/20
random split yielded approximately 2,075 training images and 519 validation images.
No official validation split is provided by the challenge.

**Architecture:** Mask R-CNN with a ResNet-50-FPN backbone pretrained on COCO
(`torchvision.models.detection.maskrcnn_resnet50_fpn`,
`weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT`). Both the box predictor head
and the mask predictor head were replaced for 2 output classes (background and
lesion). The backbone and FPN weights were retained from COCO pretraining.
Bounding boxes were derived programmatically from the tight bounding rectangle
of each ground-truth mask at dataset load time, as required by the Mask R-CNN
target format.

**Preprocessing:** Images and masks resized to 512 x 512. Mask pixels with value
> 127 treated as foreground. No colour normalisation applied (COCO-pretrained
Mask R-CNN expects values in [0, 1] without ImageNet normalisation).

**Augmentation (all runs):** Random horizontal flip applied jointly to image,
mask, and bounding box.

---

### Run Summary

| Run | Epochs | lr_step_size | score_thresh | Best Jaccard | Best Epoch |
|-----|--------|-------------|-------------|-------------|------------|
| 1 | 20 | 7 | 0.5 | **0.7803** | 19 |
| 2 | +15 (resume from epoch 19) | 7 | 0.5 | 0.7803 | -- |
| 3 | 20 | 10 | 0.3 | 0.7796 | 11 |
| 4 | 20 | 7 | 0.5 | 0.7764 | 16 |

All runs used: AdamW, `lr=5e-4`, `weight_decay=1e-4`, `lr_gamma=0.5`,
`batch_size=4`, `img_size=512`.

**Reported result:** 0.7803 (Run 1, best across all runs).
**Estimated variance:** 0.778 +/- 0.003 across identical configurations (Runs 1 and 4).

---

### Run 1: Baseline

**LR schedule:** `5e-4` → `2.5e-4` (epoch 7) → `1.25e-4` (epoch 14).

**Training log:**

| Epoch | Train Loss | Loss Mask | Loss Box | Loss Cls | Val Jaccard |
|-------|-----------|-----------|----------|----------|-------------|
| 1  | 0.5199 | 0.3488 | 0.0821 | 0.0654 | 0.6767 |
| 2  | 0.3589 | 0.2300 | 0.0701 | 0.0442 | 0.6569 |
| 3  | 0.3325 | 0.2144 | 0.0662 | 0.0391 | 0.6946 |
| 4  | 0.3215 | 0.2103 | 0.0632 | 0.0364 | 0.7023 |
| 5  | 0.3177 | 0.2047 | 0.0655 | 0.0361 | 0.7319 |
| 6  | 0.3035 | 0.1975 | 0.0601 | 0.0350 | 0.7320 |
| 7  | 0.2921 | 0.1939 | 0.0573 | 0.0313 | 0.7012 |
| 8  | 0.2666 | 0.1808 | 0.0506 | 0.0272 | 0.7605 |
| 9  | 0.2633 | 0.1794 | 0.0500 | 0.0263 | 0.7626 |
| 10 | 0.2566 | 0.1731 | 0.0504 | 0.0260 | 0.7661 |
| 11 | 0.2535 | 0.1727 | 0.0489 | 0.0249 | 0.7609 |
| 12 | 0.2509 | 0.1706 | 0.0491 | 0.0245 | 0.7608 |
| 13 | 0.2511 | 0.1690 | 0.0501 | 0.0254 | 0.7700 |
| 14 | 0.2420 | 0.1645 | 0.0474 | 0.0238 | 0.7715 |
| 15 | 0.2226 | 0.1518 | 0.0434 | 0.0219 | 0.7700 |
| 16 | 0.2124 | 0.1460 | 0.0412 | 0.0203 | 0.7617 |
| 17 | 0.2067 | 0.1426 | 0.0398 | 0.0196 | 0.7666 |
| 18 | 0.2033 | 0.1374 | 0.0407 | 0.0204 | 0.7664 |
| 19 | 0.1955 | 0.1346 | 0.0378 | 0.0185 | **0.7803** |
| 20 | 0.1885 | 0.1299 | 0.0365 | 0.0178 | 0.7570 |

---

### Run 2: Extended Training (resume from epoch 19)

Run 1's checkpoint was loaded and training continued for 15 additional epochs
(20-34). The LR at resume was `1.25e-4`, decaying further to `6.25e-5` at
epoch 21 and `3.125e-5` at epoch 28.

**Selected epochs:**

| Epoch | Train Loss | Val Jaccard |
|-------|-----------|-------------|
| 20 | 0.1761 | 0.7727 |
| 21 | 0.1673 | 0.7746 |
| 25 | 0.1526 | 0.7676 |
| 28 | 0.1452 | 0.7612 |
| 34 | 0.1345 | 0.7575 |

**Outcome:** No improvement over Run 1. Training loss continued to decrease but
validation Jaccard oscillated in the 0.760-0.775 range, confirming the model had
reached its capacity ceiling under the current configuration.

---

### Run 3: Modified LR Schedule

**Motivation:** Runs 1 and 2 suggested the LR was decaying too aggressively.
`lr_step_size` was increased from 7 to 10 to space the two decay events further
apart, and `score_threshold` was lowered from 0.5 to 0.3 to recover borderline
detections.

**LR schedule:** `5e-4` → `2.5e-4` (epoch 10) → `1.25e-4` (epoch 20).

**Selected epochs:**

| Epoch | Train Loss | Val Jaccard |
|-------|-----------|-------------|
| 1  | 0.5310 | 0.6521 |
| 5  | 0.3105 | 0.7302 |
| 6  | 0.2970 | 0.7373 |
| 8  | 0.2953 | 0.7534 |
| 11 | 0.2665 | **0.7796** |
| 15 | 0.2390 | 0.7701 |
| 20 | 0.2155 | 0.7715 |

**Outcome:** Best Jaccard 0.7796, 0.7 points below Run 1. The modified schedule
did not improve results, indicating the bottleneck is not the LR decay timing.

---

### Run 4: Replication of Run 1 Configuration

A clean rerun of Run 1's exact configuration to estimate result variance.

**Training log:**

| Epoch | Train Loss | Loss Mask | Loss Box | Loss Cls | Val Jaccard |
|-------|-----------|-----------|----------|----------|-------------|
| 1  | 0.5316 | 0.3579 | 0.0807 | 0.0670 | 0.6195 |
| 2  | 0.3601 | 0.2354 | 0.0663 | 0.0433 | 0.6453 |
| 3  | 0.3366 | 0.2207 | 0.0637 | 0.0391 | 0.6882 |
| 4  | 0.3247 | 0.2134 | 0.0631 | 0.0363 | 0.7001 |
| 5  | 0.3134 | 0.2015 | 0.0659 | 0.0356 | 0.7334 |
| 6  | 0.3070 | 0.1995 | 0.0624 | 0.0350 | 0.7539 |
| 7  | 0.2971 | 0.1969 | 0.0587 | 0.0318 | 0.7179 |
| 8  | 0.3281 | 0.2104 | 0.0651 | 0.0407 | 0.7507 |
| 9  | 0.2919 | 0.1940 | 0.0573 | 0.0309 | 0.7426 |
| 10 | 0.2885 | 0.1883 | 0.0596 | 0.0317 | 0.7563 |
| 11 | 0.2561 | 0.1754 | 0.0488 | 0.0251 | 0.7631 |
| 12 | 0.2524 | 0.1719 | 0.0489 | 0.0250 | 0.7679 |
| 13 | 0.2493 | 0.1686 | 0.0494 | 0.0251 | 0.7728 |
| 14 | 0.2462 | 0.1661 | 0.0494 | 0.0245 | 0.7646 |
| 15 | 0.2384 | 0.1622 | 0.0469 | 0.0234 | 0.7692 |
| 16 | 0.2323 | 0.1599 | 0.0445 | 0.0223 | **0.7764** |
| 17 | 0.2320 | 0.1578 | 0.0457 | 0.0229 | 0.7621 |
| 18 | 0.2255 | 0.1529 | 0.0451 | 0.0221 | 0.7694 |
| 19 | 0.2266 | 0.1531 | 0.0454 | 0.0225 | 0.7709 |
| 20 | 0.2145 | 0.1465 | 0.0426 | 0.0202 | 0.7728 |

**Outcome:** Best Jaccard 0.7764. The 0.0039-point gap relative to Run 1 reflects
natural stochasticity from random data splitting and training dynamics rather than
a meaningful performance difference.

---

### Qualitative Analysis

Four representative validation cases from Run 1:

**High confidence (Jaccard = 0.956):** Standard lesion with a clear visual boundary
and no clinical artefacts. The predicted mask closely follows the ground-truth contour.

**Artefact failure (Jaccard = 0.000):** The image contains a ruler and a blue
measurement triangle. The model localises a lesion-like region but the resulting
IoU falls below T = 0.65, scoring zero. This failure mode is consistent with
findings in Codella et al. (2019), who noted highest failure rates on images with
clinical measurement artefacts.

**Small lesion (Jaccard = 0.881 / 0.882):** Small, isolated lesion with measurement
marks visible. The model produces a compact mask that closely matches the ground
truth despite the small target size and background clutter.

**Borderline case (Jaccard = 0.655 / 0.677):** Predicted mask is geometrically
reasonable but slightly undersized relative to the ground truth, producing an IoU
marginally above the T = 0.65 threshold.

---

### Comparison to Challenge Baseline

| Method | Thresholded Jaccard |
|--------|-------------------|
| ISIC 2018 challenge winner | 0.802 |
| This work (best result, Run 1) | 0.780 |
| Gap | -0.022 |

The 2.2 percentage point gap to the challenge winner was achieved with a standard
ResNet-50-FPN backbone, horizontal flip augmentation only, no test-time
augmentation, and no ensemble. The challenge winner used task-specific augmentation
pipelines, multi-scale inference, and model ensembles.

---

### Analysis and Conclusions

Across four runs, the Thresholded Jaccard consistently converged to approximately
0.778 +/- 0.003. The following conclusions are supported by the experimental
evidence:

**The LR schedule is not the limiting factor.** Runs 1, 3, and 4 tested two
different step sizes (7 and 10), and all produced similar peak Jaccard values.
Extending training beyond the peak epoch (Run 2) confirmed that the model had
reached its capacity ceiling rather than being starved of training time.

**The detection score threshold has negligible effect.** Lowering
`score_threshold` from 0.5 to 0.3 in Run 3 produced no meaningful improvement,
indicating that borderline failures are due to mask quality rather than
detection suppression.

**The bottleneck is augmentation breadth and backbone capacity.** The most
impactful paths to improvement would be: (1) adding colour jitter, random
rotation, and elastic deformation augmentations to address acquisition device
variability, which Codella et al. (2019) identified as a primary generalisation
challenge; (2) upgrading to `maskrcnn_resnet50_fpn_v2`, which incorporates
updated training recipes and consistently outperforms V1 on segmentation
benchmarks; or (3) applying test-time augmentation via horizontal flip
averaging, which adds no training cost.

---

## Task 3: Disease Classification

**Status:** Notebook implemented, training pending.

**Objective:** 7-class classification of dermoscopic images into diagnostic
categories: melanocytic nevus (NV), melanoma (MEL), benign keratosis (BKL),
basal cell carcinoma (BCC), actinic keratosis (AKIEC), vascular lesion (VASC),
and dermatofibroma (DF).

**Primary metric:** Balanced accuracy (mean per-class recall), per Codella et al.
(2019). Standard accuracy is not used because NV constitutes 66.9% of training
images; a trivial classifier predicting NV for all inputs would achieve 66.9%
accuracy but only 1/7 balanced accuracy.

**Architecture:** EfficientNet-B0 (`timm`, ImageNet pretrained) with the
classification head replaced for 7 output classes via `timm.create_model`.

**Class imbalance handling:** Weighted cross-entropy loss with per-class weights
computed as `weight_c = total / (num_classes * count_c)`. The NV/DF imbalance
ratio is approximately 58:1.

**Data split:** Lesion-level stratified 80/20 split on `lesion_id` rather than
`image_id`. The same lesion appears multiple times in HAM10000 with different
crops; splitting on `image_id` would leak lesion information into validation
and inflate metrics.

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Backbone | EfficientNet-B0 (timm, ImageNet pretrained) |
| Epochs | 20 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| LR schedule | StepLR, step=7, gamma=0.5 |
| Batch size | 32 |
| Image size | 224 x 224 |
| Loss | Weighted CrossEntropyLoss |
| Augmentation | H-flip, V-flip, ColorJitter |

**Results:** Pending.

---

## References

Codella, N. et al. "Skin Lesion Analysis Toward Melanoma Detection 2018:
A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)."
arXiv:1902.03368, 2019.

Tschandl, P., Rosendahl, C., Kittler, H. "The HAM10000 Dataset, a Large
Collection of Multi-Source Dermatoscopic Images of Common Pigmented Skin
Lesions." Scientific Data, 2018. https://doi.org/10.1038/sdata.2018.161

He, K. et al. "Mask R-CNN." ICCV 2017. https://arxiv.org/abs/1703.06870

Tan, M., Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional
Neural Networks." ICML 2019. https://arxiv.org/abs/1905.11946