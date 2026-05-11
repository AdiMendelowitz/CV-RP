# Instance Segmentation, Detection, and Medical Imaging

From-scratch implementations of core instance segmentation components -- RoI
Align, Hungarian matching loss, and a convolutional mask head -- combined with
fine-tuned Mask R-CNN and EfficientNet models for end-to-end dermoscopic
lesion analysis on the ISIC 2018 challenge.

---

## Modules

| File | Description | Reference |
|------|-------------|-----------|
| `hungarian_loss.py` | Bipartite matching loss for set prediction: GIoU, L1 box regression, cross-entropy classification, scipy linear sum assignment | Carion et al., ECCV 2020 |
| `roi_align.py` | RoI Align with bilinear interpolation, implemented from scratch in PyTorch | He et al., ICCV 2017 |
| `mask_head.py` | Convolutional mask prediction head producing per-instance binary masks | He et al., ICCV 2017 |
| `focal_loss.py` | Alpha-balanced multi-class focal loss; reduces exactly to weighted CE at gamma=0 | Lin et al., ICCV 2017 |
| `isic_pipeline.py` | End-to-end lesion segmentation and classification pipeline chaining Mask R-CNN with EfficientNet | -- |

---

## Installation

```bash
# From the repository root
pip install -e .
```

Dependencies are specified in `pyproject.toml`. All GPU training was performed
on Kaggle T4. Local scripts run on CPU.

---

## Tests

```bash
pytest cv-detection-segmentation/instance_segmentation/ -v
```

42 tests pass across three test files covering Hungarian matching, RoI Align,
mask head, and focal loss.

| Test file | Tests | Coverage |
|-----------|-------|----------|
| `test_hungarian.py` | 17 | GIoU bounds, cost matrix, assignment optimality, set prediction loss |
| `test_roi_align_mask_head.py` | 12 | Output shape, bilinear interpolation correctness, gradient flow |
| `test_focal_loss.py` | 13 | CE equivalence at gamma=0, monotone down-weighting, input validation |

---

## Results

Full experimental logs, per-class recall tables, training curves, and analysis
are in `ISIC2018_results.md`.

### Task 1: Lesion Segmentation (Mask R-CNN, ResNet-50-FPN)

Trained on ISIC 2018 Task 1 (`tschandl/isic2018-challenge-task1-data-segmentation`),
2,594 images, 80/20 random split, 512 x 512 input, horizontal flip augmentation only.

| Method | Thresholded Jaccard (T=0.65) |
|--------|------------------------------|
| ISIC 2018 challenge winner | 0.802 |
| This work (Run 5, best) | 0.7822 |
| Estimated variance across runs | 0.780 +/- 0.003 |

### Task 3: Disease Classification (EfficientNet-B0/B3)

Trained on HAM10000 (`kmader/skin-cancer-mnist-ham10000`), 7-class
classification, lesion-level stratified 80/20 split on `lesion_id`, 224 x 224
input, progressive unfreezing, CosineAnnealingLR, albumentations augmentation.
Primary metric: balanced accuracy (mean per-class recall).

| Model | Balanced Accuracy | MEL Recall | Params |
|-------|------------------|------------|--------|
| ISIC 2018 challenge winner | 0.885 | -- | -- |
| EfficientNet-B0, weighted CE (Run 5) | 0.7457 | 0.624 | 4,016,515 |
| EfficientNet-B0, focal loss gamma=2.0 (Run 6) | 0.7376 | 0.592 | 4,016,515 |
| EfficientNet-B3, weighted CE (Run 7) | 0.7498 | 0.505 | 10,706,991 |

All results are validation set estimates on a single random split. Run-to-run
variance on identical configurations is approximately 2pp, driven by the
stochastic lesion-level split rather than configuration sensitivity.

### End-to-End Pipeline

Mask R-CNN (Task 1 checkpoint) chained with EfficientNet-B0 Run 5 (Task 3
checkpoint), evaluated on 50 HAM10000 validation images. Jaccard ground truth
is not available because the Task 1 and HAM10000 datasets use disjoint ISIC
image ID ranges; segmentation quality is assessed qualitatively.

| Metric | Value |
|--------|-------|
| Images evaluated | 50 |
| Detection failures | 0 / 50 (0.0%) |
| Classification failures given detection | 15 / 50 (30.0%) |
| Full pipeline success | 35 / 50 (70.0%) |
| Pipeline balanced accuracy | 0.5219 |
| Standalone classifier balanced accuracy (Run 5) | 0.7441 |

The 22pp gap between pipeline and standalone classification accuracy reflects
the distribution shift introduced by crop-based classification: EfficientNet-B0
was trained on full resized images and receives a tightly cropped region in
the pipeline, losing the spatial context it relies on.

![Pipeline outputs](outputs/pipeline/pipeline_outputs.png)

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `isic2018-task1-segmentation.ipynb` | Mask R-CNN training, evaluation, and qualitative analysis on ISIC 2018 Task 1 |
| `isic2018-task3-classification.ipynb` | EfficientNet training, focal loss and B3 experiments, confusion matrix, per-class F1 |
| `isic_pipeline.ipynb` | End-to-end pipeline evaluation and failure case analysis on 50 HAM10000 images |

All notebooks are stripped of output cells before committing (`nbstripout`).
Kaggle links and full training logs are in `ISIC2018_results.md`.

---

## Notes

| File | Content |
|------|---------|
| `../notes/mask_rcnn.md` | Mask R-CNN architecture and RoI Align derivation |
| `../notes/mask_rcnn_paper.md` | Paper reading notes: He et al., ICCV 2017 |
| `../notes/detr_paper.md` | Paper reading notes: Carion et al., ECCV 2020 |
| `../notes/detr_hungarian.md` | Hungarian matching and bipartite assignment derivation |
| `../notes/roialign_paper.md` | RoI Align implementation notes |
| `../notes/deformable-detr.md` | Deformable DETR: deformable attention and training speedup |
| `../notes/medical_imaging.md` | Medical imaging domain notes |
| `../notes/skim_papers_background.md` | Background paper skims |
| `hungarian_roi_align_walkthrough.md` | Step-by-step walkthrough of Hungarian matching and RoI Align |

---

## Usage

### Focal loss

```python
import torch
from focal_loss import focal_loss

logits  = torch.randn(32, 7)          # (N, C) raw model outputs
targets = torch.randint(0, 7, (32,))  # (N,) ground-truth class indices
alpha   = torch.tensor([1.5, 1.2, 0.8, 3.0, 1.1, 0.5, 4.0])  # per-class weights

loss = focal_loss(logits, targets, alpha, gamma=2.0)
```

### Hungarian matching loss

```python
import torch
from hungarian_loss import SetPredictionLoss

criterion = SetPredictionLoss(num_classes=91)

pred_logits = torch.randn(2, 100, 92)   # (B, num_queries, num_classes + 1)
pred_boxes  = torch.rand(2, 100, 4)     # (B, num_queries, 4) normalised cx cy w h
targets = [
    {"labels": torch.tensor([1, 2]), "boxes": torch.rand(2, 4)},
    {"labels": torch.tensor([3]),    "boxes": torch.rand(1, 4)},
]

losses = criterion(pred_logits, pred_boxes, targets)
# losses: {"loss_ce": tensor, "loss_bbox": tensor, "loss_giou": tensor}
```

### RoI Align

```python
import torch
from roi_align import RoIAlign

roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0 / 16)

feature_map = torch.randn(1, 256, 32, 32)                      # (B, C, H, W)
boxes       = torch.tensor([[0, 10., 10., 50., 50.]])           # (N, 5): batch_idx x1 y1 x2 y2

features = roi_align(feature_map, boxes)                        # (N, 256, 7, 7)
```

### End-to-end pipeline

```python
import torch
from isic_pipeline import ISICPipeline

# Both models must be pre-loaded and on the same device before passing to the pipeline.
pipeline = ISICPipeline(
    segment_model=seg_model,   # Mask R-CNN, expects [0, 1] float input
    class_model=cls_model,     # EfficientNet, expects ImageNet-normalised input
    img_size=224,
    score_threshold=0.5,
    jaccard_threshold=0.65,
)

image  = torch.rand(3, 512, 512)   # (3, H, W) float in [0, 1]
result = pipeline.predict(image)

print(result.class_label)          # predicted class index
print(result.class_probabilities)  # softmax probabilities, shape (C,)
print(result.detection_failed)     # True if Mask R-CNN found no lesion
```

---

## References

Carion, N. et al. End-to-End Object Detection with Transformers. ECCV 2020.
arXiv:2005.12872.

Codella, N. et al. Skin Lesion Analysis Toward Melanoma Detection 2018.
arXiv:1902.03368, 2019.

He, K. et al. Mask R-CNN. ICCV 2017. arXiv:1703.06870.

Lin, T.-Y. et al. Focal Loss for Dense Object Detection. ICCV 2017.
arXiv:1708.02002.

Tan, M. and Le, Q. EfficientNet: Rethinking Model Scaling for Convolutional
Neural Networks. ICML 2019. arXiv:1905.11946.

Tschandl, P., Rosendahl, C., and Kittler, H. The HAM10000 Dataset.
Scientific Data, 2018.

Zhu, X. et al. Deformable DETR: Deformable Transformers for End-to-End
Object Detection. ICLR 2021. arXiv:2010.04159.
