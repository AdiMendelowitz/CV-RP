# Medical Imaging: ISIC 2018 Skin Lesion Dataset

**Reference:** Codella et al., "Skin Lesion Analysis Toward Melanoma Detection 2018:
A Challenge Hosted by the International Skin Imaging Collaboration (ISIC),"
arXiv:1902.03368, 2019.

**Underlying dataset:** Tschandl et al., "The HAM10000 Dataset, a Large Collection
of Multi-Source Dermatoscopic Images of Common Pigmented Skin Lesions,"
Scientific Data, 2018. https://doi.org/10.1038/sdata.2018.161

---

## Challenge Overview

The ISIC 2018 challenge was hosted at MICCAI 2018 in Granada, Spain. It comprised
three tasks: lesion segmentation (Task 1), lesion attribute detection (Task 2), and
disease classification (Task 3). This project focuses on Tasks 1 and 3. Over 900
users registered for data download, with 115 and 159 teams submitting to the
segmentation and classification tasks respectively.

---

## Task 1: Lesion Segmentation

**Dataset:** `tschandl/isic2018-challenge-task1-data-segmentation` (Kaggle)

**Dataset size:**
- Training: 2,594 dermoscopic images with binary segmentation masks
- Validation: 100 images
- Test: 1,000 images

**Image resolution:** 600 x 450 pixels (native); resizing to 256x256 or 512x512
is standard practice in published work.

**Evaluation metric:** Thresholded Jaccard index. Standard Jaccard (intersection
over union) is computed per image; any prediction falling below a threshold of
T = 0.65 is counted as zero. This threshold was derived from inter-observer
variability measurements on the 2016 challenge data, where the lowest pairwise
annotator agreement was 0.743. The thresholded variant penalises gross segmentation
failures more directly than mean Jaccard alone.

The top submission in 2018 achieved a Thresholded Jaccard of 0.802. Even the best
algorithms failed on more than 10% of images, with failure rates highest for
seborrheic keratosis lesions.

**Implementation approach:** Mask R-CNN
(`torchvision.models.detection.maskrcnn_resnet50_fpn`, ImageNet pretrained) with
the box and mask predictor heads replaced for 2 classes (background + lesion).
Bounding boxes are derived from the mask bounding rectangle at dataset load time.
The model produces instance segmentation masks directly; predicted masks are
thresholded at 0.5 for binary evaluation. Validation is scored with Thresholded
Jaccard (T = 0.65). Best checkpoint is saved by validation Jaccard.

---

## Task 3: Disease Classification

**Dataset:** `kmader/skin-cancer-mnist-ham10000` (Kaggle)

**Dataset size:**
- Training: 10,015 images with ground-truth class labels
- Validation: 193 images
- Test: 1,512 images (split into internal: 1,196 and external: 316)

The external partition was drawn from institutions not represented in training
(Turkey, New Zealand, Sweden, Argentina), specifically to test generalisation
beyond the source distribution.

**Image resolution:** 600 x 450 pixels native. Resizing to 224x224 is standard
for EfficientNet and ResNet backbones pretrained on ImageNet.

**Classes (7):**

| Code  | Full Name                                                         |
|-------|-------------------------------------------------------------------|
| NV    | Melanocytic nevus                                                 |
| MEL   | Melanoma                                                          |
| BKL   | Benign keratosis (solar lentigo / seborrheic keratosis / LPLK)   |
| BCC   | Basal cell carcinoma                                              |
| AKIEC | Actinic keratosis / Bowen's disease (intraepithelial carcinoma)  |
| VASC  | Vascular lesion                                                   |
| DF    | Dermatofibroma                                                    |

**Class distribution (training set):**

| Class | Count | Proportion |
|-------|-------|------------|
| NV    | 6,705 | 66.9%      |
| MEL   | 1,113 | 11.1%      |
| BKL   | 1,099 | 11.0%      |
| BCC   |   514 |  5.1%      |
| AKIEC |   327 |  3.3%      |
| VASC  |   142 |  1.4%      |
| DF    |   115 |  1.1%      |
| **Total** | **10,015** | |

NV constitutes 67% of training images. DF and VASC together account for roughly
2.5%. This is a heavily long-tailed distribution with an NV/DF imbalance ratio
of approximately 58:1.

**Evaluation metric:** Balanced accuracy (mean per-class recall). Standard
accuracy is not used because it would be dominated by NV performance. Balanced
accuracy weights each class equally regardless of sample count, making it more
meaningful for clinical deployment where disease prevalence in the real world
differs from dataset prevalence.

**Implementation approach:** EfficientNet-B0 (`timm`, ImageNet pretrained) with
the classification head replaced for 7 classes. Loss is weighted cross-entropy
with per-class weights computed as `total / (num_classes * class_count)`. Best
checkpoint is saved by validation balanced accuracy.

---

## Class Imbalance: Handling Strategy

**Loss-based:** Weighted cross-entropy assigns per-class weights inversely
proportional to frequency. Focal loss (Lin et al., ICCV 2017) additionally
down-weights easy negatives, focusing gradient updates on hard minority samples.

**Sampling-based:** Oversampling minority classes via augmentation (rotation,
flipping, colour jitter) or undersampling NV. Aggressive upsampling of very
small classes (DF: 115 samples) risks overfitting; controlled augmentation to
3-5x original size is more common in published work.

**Metric-aligned training:** Since the evaluation metric is balanced accuracy,
using weighted loss or balanced batch sampling is important. A model that achieves
95% accuracy by predicting NV for every input scores 1/7 balanced accuracy.

**Weight formula used in this project:**

```
weight_c = total_samples / (num_classes * count_c)
```

This produces weights of approximately 2.1 for NV and 122 for DF, passed directly
to `torch.nn.CrossEntropyLoss(weight=...)`.

---

## Preprocessing Notes

- Native resolution is 600 x 450. Resize to 224x224 for EfficientNet-B0.
- Dermoscopic images often contain hair artefacts and vignetting at image borders.
  Hair removal is an optional preprocessing step; the gain is marginal with modern
  augmentation pipelines.
- Standard normalisation: ImageNet mean and std are appropriate when using
  pretrained backbones
  (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]).
- Ground truth for segmentation (Task 1) is stored as binary PNG masks, where
  255 indicates lesion and 0 indicates background.
- Labels for classification (Task 3) are stored in `HAM10000_metadata.csv` with
  a `dx` column containing the lowercase class code (nv, mel, bkl, bcc, akiec,
  vasc, df).

---

## Key Technical Considerations

**Domain shift:** The challenge explicitly tested generalisation by holding out
images from institutions unseen during training. Algorithms with equal internal
test performance showed substantially different external test performance,
highlighting the risk of overfitting to acquisition device or institution style.
This motivates strong augmentation policies and colour normalisation.

**Label quality:** Task 3 ground truth was established by histopathology,
reflectance confocal microscopy, or expert consensus -- a higher standard than
crowdsourced labels. Segmentation masks for Task 1 reflect single-annotator
labels, and inter-observer Jaccard among experts averages approximately 0.786.

**Clinical context:** Melanoma early detection is the primary motivation. The
5-year survival rate for melanoma exceeds 99% when caught early, dropping to
23% at late stages. This clinical asymmetry justifies treating false negatives
for MEL as more costly than false positives, which can be encoded explicitly in
the loss function rather than relying on balanced accuracy alone.