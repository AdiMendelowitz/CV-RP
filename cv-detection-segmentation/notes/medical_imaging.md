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

### Dataset Distribution (ISIC 2018 Task 3)

From direct dataset inspection on `HAM10000_metadata.csv` (training split, 8020 images after
lesion-level 80/20 split):

| Class | Count | Proportion | Weight (total / 7 * count) |
|-------|-------|------------|---------------------------|
| nv    | 5369  | 66.9%      | 0.213                     |
| mel   | 891   | 11.1%      | 1.280                     |
| bkl   | 876   | 10.9%      | 1.300                     |
| bcc   | 411   |  5.1%      | 2.767                     |
| akiec | 262   |  3.3%      | 4.323                     |
| vasc  | 109   |  1.4%      | 10.416                    |
| df    | 102   |  1.3%      | 13.169                    |

The NV/DF imbalance ratio is approximately 53:1. Because NV constitutes two-thirds
of the training set, a naive classifier that predicts NV for every input achieves
~67% accuracy but 1/7 (14.3%) balanced accuracy -- the primary evaluation metric.

---

### Loss Function Analysis

#### Cross-Entropy (Baseline)

Standard cross-entropy for a sample with true class c and predicted probability p_c:

```
CE(p_c) = -log(p_c)
```

The loss is the same regardless of whether the model was confident or uncertain. A
highly confident wrong prediction (p_c = 0.01) and a marginally wrong prediction
(p_c = 0.45) receive different loss magnitudes, but the gradient from the dominant
NV class still dominates parameter updates in proportion to class frequency. With
67% of batches containing NV samples, the backbone learns NV-discriminative features
disproportionately.

#### Weighted Cross-Entropy

Per-class weights w_c are applied to scale the per-sample loss:

```
WCE(p_c) = -w_c * log(p_c)

where w_c = total_samples / (num_classes * count_c)
```

This re-balances the expected loss contribution of each class to be equal across
training. For HAM10000 with the weights above, a single DF sample contributes
approximately 62x more loss signal than a single NV sample (13.169 / 0.213).

This strategy is implemented in this project via `torch.nn.CrossEntropyLoss(weight=...)`.

**What it addresses:** Frequency imbalance. Each class is equally represented in
the aggregate gradient, regardless of how many samples it has.

**What it does not address:** Difficulty imbalance. Within the NV class, there are
easy NV samples (unambiguous texture, standard acquisition) and hard NV samples
(atypical presentation, acquisition artefacts). Weighted CE treats all NV samples
identically -- it only scales by class, not by sample confidence.

#### Focal Loss (Lin et al., ICCV 2017, arXiv:1708.02002)

Focal loss was introduced for one-stage object detection (RetinaNet), where the
extreme foreground/background imbalance (up to 100,000:1) caused standard cross-
entropy training to degenerate -- the model learned to predict background with high
confidence, generating near-zero loss for each background sample, but the sheer
volume of easy background examples dominated the gradient.

The focal loss adds a modulating factor (1 - p_c)^gamma to cross-entropy:

```
FL(p_c) = -(1 - p_c)^gamma * log(p_c)
```

When gamma = 0, this reduces to standard cross-entropy. As gamma increases:

- Well-classified samples (p_c close to 1): (1 - p_c)^gamma approaches 0,
  suppressing their contribution to the gradient nearly to zero.
- Misclassified or uncertain samples (p_c close to 0): (1 - p_c)^gamma
  approaches 1, preserving the full cross-entropy loss.

Lin et al. found gamma = 2 to work well in practice. A sample predicted with
p_c = 0.9 receives a loss weight of (1 - 0.9)^2 = 0.01 relative to standard CE,
while a sample predicted with p_c = 0.1 retains (1 - 0.1)^2 = 0.81 of its CE loss.

The alpha-balanced variant combines focal weighting with class-frequency weighting:

```
FL(p_c) = -alpha_c * (1 - p_c)^gamma * log(p_c)
```

where alpha_c plays the same role as w_c in weighted cross-entropy.

**What it addresses:** Both frequency imbalance (via alpha) and difficulty
imbalance (via gamma). Easy majority-class samples are suppressed dynamically
during training, not just scaled statically at the start.

**What it does not address:** It introduces two hyperparameters (alpha, gamma)
that require tuning. Alpha is the same as the class weight in WCE and can be set
by frequency. Gamma must be swept (typical range: 0.5 to 5.0). Lin et al. report
gamma = 2 as robust across datasets, but this was demonstrated on object detection
tasks, not medical image classification.

---

### Comparison: When to Use Each

| Criterion | Weighted CE | Focal Loss |
|-----------|-------------|------------|
| Imbalance source | Frequency only | Frequency + difficulty |
| Hyperparameters | w_c (set by formula) | w_c + gamma (requires tuning) |
| Implementation | `nn.CrossEntropyLoss(weight=...)` | Custom or `torchvision.ops.sigmoid_focal_loss` |
| Stability | High -- loss scale is predictable | Lower -- gamma interacts with LR schedule |
| Best suited for | Moderate imbalance, stable training | Severe imbalance, hard negatives present |
| Typical imbalance ratio | Up to ~20:1 | 20:1 and above, or detection tasks |

For HAM10000 specifically, the NV/DF ratio of 53:1 and MEL/NV visual similarity
make focal loss a reasonable candidate. However, published results on HAM10000
(including the Stanford study using EfficientNet) show that weighted cross-entropy
with strong augmentation and dropout regularisation closes most of the gap with
focal loss. The primary bottleneck in HAM10000 classification is overfitting to
the training distribution, not the loss function's ability to handle hard examples.

---

### Chosen Strategy and Justification

This project uses weighted cross-entropy (`w_c = total / (7 * count_c)`).

**Justification:**

1. Overfitting is the dominant failure mode, not hard-example mining. Run 1
   (no regularisation) showed train loss collapsing to 0.02 while val loss
   plateaued at 0.78, a 40x train/val ratio. No loss function resolves
   memorisation -- that requires dropout, weight decay, and augmentation.

2. Focal loss introduces a second hyperparameter (gamma) that requires its own
   sweep. With only 20 training epochs per run and meaningful variance between
   identical runs (0.7441 vs 0.7233), the experimental budget does not support
   a reliable gamma sweep.

3. Weighted CE is sufficient for the imbalance ratios present. The 53:1 NV/DF
   ratio is severe, but the effective weight correction (w_DF / w_NV = 61.8x)
   brings the expected per-class loss contribution to parity. The remaining
   challenge is MEL/NV visual confusion, which is a feature representation
   problem (addressed by backbone fine-tuning) rather than a loss weighting problem.

4. Focal loss is the natural next experiment if balanced accuracy plateaus below
   0.75 after augmentation and regularisation improvements. The implementation
   would use the alpha-balanced variant with alpha_c set by the same frequency
   formula and a gamma sweep over {0.5, 1.0, 2.0}.

---
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