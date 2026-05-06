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
disease classification (Task 3). This project implements Tasks 1 and 3. Over 900
users registered for data download, with 115 and 159 teams submitting to the
segmentation and classification tasks respectively.

---

## Task 1: Lesion Segmentation

**Dataset:** `tschandl/isic2018-challenge-task1-data-segmentation` (Kaggle)

**Official dataset size:**
- Training: 2,594 dermoscopic images with binary segmentation masks
- Validation: 100 images (official; not used in this project)
- Test: 1,000 images (official; not used in this project)

**Project split:** 80/20 random split of the 2,594 training images, yielding
approximately 2,075 training images and 519 validation images. No official
validation split is provided by the challenge organisers.

**Image resolution:** 600 x 450 pixels (native). Resized to 512 x 512 in this
project; 256 x 256 and 512 x 512 are both standard in published work.

**Evaluation metric:** Thresholded Jaccard index (T = 0.65). Standard Jaccard
(intersection over union) is computed per image; values below T = 0.65 are set
to zero before averaging across the validation set. This threshold was derived
from inter-observer variability measurements on the 2016 challenge data, where
the lowest pairwise annotator agreement was 0.743. The thresholded variant
penalises gross segmentation failures more directly than mean Jaccard alone.

The top submission in 2018 achieved a Thresholded Jaccard of 0.802. Even the
best algorithms failed on more than 10% of images, with failure rates highest
for images containing clinical measurement artefacts such as rulers and ink marks
(Codella et al., 2019).

**Implementation approach:** Mask R-CNN with a ResNet-50-FPN backbone pretrained
on COCO (`torchvision.models.detection.maskrcnn_resnet50_fpn`,
`weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT`). Both the box predictor head
and the mask predictor head were replaced for 2 output classes (background and
lesion). The backbone and FPN weights were retained from COCO pretraining.
Bounding boxes are derived from the tight bounding rectangle of each ground-truth
mask at dataset load time, as required by the Mask R-CNN target format. Predicted
soft masks are binarised at 0.5; validation is scored with Thresholded Jaccard
(T = 0.65). Best checkpoint is saved by validation Jaccard.

**Result:** Best validation Thresholded Jaccard of 0.7822 (Run 5, epoch 16).

---

## Task 3: Disease Classification

**Dataset:** `kmader/skin-cancer-mnist-ham10000` (Kaggle)

**Official dataset size:**
- Training: 10,015 images with ground-truth class labels
- Validation: 193 images (official; not used in this project)
- Test: 1,512 images split into internal (1,196) and external (316) partitions

The external test partition was drawn from institutions not represented in
training (Turkey, New Zealand, Sweden, Argentina), specifically to test
generalisation beyond the source distribution.

**Project split:** Lesion-level stratified 80/20 split of the 10,015 training
images, yielding 8,020 training images and 1,995 validation images. The split
is performed on `lesion_id`, not `image_id`, because the same lesion appears
multiple times in HAM10000 with different crops. Splitting on `image_id` would
leak lesion information into validation and inflate metrics.

**Image resolution:** 600 x 450 pixels native. Resized to 224 x 224, the
canonical input size for EfficientNet-B0 pretrained on ImageNet.

**Classes (7):**

| Code  | Full Name                                                                        |
|-------|----------------------------------------------------------------------------------|
| NV    | Melanocytic nevus                                                                |
| MEL   | Melanoma                                                                         |
| BKL   | Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis) |
| BCC   | Basal cell carcinoma                                                             |
| AKIEC | Actinic keratosis / Bowen's disease (intraepithelial carcinoma)                 |
| VASC  | Vascular lesion                                                                  |
| DF    | Dermatofibroma                                                                   |

**Class distribution (full dataset, 10,015 images):**

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

NV constitutes 67% of the full dataset. DF and VASC together account for roughly
2.5%. This is a heavily long-tailed distribution. The NV/DF ratio in the full
dataset is 58:1; in the project training split (8,020 images) it is approximately
53:1.

**Evaluation metric:** Balanced accuracy (mean per-class recall), per Codella et
al. (2019). Standard accuracy is not used because it is dominated by NV
performance -- a model that predicts NV for every input achieves 67% accuracy
but 1/7 (14.3%) balanced accuracy.

**Implementation approach:** EfficientNet-B0 (`timm`, ImageNet pretrained) with
the classification head replaced for 7 output classes via `timm.create_model`.
Loss is weighted cross-entropy with per-class weights `weight_c = total /
(num_classes * count_c)`. Best checkpoint is saved by validation balanced
accuracy.

**Result:** Best validation balanced accuracy of 0.7457 (Run 5, epoch 22).

---

## Class Imbalance: Handling Strategy

### Dataset Distribution (Project Training Split, 8,020 images)

From direct inspection of `HAM10000_metadata.csv` after the lesion-level 80/20
split:

| Class | Count | Proportion | Weight (total / 7 * count) |
|-------|-------|------------|---------------------------|
| nv    | 5,369 | 66.9%      | 0.213                     |
| mel   |   891 | 11.1%      | 1.280                     |
| bkl   |   876 | 10.9%      | 1.300                     |
| bcc   |   411 |  5.1%      | 2.767                     |
| akiec |   262 |  3.3%      | 4.323                     |
| vasc  |   109 |  1.4%      | 10.416                    |
| df    |   102 |  1.3%      | 13.169                    |

The NV/DF imbalance ratio in the training split is approximately 53:1. A naive
classifier predicting NV for every input achieves ~67% accuracy but 1/7 (14.3%)
balanced accuracy -- the primary evaluation metric.

---

### Loss Function Analysis

#### Cross-Entropy (Baseline)

Standard cross-entropy for a sample with true class c and predicted probability
p_c:

```
CE(p_c) = -log(p_c)
```

The loss magnitude is determined solely by p_c, with no scaling by class
frequency. With 67% of batches containing NV samples, the backbone learns
NV-discriminative features disproportionately, and gradient updates are dominated
by the majority class regardless of whether individual NV predictions are correct.

#### Weighted Cross-Entropy

Per-class weights w_c scale the per-sample loss:

```
WCE(p_c) = -w_c * log(p_c)

where w_c = total_samples / (num_classes * count_c)
```

This re-balances the expected loss contribution of each class to be equal in
expectation across training. For the project training split, a single DF sample
contributes approximately 62x more loss signal than a single NV sample
(13.169 / 0.213).

Implemented via `torch.nn.CrossEntropyLoss(weight=...)`.

**What it addresses:** Frequency imbalance. Each class contributes equally to
the aggregate gradient in expectation, regardless of sample count.

**What it does not address:** Difficulty imbalance. Within a class, easy and
hard samples are treated identically -- the loss is scaled by class membership,
not by prediction confidence.

#### Focal Loss (Lin et al., ICCV 2017, arXiv:1708.02002)

Focal loss was introduced for one-stage object detection (RetinaNet), where the
extreme foreground/background imbalance (up to 100,000:1 in dense detection)
caused standard cross-entropy training to degenerate -- easy background examples
generated near-zero loss individually but dominated the gradient in aggregate.

The focal loss adds a modulating factor (1 - p_c)^gamma to cross-entropy:

```
FL(p_c) = -(1 - p_c)^gamma * log(p_c)
```

When gamma = 0 this reduces to standard cross-entropy. As gamma increases:

- Well-classified samples (p_c close to 1): (1 - p_c)^gamma approaches 0,
  suppressing their loss contribution to near zero.
- Misclassified or uncertain samples (p_c close to 0): (1 - p_c)^gamma
  approaches 1, preserving the full cross-entropy loss.

Lin et al. found gamma = 2 to work well in practice. A sample predicted with
p_c = 0.9 receives a loss weight of (1 - 0.9)^2 = 0.01 relative to standard
CE; a sample predicted with p_c = 0.1 retains (1 - 0.1)^2 = 0.81 of its CE
loss.

The alpha-balanced variant combines focal weighting with class-frequency
weighting:

```
FL(p_c) = -alpha_c * (1 - p_c)^gamma * log(p_c)
```

where alpha_c plays the same role as w_c in weighted cross-entropy.

**What it addresses:** Both frequency imbalance (via alpha) and difficulty
imbalance (via gamma). Easy majority-class samples are suppressed dynamically
during training rather than scaled statically.

**What it does not address:** It introduces two hyperparameters (alpha, gamma)
requiring tuning. Alpha can be set by the same frequency formula as weighted CE.
Gamma requires a sweep; Lin et al. report gamma = 2 as robust across datasets,
but this was demonstrated on object detection, not medical image classification.

---

### Comparison: When to Use Each

| Criterion | Weighted CE | Focal Loss |
|-----------|-------------|------------|
| Imbalance source | Frequency only | Frequency + difficulty |
| Hyperparameters | w_c (set by formula) | w_c + gamma (requires sweep) |
| PyTorch implementation | `nn.CrossEntropyLoss(weight=...)` | Custom or `torchvision.ops.sigmoid_focal_loss` |
| Training stability | High -- loss scale is predictable | Lower -- gamma interacts with LR schedule |
| Best suited for | Moderate imbalance, stable training | Severe imbalance, overconfident predictions |
| Typical imbalance ratio | Up to ~20:1 | 20:1 and above |

For HAM10000, the NV/DF ratio of 53:1 and MEL/NV visual similarity make focal
loss a reasonable candidate. However, published results on HAM10000 show that
weighted cross-entropy with strong augmentation and dropout regularisation closes
most of the gap with focal loss (Butskova, 2020; Lin et al., arXiv:2009.05977).
The primary bottleneck in HAM10000 classification is overfitting to the training
distribution, not the loss function's ability to handle hard examples.

---

### Chosen Strategy and Justification

This project uses weighted cross-entropy (`w_c = total / (7 * count_c)`).

**Justification:**

1. Overfitting is the dominant failure mode. Run 1 (no regularisation) showed
   train loss collapsing to 0.02 while val loss plateaued at 0.78 -- a 40:1
   train/val ratio. No loss function resolves memorisation; that requires
   dropout, weight decay, and augmentation.

2. Focal loss introduces gamma as a second hyperparameter requiring its own
   sweep. With meaningful variance between identical runs (0.7441 vs 0.7233 on
   the same configuration), the experimental budget does not support a reliable
   gamma sweep on a single Kaggle session.

3. Weighted CE is sufficient for the imbalance ratios present. The 53:1 NV/DF
   ratio is severe, but the weight correction (w_DF / w_NV = 61.8x) brings the
   expected per-class loss contribution to parity. The remaining challenge is
   MEL/NV visual confusion -- a feature representation problem addressed by
   backbone fine-tuning, not a loss weighting problem.

4. Focal loss is the natural next experiment if balanced accuracy plateaus below
   0.75 after augmentation and regularisation improvements. The implementation
   would use the alpha-balanced variant with alpha_c set by the frequency formula
   and a gamma sweep over {0.5, 1.0, 2.0}.

---

## Preprocessing Notes

**Task 1 (Segmentation):**
- Native resolution is 600 x 450. Resized to 512 x 512 in this project.
- Ground truth is stored as binary PNG masks: 255 indicates lesion, 0 indicates
  background. Pixels with value > 127 are treated as foreground.
- No colour normalisation applied. COCO-pretrained Mask R-CNN expects pixel
  values in [0, 1] without ImageNet normalisation.

**Task 3 (Classification):**
- Native resolution is 600 x 450. Resized to 224 x 224 for EfficientNet-B0.
- Dermoscopic images often contain hair artefacts and vignetting at image
  borders. Hair removal is an optional preprocessing step; the gain is marginal
  with modern augmentation pipelines.
- ImageNet normalisation applied (mean = [0.485, 0.456, 0.406],
  std = [0.229, 0.224, 0.225]), matching the timm EfficientNet-B0 pretrained
  weights.
- Labels are stored in `HAM10000_metadata.csv` with a `dx` column containing
  the lowercase class code (nv, mel, bkl, bcc, akiec, vasc, df).

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
labels; inter-observer Jaccard among experts averages approximately 0.786
(Codella et al., 2019).

**Clinical context:** Melanoma early detection is the primary motivation. The
5-year survival rate for melanoma exceeds 99% when caught early, dropping to
23% at late stages. This clinical asymmetry justifies treating false negatives
for MEL as more costly than false positives, which can be encoded explicitly in
the loss function rather than relying on balanced accuracy alone.

---

## References

Codella, N. et al. "Skin Lesion Analysis Toward Melanoma Detection 2018:
A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)."
arXiv:1902.03368, 2019.

Tschandl, P., Rosendahl, C., Kittler, H. "The HAM10000 Dataset, a Large
Collection of Multi-Source Dermatoscopic Images of Common Pigmented Skin
Lesions." Scientific Data, 2018. https://doi.org/10.1038/sdata.2018.161

Lin, T.-Y. et al. "Focal Loss for Dense Object Detection." ICCV, 2017.
arXiv:1708.02002.

Tan, M., Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional
Neural Networks." ICML, 2019. arXiv:1905.11946.

He, K. et al. "Mask R-CNN." ICCV, 2017. arXiv:1703.06870.