# Medical Imaging: ISIC 2018 Skin Lesion Dataset

**Challenge reference:** Codella et al., “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC),” arXiv:1902.03368, 2019.  
**Underlying dataset:** Tschandl et al., “The HAM10000 Dataset, a Large Collection of Multi‑Source Dermatoscopic Images of Common Pigmented Skin Lesions,” *Scientific Data*, 2018.

---

## Challenge overview

The ISIC 2018 challenge (MICCAI 2018, Granada) comprised three tasks:

- Task 1: Lesion segmentation.  
- Task 2: Lesion attribute detection.  
- Task 3: Disease classification.

This project implements **Task 1 (segmentation)** and **Task 3 (classification)**. Over 900 users registered for data download; 115 teams submitted for segmentation and 159 for classification.

---

## Task 1: Lesion segmentation

**Dataset:** `tschandl/isic2018-challenge-task1-data-segmentation` (Kaggle wrap of ISIC 2018 Task 1).

### Dataset and splits

Official ISIC 2018 Task 1 sizes:

- Training: 2,594 dermoscopic images with corresponding binary masks.  
- Validation: 100 images (challenge validation; not used in this project).  
- Test: 1,000 images (held out by organisers; not used here).

Project split:

- Random 80/20 split of the 2,594 training images → ≈2,075 train / 519 validation.  
- No official training/validation split is provided; this internal split is standard practice.

Image resolution:

- Native images: 600×450 pixels.  
- Resized to **512×512** for this project; 256×256 and 512×512 are both common in published work.

### Metric: thresholded Jaccard

The official ISIC 2018 segmentation metric is **thresholded Jaccard index** `T = 0.65`:

1. Compute standard Jaccard (IoU) per image between prediction and ground truth.  
2. For each image, if IoU < 0.65, set that image’s score to 0.  
3. Average over the dataset.

The 0.65 threshold derives from inter‑observer variability on ISIC 2016: the lowest pairwise expert IoU was ≈0.743. Thresholding penalises gross failures (e.g. missing the lesion entirely) much more strongly than mean IoU alone.

Top 2018 submission:

- Best thresholded Jaccard ≈0.802; even top methods failed on >10% of images.  
- Failure rates were highest for images with artefacts such as rulers, ink marks, and occluding hair.

### Implementation: Mask R‑CNN

Model:

- **Mask R‑CNN** with ResNet‑50‑FPN backbone pretrained on COCO.  
- Implementation: `torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)`.  
- Box and mask heads replaced to output 2 classes: background and lesion.

Targets:

- Bounding boxes derived from the tight bounding rectangle of each ground‑truth mask at load time, matching Mask R‑CNN’s expected target format.  
- Predicted soft masks binarised at 0.5; validation scored with thresholded Jaccard (T = 0.65).

Training:

- COCO‑pretrained backbone + FPN frozen or fine‑tuned; best checkpoint selected by validation thresholded Jaccard.

Result:

- Best validation **thresholded Jaccard = 0.7822** (Run 5, epoch 16), close to the original challenge SOTA given differences in data splits and model variants.

---

## Task 3: Disease classification

**Dataset:** `kmader/skin-cancer-mnist-ham10000` (Kaggle mirror of HAM10000).

### Dataset and splits

Official HAM10000 / ISIC Task 3 sizes:

- Training: 10,015 images with ground‑truth labels.  
- Validation: 193 images (challenge validation set; not used here).  
- Test: 1,512 images, split into:  
  - Internal test: 1,196 images from training institutions.  
  - External test: 316 images from new institutions (Turkey, New Zealand, Sweden, Argentina) to test cross‑site generalisation.

Project split:

- **Lesion‑level** stratified 80/20 split of the 10,015 training images → 8,020 train / 1,995 validation.  
- Splitting by `lesion_id` (not `image_id`) prevents the same lesion (multiple crops) from appearing in both train and validation, which would inflate validation metrics.

Image resolution:

- Native: 600×450 pixels.  
- Resized to **224×224**, the canonical input size for EfficientNet‑B0 pretrained on ImageNet.

### Classes and imbalance

Seven diagnostic categories:

| Code | Full name                                                                 |
|------|---------------------------------------------------------------------------|
| NV   | Melanocytic nevus                                                         |
| MEL  | Melanoma                                                                  |
| BKL  | Benign keratosis (solar lentigo / seborrheic keratosis / LPLK)           |
| BCC  | Basal cell carcinoma                                                      |
| AKIEC| Actinic keratosis / Bowen’s disease (intraepithelial carcinoma)          |
| VASC | Vascular lesion                                                           |
| DF   | Dermatofibroma                                                            |

Full 10,015‑image distribution:

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

NV forms ~67% of the dataset; DF and VASC together are ~2.5%. This induces a long‑tailed distribution with NV:DF ≈58:1 overall, ≈53:1 in the project training split.

### Metric: balanced accuracy

Following Codella et al. and the challenge design, the primary metric is **balanced accuracy** (mean per‑class recall):

- Standard accuracy is dominated by NV; a “predict NV always” classifier reaches ~67% accuracy but only 1/7 ≈14.3% balanced accuracy.  
- Balanced accuracy treats each class equally, which is more clinically meaningful given severe class imbalance.

### Implementation: EfficientNet‑B0

Model:

- EfficientNet‑B0 from `timm`, pretrained on ImageNet.  
- Final classification head replaced with a linear layer producing 7 logits (one per class).

Preprocessing:

- Resize to 224×224.  
- ImageNet normalisation (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]).  
- Labels loaded from `HAM10000_metadata.csv` with `dx` codes (nv, mel, bkl, bcc, akiec, vasc, df).

Loss:

- **Weighted cross‑entropy** with per‑class weights  
  `w_c = total\_samplesnum\_classes * count_c`, implemented via `torch.nn.CrossEntropyLoss(weight=...)`.  
- This rebalances expected loss contributions so all classes contribute equally in expectation.

Result:

- Best validation balanced accuracy **0.7457** (Run 5, epoch 22).  
- A naive “always NV” classifier would be at ~14.3% balanced accuracy, so this represents substantial improvement in minority‑class performance.

---

## Class imbalance strategy

### Training split distribution (8,020 images)

From `HAM10000_metadata.csv` after lesion‑level 80/20 split:

| Class | Count | Proportion | Weight `w_c` |
|-------|-------|------------|----------------|
| nv    | 5,369 | 66.9%      | 0.213          |
| mel   |   891 | 11.1%      | 1.280          |
| bkl   |   876 | 10.9%      | 1.300          |
| bcc   |   411 |  5.1%      | 2.767          |
| akiec |   262 |  3.3%      | 4.323          |
| vasc  |   109 |  1.4%      | 10.416         |
| df    |   102 |  1.3%      | 13.169         |

A single DF sample contributes ≈13.169 / 0.213 ≈ 62× more loss than a single NV sample under this weighting, equalising expected per‑class contributions.

### Cross-entropy vs weighted CE vs focal loss

**Standard cross‑entropy** for true class c with predicted probability `p_c`:

```text
CE(p_c) = -log(p_c)
```

- No accounting for class frequency; majority classes dominate gradients simply by being more frequent.

**Weighted cross‑entropy**:

```text
WCE(p_c) = - w_c · log(p_c)
w_c = total_samples / (num_classes · count_c)
```

- Addresses **frequency imbalance** by up‑weighting rare classes so that each class contributes roughly equally in expectation.  
- Does *not* address **difficulty imbalance** (easy vs hard samples within a class).

**Focal loss** (Lin et al. 2017):

```text
FL(p_c) = - (1 - p_c)^γ · log(p_c)
```

- γ > 0 down‑weights well‑classified samples (p_c near 1) and focuses on hard or misclassified examples (p_c near 0).  
- The α‑balanced variant combines focal modulation with class weighting:  
  `FL(p_c) = - alpha_c (1 - p_c)^γ log p_c`.  
- Addresses both **frequency (α_c)** and **difficulty (γ)** imbalance, but introduces additional hyperparameters.

### Chosen approach and rationale

This project uses **weighted cross‑entropy** with `w_c = (N)/(7 * count)_c`.

Rationale:

1. **Overfitting dominates:** Initial runs showed training loss collapsing while validation loss plateaued (strong train/val gap), indicating memorisation; this is best addressed with augmentation, dropout, and weight decay, not a more complex loss.  
2. **Hyperparameter budget:** Focal loss adds γ (and often α) which require sweeps; given run‑to‑run variance in balanced accuracy, there is limited budget for reliable γ tuning.  
3. **Imbalance handled sufficiently:** For NV/DF ≈53:1, weighting alone yields ~60× weight ratio, which largely corrects frequency imbalance; remaining errors are due to MEL/NV visual similarity, which is fundamentally a representation problem.  
4. **Focal loss as follow‑up:** If balanced accuracy plateaus <0.75 even after stronger regularisation and augmentation, an α‑balanced focal loss with γ ∈ {0.5,1,2} would be a natural next experiment.

---

## Preprocessing notes

**Task 1 (segmentation):**

- Resize from 600×450 to 512×512.  
- Ground‑truth masks: binary PNGs (e.g. 0/255); any pixel >127 treated as lesion.  
- Inputs scaled to [0,1]; ImageNet normalisation is not required for torchvision’s COCO‑pretrained Mask R‑CNN.

**Task 3 (classification):**

- Resize from 600×450 to 224×224.  
- Apply ImageNet normalisation (mean/std as above) to match EfficientNet‑B0’s pretrained weights.  
- Optional hair removal and illumination correction are possible, but with modern augmentation and regularisation they often yield only marginal gains.

---

## Key technical and clinical considerations

- **Domain shift:** ISIC’s external test set (institutions unseen during training) reveals large drops in performance for some methods, underlining the importance of strong augmentation and potentially domain‑invariant representations.  
- **Label quality:** HAM10000 labels are based on histopathology, reflectance confocal microscopy, or expert consensus, giving relatively high label reliability; segmentation masks, however, reflect single‑annotator boundaries and inter‑observer Jaccard is ≈0.78, which is itself an upper‑bound reference for algorithm performance.  
- **Clinical risk asymmetry:** Early melanoma detection has a massive impact on survival; late‑stage melanoma outcomes are far worse than benign false positives. In a deployment setting, class weights or decision thresholds could be further skewed to penalise MEL false negatives more heavily than other errors.

---

## References

- Codella, N. et al. “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the ISIC.” arXiv:1902.03368, 2019.  
- Tschandl, P., Rosendahl, C., Kittler, H. “The HAM10000 Dataset, a Large Collection of Multi‑Source Dermatoscopic Images of Common Pigmented Skin Lesions.” *Scientific Data*, 2018.  
- Lin, T.-Y. et al. “Focal Loss for Dense Object Detection.” ICCV 2017, arXiv:1708.02002.  
- Tan, M., Le, Q. “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.” ICML 2019, arXiv:1905.11946.  
- He, K. et al. “Mask R‑CNN.” ICCV 2017, arXiv:1703.06870.
