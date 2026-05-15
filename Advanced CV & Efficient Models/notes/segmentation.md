# Image Segmentation: U-Net and Mask R-CNN

---

## Semantic vs. Instance Segmentation

These two tasks are often conflated but answer fundamentally different questions.[file:127][web:51]

**Semantic segmentation** assigns a class label to every pixel in the image. All pixels belonging to the same category receive the same label, regardless of whether they belong to different object instances.[file:127][web:51] For example, a street scene would have all car pixels labeled “car”, all pedestrians labeled “person”, etc., with no distinction between individual cars.

**Instance segmentation** detects each individual object and produces a pixel-precise mask for every instance.[file:127][web:51] In the same street scene, each car and each pedestrian would have its own mask (car 1, car 2, pedestrian 1, pedestrian 2, …). Instance segmentation combines object detection (which object, where) with segmentation (which pixels).[file:127][web:51]

Semantic segmentation is sufficient when only class regions matter (e.g., road vs sidewalk vs sky). Instance segmentation is required when objects must be counted, tracked, or analysed individually (e.g., cell counting in microscopy, vehicle or pedestrian tracking in autonomous driving).[file:127][web:51]

---

## U-Net (Ronneberger, Fischer, Brox — MICCAI 2015)

**Reference:** Ronneberger O., Fischer P., Brox T. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” MICCAI 2015, LNCS 9351.[file:127][web:31]

### Motivation

Before U-Net, a common approach to pixel-wise prediction was sliding-window classification: a patch around each pixel is fed through a CNN to predict that pixel’s label.[file:127][web:31] This is:

- Slow: the network runs once per pixel, with heavy overlap between neighbouring patches.  
- Limited by a trade-off: larger patches offer more context but blur localisation; smaller patches sharpen localisation but lose global context.[file:127][web:31]

U-Net resolves this trade-off using an encoder–decoder architecture with skip connections that provide both global context and fine spatial detail.[file:127][web:31]

### Architecture

U-Net is a symmetric encoder–decoder network.[file:127][web:31]

- **Encoder (contracting path):** progressively reduces spatial resolution and increases channels, building high-level semantic features.  
- **Decoder (expanding path):** progressively upsamples and uses skip connections to recover spatial detail.

**Encoder (contracting path).** Each stage applies:

- Two 3×3 *valid* (unpadded) convolutions with ReLU.  
- A 2×2 max-pooling with stride 2.  

Each 3×3 valid conv shrinks spatial dimensions by 2 (1 pixel per side). The number of feature channels doubles at each downsampling step.[file:127][web:31]

**Decoder (expanding path).** Each stage applies:

- A 2×2 transposed convolution (up-conv) that doubles spatial dimensions and halves channels.  
- Concatenation with the corresponding encoder feature map (cropped to match size).  
- Two 3×3 convs with ReLU.[file:127][web:31]

**Final layer.** A 1×1 convolution maps the last 64-channel feature map to the desired number of classes (e.g., 2 for foreground/background).[file:127][web:31]

**Standard configuration channel and size progression (original paper):**[file:127][web:31]

```text
Input:      1 x 572 x 572

Encoder:
  Block 1:  64 ch,  568 x 568  -> pool -> 64 ch,  284 x 284
  Block 2: 128 ch,  280 x 280  -> pool -> 128 ch, 140 x 140
  Block 3: 256 ch,  136 x 136  -> pool -> 256 ch,  68 x 68
  Block 4: 512 ch,   64 x 64   -> pool -> 512 ch,  32 x 32

Bottleneck:
  Block 5: 1024 ch,  28 x 28

Decoder:
  Block 6:  512 ch,  52 x 52   (after up-conv + concat + 2x conv)
  Block 7:  256 ch, 100 x 100
  Block 8:  128 ch, 196 x 196
  Block 9:   64 ch, 388 x 388

Output:       2 ch, 388 x 388  (1x1 conv, softmax)
```

The output (388×388) is smaller than the input (572×572) due to repeated valid convolutions.[file:127][web:31] The paper handles larger images with an “overlap-tile” strategy: process overlapping tiles and stitch predictions.

### ASCII Architecture Diagram

Your ASCII diagram matches the original U-Net structure (valid convs, cropping for skips):[file:127][web:31]

```text
Input: 1x572x572
        |
  [2x Conv3x3 + ReLU]  -> 64ch, 568x568
        |____________(skip 1: crop to 392x392)
  [MaxPool 2x2]          -> 64ch, 284x284
        |
  [2x Conv3x3 + ReLU]  -> 128ch, 280x280
        |____________(skip 2: crop to 200x200)
  [MaxPool 2x2]          -> 128ch, 140x140
        |
  [2x Conv3x3 + ReLU]  -> 256ch, 136x136
        |____________(skip 3: crop to 104x104)
  [MaxPool 2x2]          -> 256ch, 68x68
        |
  [2x Conv3x3 + ReLU]  -> 512ch, 64x64
        |____________(skip 4: crop to 56x56)
  [MaxPool 2x2]          -> 512ch, 32x32
        |
  [2x Conv3x3 + ReLU]  -> 1024ch, 28x28      <- bottleneck
        |
  [UpConv 2x2]           -> 512ch, 56x56
  [Concat skip 4]        -> 1024ch, 56x56
  [2x Conv3x3 + ReLU]  -> 512ch, 52x52
        |
  [UpConv 2x2]           -> 256ch, 104x104
  [Concat skip 3]        -> 512ch, 104x104
  [2x Conv3x3 + ReLU]  -> 256ch, 100x100
        |
  [UpConv 2x2]           -> 128ch, 200x200
  [Concat skip 2]        -> 256ch, 200x200
  [2x Conv3x3 + ReLU]  -> 128ch, 196x196
        |
  [UpConv 2x2]           -> 64ch, 392x392
  [Concat skip 1]        -> 128ch, 392x392
  [2x Conv3x3 + ReLU]  -> 64ch, 388x388
        |
  [Conv 1x1]             -> 2ch, 388x388     <- output segmentation map
```

### Skip Connections: U-Net vs ResNet

Your conceptual distinction is correct and useful:[file:127][web:31]

- **ResNet skip connections:**  
  Add input to output (element‑wise) to form residual blocks, primarily to combat vanishing gradients and ease optimisation.[web:63] They do not change the spatial resolution hierarchy; skips are within the same resolution level.

- **U-Net skip connections:**  
  Concatenate encoder and decoder feature maps at corresponding resolutions to restore fine-grained spatial information lost through pooling and valid convolutions.[file:127][web:31] This directly addresses an information bottleneck, not gradient flow.

In short: ResNet skips are about **optimisation**; U‑Net skips are about **restoring spatial detail**.[file:127][web:31][web:63]

### Loss Function

The original U-Net uses pixel-wise softmax followed by cross-entropy, weighted by a spatial weight map:[file:127][web:31]

```text
w(x) = w_class(x) + w0 * exp( - (d1(x) + d2(x))^2 / (2 * sigma^2) )
```

- \(w_{class}(x)\): class-balance term (higher for rare classes).  
- \(d_1(x)\): distance from pixel x to the nearest cell border.  
- \(d_2(x)\): distance to the second nearest cell border.  
- Recommended values: \(w_0 = 10\), \(\sigma = 5\) pixels (paper defaults).[file:127][web:31]

This emphasises boundaries between touching cells so the model learns to separate adjacent instances.

### Why It Trains Well on Small Datasets

U-Net was designed for biomedical segmentation with limited annotated data. Two key factors:[file:127][web:31]

- Strong inductive bias: the encoder–decoder with skips is well‑suited to segmentation, reducing the need for massive datasets.  
- Aggressive **elastic deformation augmentation**: random smooth displacement fields applied to training images/masks simulate plausible biological deformations, effectively enlarging the training set and improving robustness.[file:127][web:31]

---

## Mask R-CNN (He, Gkioxari, Dollár, Girshick — ICCV 2017)

**Reference:** He K., Gkioxari G., Dollár P., Girshick R. “Mask R-CNN.” ICCV 2017.[file:127][web:52]

### Problem Statement

Mask R-CNN addresses **instance segmentation**: detect each object instance and produce a binary mask for each one.[file:127][web:52] It extends Faster R-CNN—originally a two-stage object detector predicting classes and bounding boxes—by adding a parallel mask prediction branch.[web:52]

### Architecture Overview

Mask R-CNN has three key stages:[file:127][web:52]

1. **Backbone + FPN.**  
   - A ResNet-50/101 backbone extracts feature maps.  
   - A Feature Pyramid Network (FPN) builds a multi-scale feature pyramid, combining high-resolution and high-semantic-level features to handle objects of different sizes.[web:52]

2. **Region Proposal Network (RPN).**  
   - Same concept as in Faster R-CNN: slides over FPN maps, proposes candidate boxes (anchors refined into proposals) with objectness scores; low-score proposals are filtered via NMS.[web:51][web:52]

3. **Per-RoI heads (parallel branches).**  
   For each proposal, RoIAlign extracts fixed-size features from the appropriate FPN level, then three heads operate in parallel:[file:127][web:52]  
   - Classification head: predicts class.  
   - Box regression head: refines bounding box coordinates.  
   - Mask head: predicts a binary segmentation mask.

Total loss:[file:127][web:52]

```text
L = L_cls + L_box + L_mask
```

where \(L_{cls}\) and \(L_{box}\) are as in Faster R-CNN (cross-entropy and smooth L1), and \(L_{mask}\) is per-pixel binary cross-entropy over predicted masks.[web:52]

### RoIAlign: The Critical Contribution

**Issue with RoIPool.**  

RoIPool (used in Fast/Faster R-CNN) quantises RoI coordinates and bin boundaries to integers on the feature map grid.[file:127][web:52] This introduces misalignment between the proposal and pooled features. While acceptable for bounding box detection, this misalignment hurts pixel-level mask prediction.

**RoIAlign.**  

RoIAlign removes rounding:[file:127][web:52]

- RoI coordinates remain in floating-point on the feature map.  
- Each bin of the output grid (e.g. 7×7 or 14×14) is sampled at fixed fractional positions.  
- Each sample is computed via bilinear interpolation from neighbouring feature map pixels.  
- Sampled values are aggregated (max or average) per bin.

No coordinate rounding is performed. This significantly improves mask AP, especially at higher IoU thresholds.[file:127][web:52]

```text
RoIPool (old):                      RoIAlign (new):
  proposal: (x=12.7, y=8.3)           proposal: (x=12.7, y=8.3)
  -> round to (13, 8)                 kept as (12.7, 8.3)
  -> bin boundaries rounded           bin boundaries keep float coords
  -> discrete pixel lookups           4 samples per bin via bilinear interp
```

### Mask Head Design

The mask branch operates on a higher‑resolution RoI feature map than the box/class heads:[file:127][web:52]

- Input: 256 × 14 × 14 RoI feature map from RoIAlign (for the mask branch).  
- Four 3×3 conv layers (256 channels, ReLU).  
- One 2×2 transposed convolution (stride 2) to upsample to 28×28.  
- One 1×1 conv producing K channels (K = number of classes), giving K 28×28 mask logits.  
- Sigmoid per pixel to yield K independent binary masks.

A crucial design choice: **decouple mask and class prediction.**[file:127][web:52]

- The mask branch outputs one binary mask per class, without softmax competition across channels.  
- The class label is determined by the classification head.  
- At inference, only the mask corresponding to the predicted class is selected.

This decoupling improves mask AP compared to a joint class+mask softmax, as shown in the paper (+2.1 mask AP in their ablation).[web:52]

```text
RoIAlign output: 256 x 14 x 14
     |
[Conv 3x3, 256ch, ReLU] x4
     |
[ConvTranspose 2x2, stride 2]  -> 256 x 28 x 28
     |
[Conv 1x1]                     -> K x 28 x 28 (K binary mask logits)
     |
Sigmoid per pixel              -> K independent binary mask predictions
```

At inference: choose the mask for the predicted class, rescale it to the RoI box inside the original image, then threshold (e.g. at 0.5) to produce a binary mask.[file:127][web:52]

### Comparison with Semantic Segmentation

Mask R-CNN is an instance segmentation model. It differs from semantic segmentation architectures like U-Net in both task and structure:[file:127][web:51][web:52]

- **U-Net / semantic segmentation:**  
  - Single dense prediction over the entire image.  
  - Labels every pixel, including “background” regions.  
  - No explicit object instances.

- **Mask R-CNN / instance segmentation:**  
  - Detects objects first (via RPN + detection head).  
  - Predicts masks **per detected region**, not over the whole image at once.  
  - Does not explicitly label all background pixels; coverage is defined only over regions with detected objects.

---

## Summary Comparison

| Property                  | U-Net                          | Mask R-CNN                                  |
|---------------------------|--------------------------------|---------------------------------------------|
| Task                      | Semantic segmentation          | Instance segmentation                       |
| Object instances          | Not distinguished              | Per-instance masks                          |
| Input handling            | Full image, dense prediction   | Two-stage: proposals then per-RoI heads     |
| Skip connection mechanism | Encoder–decoder concatenation  | No cross-resolution skips; uses FPN instead |
| Typical domain            | Medical/biomedical imaging     | Natural images (COCO, etc.)                 |
| Output                    | H × W × #classes label map     | Per-RoI masks + boxes + class labels        |
| Key innovation            | Encoder–decoder with skips     | RoIAlign + decoupled mask head              |

[file:127][web:31][web:52]

---

## Metrics

**IoU (Intersection over Union).**  

For a class or an instance mask, IoU is:

\[
\text{IoU} = \frac{|A \cap B|}{|A \cup B|}
\]

where A is predicted region, B is ground-truth.[file:127][web:51] It ranges from 0 (no overlap) to 1 (perfect overlap). This is the standard localisation metric for segmentation and detection (for boxes, IoU is computed on bounding boxes).

**Dice coefficient.**  

Used extensively in medical segmentation:[file:127][web:31]

\[
\text{Dice} = \frac{2 |A \cap B|}{|A| + |B|}
\]

Dice and IoU are monotonically related via \(\text{Dice} = \frac{2 \, \text{IoU}}{1 + \text{IoU}}\), but Dice can be more sensitive for small structures.

**Mask AP (COCO).**  

Mask R-CNN is evaluated on COCO using mask average precision (AP):[file:127][web:52]

- AP is computed over multiple IoU thresholds (0.5:0.05:0.95) using mask IoU, not box IoU.  
- Commonly reported metrics: AP, AP\(_{50}\) (IoU ≥ 0.5), AP\(_{75}\), AP\(_S\)/AP\(_M\)/AP\(_L\) (small/medium/large objects).

This mirrors detection AP but uses mask overlap instead of bounding boxes.[web:52]
