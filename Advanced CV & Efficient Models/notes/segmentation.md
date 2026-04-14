# Image Segmentation: U-Net and Mask R-CNN

## Semantic vs. Instance Segmentation

These two tasks are often conflated but answer fundamentally different questions.

**Semantic segmentation** assigns a class label to every pixel in the image. All pixels belonging to the same category receive the same label, regardless of whether they come from different objects. A street scene processed with semantic segmentation would color every car identically, every pedestrian identically, and every road surface identically, with no notion that two adjacent cars are distinct objects.

**Instance segmentation** goes further: it detects each individual object as a separate instance and produces a pixel-precise mask for each one. The same street scene would yield a separate mask for car 1, car 2, pedestrian 1, pedestrian 2, and so on. Instance segmentation combines the localization of object detection with the pixel precision of segmentation.

The distinction matters for applications. Semantic segmentation is sufficient for road surface detection or sky parsing. Instance segmentation is required whenever individual objects must be counted, tracked, or independently analyzed, such as cell counting in microscopy or vehicle tracking in autonomous driving.

---

## U-Net (Ronneberger, Fischer, Brox -- MICCAI 2015)

**Reference:** Ronneberger O., Fischer P., Brox T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015, LNCS 9351, pp. 234-241.

### Motivation

Before U-Net, the dominant approach for pixel-wise prediction was sliding-window classification: a patch centered on each target pixel was fed through a network that predicted the class of that pixel. This had two problems. It was slow because the network had to run independently for each pixel, with heavy redundancy from overlapping patches. And it faced a direct tradeoff between localization accuracy and the use of context: a larger patch gave more context but reduced spatial precision, while a smaller patch improved localization but reduced context.

U-Net resolves this tradeoff through an encoder-decoder design with skip connections that simultaneously provide global context and fine spatial detail.

### Architecture

The architecture is a symmetric encoder-decoder. The encoder (contracting path) progressively reduces spatial resolution while increasing channel depth, building a rich semantic representation. The decoder (expanding path) progressively recovers spatial resolution, guided by high-resolution feature maps transferred directly from the encoder via skip connections.

**Encoder (contracting path).** Each stage applies two 3x3 unpadded convolutions, each followed by ReLU, then a 2x2 max pooling with stride 2 for downsampling. Because the convolutions are unpadded, each 3x3 conv reduces spatial dimensions by 2 pixels (1 pixel per side). The number of feature channels doubles at each downsampling step.

**Decoder (expanding path).** Each stage applies a 2x2 transposed convolution (up-conv) that doubles spatial dimensions and halves the channel count, concatenates the corresponding encoder feature map (cropped to match the decoder's smaller spatial size), then applies two 3x3 convolutions with ReLU.

**Final layer.** A 1x1 convolution maps the 64-channel feature map to the desired number of output classes.

**Standard configuration channel progression:**

```
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

Note that the output (388x388) is smaller than the input (572x572). This is a direct consequence of unpadded convolutions: each 3x3 valid conv reduces each spatial dimension by 2. The paper addresses this with a "overlap-tile" inference strategy for large images.

### ASCII Architecture Diagram

```
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

### Skip Connections: U-Net vs. ResNet

This is a conceptually important distinction.

**ResNet skip connections** add the input of a block to its output via a residual (element-wise addition). Their purpose is to improve gradient flow during backpropagation. When a network is very deep, gradients in the early layers become vanishingly small because they are multiplied by many small values as they propagate backward. The residual connection creates a direct path for gradients to bypass the block, so the gradient of the loss with respect to early-layer parameters is at least 1 (from the identity path) plus whatever the residual branch contributes. This stabilizes training but does not transfer spatial detail across architectural levels.

**U-Net skip connections** concatenate encoder feature maps with decoder feature maps across the spatial resolution hierarchy (channel concatenation, not addition). Their purpose is to restore spatial information that was discarded during pooling. When the encoder applies max pooling, it discards which of the 4 pixels in each 2x2 window contained the maximum value -- the spatial detail is gone from the activation map, even though the semantic content is preserved. By concatenating the pre-pooling feature maps directly into the decoder, the decoder has access to the fine-grained edge and texture information that pooling discarded. This is what enables the decoder to produce pixel-precise boundaries rather than coarse masks.

In short: ResNet skip connections solve an optimization problem (vanishing gradients). U-Net skip connections solve an information bottleneck problem (lost spatial detail).

### Loss Function

The paper uses pixel-wise cross-entropy with a custom weight map that gives higher weight to pixels near cell boundaries. For two adjacent cells touching each other, the boundary between them is given the highest weight so the network is pushed to predict the separation correctly. The weight for each pixel is:

```
w(x) = w_class(x) + w0 * exp( -(d1(x) + d2(x))^2 / (2 * sigma^2) )
```

where `w_class(x)` is a class frequency weight to handle imbalance, `d1(x)` is the distance from pixel x to the nearest cell border, `d2(x)` is the distance to the second nearest cell border, `w0 = 10`, and `sigma = 5` pixels.

### Why It Trains Well on Small Datasets

The paper achieves strong results from very few annotated images by using aggressive elastic deformation augmentation. Elastic deformation shifts pixels by a smooth random displacement field, mimicking the kind of natural tissue deformation that occurs in biological samples. Because the model sees the same image in many geometrically plausible configurations, it learns shape-invariant representations without requiring additional ground truth.

---

## Mask R-CNN (He, Gkioxari, Dollar, Girshick -- ICCV 2017)

**Reference:** He K., Gkioxari G., Dollar P., Girshick R. "Mask R-CNN." Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2961-2969.

### Problem Statement

Mask R-CNN addresses instance segmentation: detecting each object instance and producing a per-pixel binary mask for it. It extends Faster R-CNN, which was a two-stage detector producing class labels and bounding boxes per region, by adding a parallel mask prediction branch.

### Architecture Overview

The full forward pass has three stages.

**Stage 1: Backbone + FPN.** A convolutional backbone (ResNet-50 or ResNet-101) extracts feature maps. These are passed through a Feature Pyramid Network (FPN), which combines features at multiple resolution levels into a single multi-scale feature pyramid. This allows the detector to handle objects at different scales.

**Stage 2: Region Proposal Network (RPN).** Identical to Faster R-CNN's RPN. It slides a small network over the feature pyramid and proposes candidate bounding boxes (region proposals) along with objectness scores. Proposals with low objectness scores are filtered out.

**Stage 3: Per-RoI heads (parallel).** For each surviving region proposal, features are extracted from the feature pyramid using RoIAlign (described below). Three parallel branches then operate on these features:
- Classification branch: predicts the object class.
- Box regression branch: refines the bounding box coordinates.
- Mask branch: predicts a binary segmentation mask.

The three branches are trained jointly with a combined loss:

```
L = L_cls + L_box + L_mask
```

`L_cls` and `L_box` are the same cross-entropy and smooth-L1 losses used in Faster R-CNN. `L_mask` is the average binary cross-entropy loss over the mask output.

### RoIAlign: The Critical Contribution

The most important technical contribution in the paper is RoIAlign, which replaces the RoIPool operation from earlier R-CNN variants.

**The problem with RoIPool.** A region proposal has floating-point coordinates in the original image space. To extract features from the corresponding region of a feature map (which has lower resolution due to pooling strides), RoIPool first rounds the proposal coordinates to the nearest integer on the feature map grid, then divides the resulting region into a fixed grid of bins (e.g., 7x7), and rounds each bin boundary to the nearest integer as well. This quantization introduces spatial misalignment between the region proposal and the extracted features. For classification and bounding box detection, a few pixels of misalignment is tolerable. For mask prediction, where the goal is pixel-precise boundary localization, this misalignment is fatal.

**RoIAlign.** RoIAlign removes all quantization. The proposal boundaries are kept as floating-point values on the feature map coordinate system. Each bin of the output grid (e.g., 7x7) is sampled at 4 regularly spaced points, and each sample point is computed by bilinear interpolation from the four neighboring feature map pixels. The four sampled values are aggregated (max or average) to produce the bin's output. No coordinates are rounded at any stage. The paper shows that this change alone improves mask AP by a relative 10% to 50% depending on the localization threshold.

```
RoIPool (old):                      RoIAlign (new):
  proposal: (x=12.7, y=8.3)           proposal: (x=12.7, y=8.3)
  -> round to (13, 8)                  kept as (12.7, 8.3)
  -> bin boundaries rounded            bin boundaries kept as floats
  -> discrete pixel lookups            4 sample points per bin
                                       each computed by bilinear interp
```

### Mask Head Design

The mask branch operates on the 14x14 feature map output of RoIAlign (the box and class branches use 7x7; the mask branch uses a larger resolution for finer spatial detail). It consists of four 3x3 conv layers each with 256 channels and ReLU, followed by a 2x2 transposed convolution that doubles the resolution to 28x28, followed by a 1x1 conv that produces K binary masks (one per class) at 28x28 resolution.

A critical design decision is **decoupled mask and class prediction**. The mask branch produces K independent binary masks, one for each class, without any competition between classes. The class label comes from the separate classification branch. During inference, only the mask corresponding to the predicted class is selected. This decoupling allows the mask branch to focus entirely on accurate pixel-level boundary prediction without being confused by inter-class competition, which the paper shows yields a 2.1 mask AP improvement over a coupled design.

```
RoIAlign output: 256 x 14 x 14
     |
[Conv 3x3, 256ch, ReLU] x4
     |
[ConvTranspose 2x2, stride 2]  -> 256 x 28 x 28
     |
[Conv 1x1]                     -> K x 28 x 28  (K binary masks)
     |
Sigmoid per pixel              -> K independent binary mask predictions
```

At inference: select mask for predicted class, resize to RoI dimensions, threshold at 0.5.

### Comparison with Semantic Segmentation

Mask R-CNN is an instance segmentation model, not a semantic segmentation model. The distinction in the architecture reflects the task difference:

- Semantic segmentation (U-Net style) produces a single segmentation map over the entire image where every pixel is labeled. There is no concept of individual instances.
- Instance segmentation (Mask R-CNN) first detects objects, then segments each detected object in isolation. The mask branch runs independently per region proposal, not over the whole image.

This means Mask R-CNN cannot assign labels to background regions and does not produce complete pixel coverage of the image. It handles only the detected object instances.

---

## Summary Comparison

| Property | U-Net | Mask R-CNN |
|---|---|---|
| Task | Semantic segmentation | Instance segmentation |
| Object instances | Not distinguished | Per-instance masks |
| Input | Full image, dense prediction | Per-region, two-stage |
| Skip connection mechanism | Channel concatenation (spatial detail) | None (FPN handles scale) |
| Typical domain | Medical imaging, few labeled samples | Natural images, COCO-scale data |
| Output | H x W x num_classes label map | Per-RoI binary masks + boxes + labels |
| Key innovation | Encoder-decoder with skip connections | RoIAlign + decoupled mask branch |

---

## Metrics

**IoU (Intersection over Union).** The standard per-class segmentation metric. For a given class, IoU = (predicted AND ground truth) / (predicted OR ground truth). A value of 1.0 means perfect overlap; 0 means no overlap. Also called the Jaccard index.

**Dice coefficient.** Common in medical segmentation. Dice = 2 * |A AND B| / (|A| + |B|). Dice is related to IoU by the formula Dice = 2 * IoU / (1 + IoU), so they are monotonically related but Dice weights overlap more heavily for small regions.

**mAP for instance segmentation.** Mask R-CNN is evaluated using COCO's mask AP metric, which is equivalent to detection AP but using mask IoU rather than bounding box IoU. AP is computed at multiple IoU thresholds (0.5:0.05:0.95) and averaged. This is why Mask R-CNN papers report mAP@0.5 and mAP@0.5:0.95 as separate numbers.