# Mask R-CNN

**Paper:** “Mask R-CNN”  
**Authors:** Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick  
**Venue:** ICCV 2017 (Best Paper Award)  
**arXiv:** https://arxiv.org/abs/1703.06870

---

## Background: Faster R-CNN and the RoIPool operation

Mask R‑CNN extends Faster R‑CNN, so understanding the earlier system is a prerequisite.

### Faster R-CNN overview

Faster R‑CNN (Ren et al., NeurIPS 2015) is a **two‑stage** object detector.

- Stage 1: a Region Proposal Network (RPN) operates on backbone CNN features and outputs candidate boxes (region proposals) and objectness scores.  
- Stage 2: each proposal is mapped onto the feature map; a fixed‑size feature is extracted and fed to two heads that predict the final class label and a refined bounding box.

The operation that extracts a fixed‑size feature from an arbitrary‑sized proposal is **RoIPool** (Region of Interest Pooling).

### RoIPool and quantization error

Region proposals are defined in input‑image coordinates as floating‑point rectangles. To extract features, Faster R‑CNN performs:

1. Divide proposal coordinates by the backbone stride (e.g. 16 for VGG/ResNet) to map from image space to feature‑map space, yielding non‑integer coordinates in feature space.  
2. Round these floating‑point coordinates to integers to determine which feature map cells belong to the proposal (first quantization).  
3. Divide the resulting integer region into a fixed grid of bins (e.g. 7×7); bin boundaries are computed as floats and then rounded to integers (second quantization).  
4. Apply max pooling within each bin to obtain a single value per bin.

Each rounding step introduces **spatial misalignment** between the true proposal region and the area from which features are extracted. For a stride‑16 backbone, a single rounding step can misalign by up to 8 pixels in input space (half the stride), and two such steps can compound this.

For bounding box detection, this is usually tolerable: a bounding box is a coarse four‑number output, and small spatial errors in features can be corrected by the regression head. For instance segmentation, the output is a **per‑pixel mask**, so several‑pixel misalignment corrupts the spatial correspondence and yields masks that are blurred or shifted at boundaries. Mask prediction therefore demands much higher spatial precision than box prediction.

---

## RoIAlign: eliminating quantization

Mask R‑CNN replaces RoIPool with **RoIAlign**, which removes both rounding steps and preserves alignment.

### Floating-point grid sampling

Instead of rounding proposal and bin boundaries, RoIAlign keeps all coordinates as floating‑point values.

- For each bin in the fixed output grid (e.g. 7×7 for boxes, 14×14 for masks), RoIAlign places a small set of sampling points at regular *floating‑point* locations; the paper uses a 2×2 grid of four sampling points per bin by default.  
- Each sampling point lies at a non‑integer position on the feature map, so the feature value is computed by **bilinear interpolation** from the four neighbouring integer cells.

### Bilinear interpolation

For a sampling point at (x, y) in feature-map coordinates, let:

- `x1 = floor(x)`, `x2 = ceil(x)`
- `y1 = floor(y)`, `y2 = ceil(y)`

Define `dx = x - x1`, `dy = y - y1`. The interpolated feature value is

```text
v = (1 - dx)(1 - dy) · f(x1, y1)
  + dx       (1 - dy) · f(x2, y1)
  + (1 - dx) dy       · f(x1, y2)
  + dx       dy       · f(x2, y2)
```

where `f(x_i, y_i)` are the feature map values at integer locations. This yields a smoothly varying function of the sampling coordinates without discontinuities from rounding.

The final value for each output bin is the average (or max) of its sampled points. In Mask R‑CNN, average pooling is typically used for RoIAlign in the mask branch.

### Why this eliminates quantization error

RoIPool loses precision when mapping proposal boundaries and when binning; both involve rounding. RoIAlign keeps proposal and bin boundaries as floats and uses interpolation at fractional positions, so the extracted features remain **aligned** to the original region.

The paper shows that switching from RoIPool to RoIAlign—without changing the rest of the architecture—improves mask AP by roughly **2–3 points** and keypoint AP by roughly **4–5 points** (He et al. 2017, Table 2). This demonstrates that the “small” change of removing quantization has a large impact on precise, pixel‑level tasks.

---

## Mask R-CNN architecture

### Overall structure

Mask R‑CNN keeps the Faster R‑CNN two‑stage layout and adds a **third parallel head** in stage 2.

- Stage 1: RPN on top of a backbone (ResNet/ResNeXt + FPN) generates proposals.  
- Stage 2: RoIAlign extracts fixed‑size features for each proposal; three heads operate in parallel:  
  - classification head (class per proposal),  
  - box regression head (refined bounding box),  
  - mask head (per‑pixel segmentation for the proposal).

The three heads share the RoI‑aligned features but have disjoint parameters and separate loss terms. In the canonical configuration, the total loss is an **unweighted sum** of classification, box, and mask losses.

### Feature Pyramid Network backbone

Mask R‑CNN uses a **Feature Pyramid Network (FPN)** backbone (Lin et al. 2017) rather than a single‑scale feature map.

- FPN builds a top‑down pyramid (e.g. P2–P5) by combining higher‑level semantic features with higher‑resolution lower‑level features.  
- RPN proposals are assigned to pyramid levels based on their scale: small objects use higher‑resolution maps, large objects use coarser maps.  
- RoIAlign is then applied to the appropriate pyramid level for each proposal.

FPN improves multi‑scale detection and is critical for strong mask performance on COCO.

### The mask head

The mask head operates on RoI‑aligned features at a relatively high spatial resolution to preserve detail.

- Input: RoIAlign features of size 14×14×C per proposal.  
- Architecture:  
  - 4 convolutional layers (3×3, stride 1, padding 1) with ReLU, keeping 14×14 spatial size.  
  - One transposed convolution (deconvolution) with stride 2 to upscale to 28×28.  
  - Final 1×1 convolution producing **K channels** for K foreground classes.

The output is a 28×28×K tensor per RoI; each channel is a **binary mask** for one class. The mask head uses per‑pixel sigmoid activations and a binary cross‑entropy loss, as described next.

---

## Binary mask per class vs single multi-class mask

### Choice and rationale

A natural semantic‑segmentation design is a single mask with per‑pixel softmax over K+1 classes (K foreground + background). Mask R‑CNN instead predicts K **independent binary masks** and uses a sigmoid + binary cross‑entropy loss for each, decoupled from the class label prediction.

The rationale:

- In Faster R‑CNN/Mask R‑CNN, the class of each RoI is already predicted by the **classification head**, which has a global view of the RoI features.  
- The mask head should focus purely on localization: *given that this RoI belongs to class k, which pixels belong to that instance?*  
- If the mask head used a multi‑class softmax, it would also have to resolve class identity, competing with the classifier and coupling class and shape decisions.

By predicting one binary mask per class, the mask head answers only the spatial question, and the classification head answers the class question. At inference, Mask R‑CNN selects the mask channel corresponding to the predicted class and thresholds it to obtain the final instance mask.

This design also ensures **masks across classes do not compete**: per‑pixel sigmoid + binary loss does not require mask probabilities to sum to one across classes. In contrast, a softmax mask forces competition among classes at each pixel.

The paper reports that using a multi‑class softmax mask instead of per‑class sigmoids reduces mask AP by several points (≈4–5 AP), confirming this formulation is important, not cosmetic.

### Architectural difference from classifier head

The classification and mask heads reflect their distinct roles.

- **Classification head:**  
  - Input: 7×7 RoI feature (RoIAlign followed by pooling).  
  - Architecture: fully connected layers; spatial structure is collapsed to a vector.  
  - Output: K+1‑dimensional class logits; no spatial layout.

- **Mask head:**  
  - Input: 14×14 RoI feature (no global pooling).  
  - Architecture: fully convolutional (3×3 convs + upsampling); spatial structure preserved.  
  - Output: 28×28 spatial mask per class (K channels); dense prediction over pixels.

The mask head is a small FCN whose output remains aligned to the RoI region due to RoIAlign; the classifier is an image‑level (RoI‑level) predictor that discards spatial information.

---

## Summary

| Component      | Role                            | Key design choice                                      |
|----------------|---------------------------------|--------------------------------------------------------|
| RPN            | Generates candidate proposals   | Same as Faster R‑CNN                                   |
| RoIAlign       | Extracts fixed‑size RoI features| Floating‑point coordinates + bilinear interpolation    |
| Classification head | Class label per proposal  | FC layers; spatially collapsed                         |
| Box head       | Box refinement per proposal     | FC layers; spatially collapsed                         |
| Mask head      | Per‑pixel instance mask         | Conv layers; keeps spatial layout; K independent masks |

The two central contributions of Mask R‑CNN are:

- **RoIAlign**, which removes quantization from RoIPool and provides the spatial precision required for masks and keypoints.  
- The **per‑class binary mask formulation**, which decouples classification from segmentation and avoids cross‑class competition in the mask head.

Together, these changes allow a relatively simple extension of Faster R‑CNN to achieve state‑of‑the‑art instance segmentation performance on COCO.

---

## References

- He, K., Gkioxari, G., Dollár, P., and Girshick, R. “Mask R‑CNN.” ICCV 2017.  
- Ren, S., He, K., Girshick, R., and Sun, J. “Faster R‑CNN: Towards Real‑Time Object Detection with Region Proposal Networks.” NeurIPS 2015.  
- Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., and Belongie, S. “Feature Pyramid Networks for Object Detection.” CVPR 2017.
