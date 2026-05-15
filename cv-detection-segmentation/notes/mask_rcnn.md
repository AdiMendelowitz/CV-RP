# Mask R-CNN

**Paper:** “Mask R-CNN”  
**Authors:** Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick  
**Venue:** ICCV 2017 (Best Paper Award)  
**arXiv:** https://arxiv.org/abs/1703.06870[file:210][web:211][web:212]

---

## Background: Faster R-CNN and the RoIPool operation

Mask R‑CNN extends Faster R‑CNN, so understanding the earlier system is a prerequisite.[file:210][web:211]

### Faster R-CNN overview

Faster R‑CNN (Ren et al., NeurIPS 2015) is a **two‑stage** object detector.[file:210][web:215]

- Stage 1: a Region Proposal Network (RPN) operates on backbone CNN features and outputs candidate boxes (region proposals) and objectness scores.  
- Stage 2: each proposal is mapped onto the feature map; a fixed‑size feature is extracted and fed to two heads that predict the final class label and a refined bounding box.[file:210][web:215]

The operation that extracts a fixed‑size feature from an arbitrary‑sized proposal is **RoIPool** (Region of Interest Pooling).[file:210][web:215]

### RoIPool and quantization error

Region proposals are defined in input‑image coordinates as floating‑point rectangles.[file:210][web:215] To extract features, Faster R‑CNN performs:

1. Divide proposal coordinates by the backbone stride (e.g. 16 for VGG/ResNet) to map from image space to feature‑map space, yielding non‑integer coordinates in feature space.  
2. Round these floating‑point coordinates to integers to determine which feature map cells belong to the proposal (first quantization).  
3. Divide the resulting integer region into a fixed grid of bins (e.g. 7×7); bin boundaries are computed as floats and then rounded to integers (second quantization).  
4. Apply max pooling within each bin to obtain a single value per bin.[file:210][web:215]

Each rounding step introduces **spatial misalignment** between the true proposal region and the area from which features are extracted.[file:210] For a stride‑16 backbone, a single rounding step can misalign by up to 8 pixels in input space (half the stride), and two such steps can compound this.[file:210]

For bounding box detection, this is usually tolerable: a bounding box is a coarse four‑number output, and small spatial errors in features can be corrected by the regression head.[file:210] For instance segmentation, the output is a **per‑pixel mask**, so several‑pixel misalignment corrupts the spatial correspondence and yields masks that are blurred or shifted at boundaries.[file:210][web:211] Mask prediction therefore demands much higher spatial precision than box prediction.

---

## RoIAlign: eliminating quantization

Mask R‑CNN replaces RoIPool with **RoIAlign**, which removes both rounding steps and preserves alignment.[file:210][web:211][web:212]

### Floating-point grid sampling

Instead of rounding proposal and bin boundaries, RoIAlign keeps all coordinates as floating‑point values.[file:210][web:211]

- For each bin in the fixed output grid (e.g. 7×7 for boxes, 14×14 for masks), RoIAlign places a small set of sampling points at regular *floating‑point* locations; the paper uses a 2×2 grid of four sampling points per bin by default.[file:210][web:211]  
- Each sampling point lies at a non‑integer position on the feature map, so the feature value is computed by **bilinear interpolation** from the four neighbouring integer cells.[file:210][web:211]

### Bilinear interpolation

For a sampling point at (x, y) in feature‑map coordinates, let:

- \((x_1, y_1) = (\lfloor x \rfloor, \lfloor y \rfloor)\)  
- \((x_2, y_1) = (\lceil x \rceil, \lfloor y \rfloor)\)  
- \((x_1, y_2) = (\lfloor x \rfloor, \lceil y \rceil)\)  
- \((x_2, y_2) = (\lceil x \rceil, \lceil y \rceil)\)

Define \(dx = x - \lfloor x \rfloor\), \(dy = y - \lfloor y \rfloor\).[file:210] The interpolated feature value is

```text
v = (1 - dx)(1 - dy) · f(x1, y1)
  + dx       (1 - dy) · f(x2, y1)
  + (1 - dx) dy       · f(x1, y2)
  + dx       dy       · f(x2, y2)
```

where \(f(x_i, y_i)\) are the feature map values at integer locations.[file:210] This yields a smoothly varying function of the sampling coordinates without discontinuities from rounding.

The final value for each output bin is the average (or max) of its sampled points.[file:210] In Mask R‑CNN, average pooling is typically used for RoIAlign in the mask branch.[web:211]

### Why this eliminates quantization error

RoIPool loses precision when mapping proposal boundaries and when binning; both involve rounding.[file:210] RoIAlign keeps proposal and bin boundaries as floats and uses interpolation at fractional positions, so the extracted features remain **aligned** to the original region.[file:210][web:211][web:212]

The paper shows that switching from RoIPool to RoIAlign—without changing the rest of the architecture—improves mask AP by roughly **2–3 points** and keypoint AP by roughly **4–5 points** (He et al. 2017, Table 2).[file:210][web:211][web:212] This demonstrates that the “small” change of removing quantization has a large impact on precise, pixel‑level tasks.

---

## Mask R-CNN architecture

### Overall structure

Mask R‑CNN keeps the Faster R‑CNN two‑stage layout and adds a **third parallel head** in stage 2.[file:210][web:211][web:214]

- Stage 1: RPN on top of a backbone (ResNet/ResNeXt + FPN) generates proposals.  
- Stage 2: RoIAlign extracts fixed‑size features for each proposal; three heads operate in parallel:  
  - classification head (class per proposal),  
  - box regression head (refined bounding box),  
  - mask head (per‑pixel segmentation for the proposal).[file:210][web:211]

The three heads share the RoI‑aligned features but have disjoint parameters and separate loss terms.[file:210] In the canonical configuration, the total loss is an **unweighted sum** of classification, box, and mask losses.[web:211][web:215]

### Feature Pyramid Network backbone

Mask R‑CNN uses a **Feature Pyramid Network (FPN)** backbone (Lin et al. 2017) rather than a single‑scale feature map.[file:210][web:211][web:215]

- FPN builds a top‑down pyramid (e.g. P2–P5) by combining higher‑level semantic features with higher‑resolution lower‑level features.  
- RPN proposals are assigned to pyramid levels based on their scale: small objects use higher‑resolution maps, large objects use coarser maps.[file:210][web:211]  
- RoIAlign is then applied to the appropriate pyramid level for each proposal.

FPN improves multi‑scale detection and is critical for strong mask performance on COCO.[web:211][web:215]

### The mask head

The mask head operates on RoI‑aligned features at a relatively high spatial resolution to preserve detail.[file:210][web:211][web:213]

- Input: RoIAlign features of size 14×14×C per proposal.  
- Architecture:  
  - 4 convolutional layers (3×3, stride 1, padding 1) with ReLU, keeping 14×14 spatial size.  
  - One transposed convolution (deconvolution) with stride 2 to upscale to 28×28.  
  - Final 1×1 convolution producing **K channels** for K foreground classes.[file:210][web:211][web:213]

The output is a 28×28×K tensor per RoI; each channel is a **binary mask** for one class.[file:210][web:211][web:215] The mask head uses per‑pixel sigmoid activations and a binary cross‑entropy loss, as described next.[web:211][web:216]

---

## Binary mask per class vs single multi-class mask

### Choice and rationale

A natural semantic‑segmentation design is a single mask with per‑pixel softmax over K+1 classes (K foreground + background).[file:210] Mask R‑CNN instead predicts K **independent binary masks** and uses a sigmoid + binary cross‑entropy loss for each, decoupled from the class label prediction.[file:210][web:211][web:216]

The rationale:

- In Faster R‑CNN/Mask R‑CNN, the class of each RoI is already predicted by the **classification head**, which has a global view of the RoI features.[file:210][web:211][web:215]  
- The mask head should focus purely on localization: *given that this RoI belongs to class k, which pixels belong to that instance?*  
- If the mask head used a multi‑class softmax, it would also have to resolve class identity, competing with the classifier and coupling class and shape decisions.[web:216]

By predicting one binary mask per class, the mask head answers only the spatial question, and the classification head answers the class question.[file:210][web:211][web:216] At inference, Mask R‑CNN selects the mask channel corresponding to the predicted class and thresholds it to obtain the final instance mask.[file:210][web:211]

This design also ensures **masks across classes do not compete**: per‑pixel sigmoid + binary loss does not require mask probabilities to sum to one across classes.[web:215][web:216] In contrast, a softmax mask forces competition among classes at each pixel.

The paper reports that using a multi‑class softmax mask instead of per‑class sigmoids reduces mask AP by several points (≈4–5 AP), confirming this formulation is important, not cosmetic.[file:210][web:211][web:215][web:216]

### Architectural difference from classifier head

The classification and mask heads reflect their distinct roles.[file:210][web:211][web:215]

- **Classification head:**  
  - Input: 7×7 RoI feature (RoIAlign followed by pooling).  
  - Architecture: fully connected layers; spatial structure is collapsed to a vector.  
  - Output: K+1‑dimensional class logits; no spatial layout.

- **Mask head:**  
  - Input: 14×14 RoI feature (no global pooling).  
  - Architecture: fully convolutional (3×3 convs + upsampling); spatial structure preserved.  
  - Output: 28×28 spatial mask per class (K channels); dense prediction over pixels.[file:210][web:211][web:213]

The mask head is a small FCN whose output remains aligned to the RoI region due to RoIAlign; the classifier is an image‑level (RoI‑level) predictor that discards spatial information.[file:210][web:211]

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

- **RoIAlign**, which removes quantization from RoIPool and provides the spatial precision required for masks and keypoints.[file:210][web:211][web:212]  
- The **per‑class binary mask formulation**, which decouples classification from segmentation and avoids cross‑class competition in the mask head.[file:210][web:211][web:215][web:216]

Together, these changes allow a relatively simple extension of Faster R‑CNN to achieve state‑of‑the‑art instance segmentation performance on COCO.[web:211][web:213]

---

## References

- He, K., Gkioxari, G., Dollár, P., and Girshick, R. “Mask R‑CNN.” ICCV 2017.[web:211][web:212][web:213]  
- Ren, S., He, K., Girshick, R., and Sun, J. “Faster R‑CNN: Towards Real‑Time Object Detection with Region Proposal Networks.” NeurIPS 2015.[web:215]  
- Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., and Belongie, S. “Feature Pyramid Networks for Object Detection.” CVPR 2017.[web:215]
