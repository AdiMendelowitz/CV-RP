# Mask R-CNN

**Paper:** "Mask R-CNN"  
**Authors:** Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick  
**Venue:** ICCV 2017 (Best Paper Award)  
**arXiv:** https://arxiv.org/abs/1703.06870

---

## Background: Faster R-CNN and the RoIPool Operation

Mask R-CNN extends Faster R-CNN, so understanding the earlier system is a prerequisite for understanding what Mask R-CNN changes and why.

### Faster R-CNN overview

Faster R-CNN (Ren et al., NeurIPS 2015) is a two-stage object detector. In the first stage, a Region Proposal Network (RPN) operates on the feature map produced by a backbone CNN and outputs a set of candidate bounding boxes, called region proposals, along with objectness scores. In the second stage, each proposal is mapped onto the feature map, a fixed-size feature vector is extracted from that region, and a pair of heads predict the final class label and a refined bounding box.

The operation that extracts a fixed-size feature from an arbitrarily sized proposal region is called RoI Pooling (Region of Interest Pooling).

### RoI Pooling and quantization error

A region proposal is defined in input image coordinates as a floating-point rectangle. To extract features from this region, two coordinate transformations are required:

1. The proposal coordinates are divided by the backbone stride (typically 16 for a VGG or ResNet backbone) to map from image space to feature map space. Since the stride is an integer and the proposal coordinates are floating-point, this division generally produces non-integer feature map coordinates.

2. The resulting floating-point coordinates are rounded to the nearest integer to identify which feature map cells fall within the proposal. This rounding is the first quantization step.

3. The rounded region is then divided into a fixed grid of bins (typically 7x7). The bin boundaries are again computed as floating-point values and rounded to integers. This is the second quantization step.

4. Within each bin, max pooling produces a single value.

Each rounding step introduces a spatial misalignment between the actual proposal region and the region from which features are extracted. For a stride-16 backbone, a single rounding operation can misalign by up to 8 pixels in input image space (half the stride), and two rounding operations can compound this further.

For bounding box detection, this misalignment is tolerable. A box prediction is a coarse, four-number output; small spatial errors in the extracted features produce small errors in the predicted coordinates, which are further corrected by the bounding box regression head. The quantization error is absorbed by the regression.

For instance segmentation, the output is a per-pixel spatial map, not a four-number summary. A misalignment of several pixels in the feature extraction directly corrupts the spatial correspondence between the feature map and the object, producing masks that are blurred at boundaries or systematically shifted. The precision requirement for masks is fundamentally higher than for boxes, and quantization error that is acceptable for detection is destructive for segmentation.

---

## RoIAlign: Eliminating Quantization

Mask R-CNN replaces RoI Pooling with RoIAlign, which removes both rounding steps entirely.

### Floating-point grid sampling

Instead of rounding proposal and bin boundaries to integer coordinates, RoIAlign keeps all coordinates as floating-point values throughout. Within each bin of the fixed output grid, a small set of sampling points is placed at regular floating-point locations (the paper uses a 2x2 grid of four points per bin by default).

Each sampling point falls at a non-integer position in the feature map. Since there is no feature map value defined at a non-integer position, the value at each sampling point is computed by bilinear interpolation from the four nearest integer feature map locations.

### Bilinear interpolation

For a sampling point at floating-point position (x, y) in the feature map, the four surrounding integer-coordinate cells are (floor(x), floor(y)), (ceil(x), floor(y)), (floor(x), ceil(y)), and (ceil(x), ceil(y)). Let dx = x - floor(x) and dy = y - floor(y). The interpolated value is:

```
v = (1 - dx)(1 - dy) * f(x1, y1)
  + dx * (1 - dy)    * f(x2, y1)
  + (1 - dx) * dy    * f(x1, y2)
  + dx * dy          * f(x2, y2)
```

where f(xi, yi) denotes the feature map value at integer cell (xi, yi). This is a weighted average of four neighbours, with weights proportional to proximity. The result is a smoothly varying function of the proposal coordinates, with no discontinuities from rounding.

The final value for each output bin is the average (or max) of the sampled points within that bin.

### Why this eliminates quantization error

RoI Pooling loses spatial precision at two points: when mapping proposal boundaries to the feature map and when dividing the region into bins. Both involve rounding. RoIAlign removes both rounding steps: proposal boundaries are kept as floating-point values, bin boundaries are kept as floating-point values, and feature values at floating-point positions are computed by interpolation rather than cell lookup.

The result is that the extracted features are precisely aligned with the actual proposal region. The paper demonstrates that this single change improves mask AP by around 3 points and keypoint AP by around 5 points, with no change to the model architecture or loss function.

---

## Mask R-CNN Architecture

### Overall structure

Mask R-CNN keeps the Faster R-CNN two-stage structure intact and adds a third parallel head to the second stage. Where Faster R-CNN has a classification head and a box regression head operating on each RoI feature, Mask R-CNN adds a mask head that predicts a spatial segmentation mask for the same RoI.

The three heads share the RoI feature but are otherwise independent: each has its own parameters and its own loss term. The total loss is a sum of the classification loss, the box regression loss, and the mask loss, with no weighting between them in the published configuration.

### Feature Pyramid Network backbone

The published Mask R-CNN uses a Feature Pyramid Network (FPN, Lin et al. 2017) as the backbone rather than a single-scale feature map. FPN constructs a multi-scale feature pyramid by combining bottom-up features from a ResNet with top-down lateral connections, producing feature maps at four spatial scales. RPN proposals at different scales are assigned to different pyramid levels based on proposal size, so small objects are detected from high-resolution feature maps and large objects from low-resolution ones. RoIAlign is applied at whichever pyramid level the proposal is assigned to.

### The mask head

The mask head receives the RoI-aligned feature for each proposal, which has a fixed spatial size of 14x14 (larger than the 7x7 used for the box and class heads, to preserve more spatial detail). It applies a sequence of four 3x3 convolutional layers, each followed by ReLU, and then a single transposed convolution (deconvolution) with stride 2 that upsamples the feature map to 28x28. A final 1x1 convolution produces the output.

The output has K channels for K foreground classes, giving a 28x28xK tensor per RoI. Each of the K channels is a binary mask predicting, independently, whether each pixel belongs to that class.

---

## Binary Mask Per Class vs. Single Multi-Class Mask

### The choice and its rationale

A natural way to formulate instance segmentation would be to predict a single spatial map where each pixel is assigned a class label from a softmax over K+1 classes (K foreground classes plus background). This is the approach used in semantic segmentation (FCN, DeepLab). Mask R-CNN instead predicts K independent binary masks, one per class, and uses a sigmoid with binary cross-entropy loss on each independently.

The key reason is the decoupling of classification from segmentation. In the two-stage Faster R-CNN framework, the class of each proposal is already predicted by the classification head, which has access to the full spatial feature and is well-suited to that task. If the mask head also had to resolve the class identity (as a multi-class softmax mask would require), it would be competing with the classification head and potentially interfering with it.

By predicting a binary mask per class, the mask head is only asked to answer the spatial question: for this proposal, which pixels belong to an object of class k? The class identity question (which k to use at inference) is answered separately by the classification head. At inference, only the mask channel corresponding to the predicted class is selected and thresholded to produce the final instance mask.

This formulation prevents competition between classes within the mask head. In a multi-class softmax mask, the probabilities for all classes at each pixel must sum to one, so a pixel can only be assigned to one class. If there is any ambiguity in the spatial features about the exact boundary, the softmax forces a hard competition. Binary masks per class allow each class to independently assess its own evidence without suppressing others, which is a more appropriate inductive bias when the class identity is already resolved upstream.

The paper reports that replacing the per-class binary mask formulation with a single softmax mask reduces mask AP by around 5 points, confirming that the decoupling is materially important rather than a minor implementation choice.

### Architectural difference from a classifier head

The classification head and the mask head differ in both structure and output type, reflecting their different functions.

The classification head takes the 7x7 RoI feature, collapses it to a vector via global average pooling or flattening, and passes it through one or two fully connected layers to produce a class score vector of length K+1. Spatial structure is discarded; only a class label (a distribution over K+1 scalars) is produced. This is the standard design for image classification.

The mask head never collapses spatial structure. It applies convolutional layers throughout, preserving the 2D spatial layout of the feature. The upsampling step via transposed convolution increases spatial resolution from 14x14 to 28x28, restoring detail that was compressed by the backbone. The output is a 2D spatial map of shape 28x28 per class, not a scalar.

The distinction is between a classification function (input -> class distribution) and a dense prediction function (input spatial map -> output spatial map). Fully connected layers are appropriate for the former because they aggregate all spatial positions into a single output. Convolutional layers are appropriate for the latter because they maintain spatial correspondence between input positions and output positions, which is precisely what is needed to predict which pixel belongs to which object.

---

## Summary

| Component | Role | Key Design Choice |
|-----------|------|------------------|
| RPN | Generates candidate proposals | Shared with Faster R-CNN, unchanged |
| RoIAlign | Extracts fixed-size features per proposal | Bilinear interpolation; no quantization |
| Classification head | Predicts class label per proposal | FC layers; collapses spatial structure |
| Box head | Predicts refined bounding box | FC layers; collapses spatial structure |
| Mask head | Predicts per-pixel binary mask | Conv layers; preserves spatial structure; K independent binary masks |

The two central contributions of Mask R-CNN are RoIAlign, which provides the spatial precision that mask prediction requires, and the per-class binary mask formulation, which decouples segmentation from classification and prevents cross-class interference in the mask head.

---

## References

- He, K., Gkioxari, G., Dollar, P., and Girshick, R. "Mask R-CNN." ICCV 2017. https://arxiv.org/abs/1703.06870
- Ren, S., He, K., Girshick, R., and Sun, J. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." NeurIPS 2015. https://arxiv.org/abs/1506.01497
- Lin, T.-Y., Dollar, P., Girshick, R., He, K., Hariharan, B., and Belongie, S. "Feature Pyramid Networks for Object Detection." CVPR 2017. https://arxiv.org/abs/1612.03144