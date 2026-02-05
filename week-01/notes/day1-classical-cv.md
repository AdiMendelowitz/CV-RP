### **Core Image Processing: Core Operations & Principles**

#### **Priority 1: Core Operations**

*   **Gaussian Blur from Scratch:** A smooth blur is achieved by convolving an image with a **Gaussian kernel**, which functions as a low-pass filter to reduce high-frequency noise,. In practice, this is often implemented using a discrete **binomial kernel** (e.g., the 5-tap filter $\frac{1}{16}$), as repeated convolutions with this kernel converge to a smooth Gaussian shape,. To implement this efficiently "from scratch," the operation is performed **separably**, applying a 1D horizontal blur followed by a 1D vertical blur,.
*   **Sobel Edge Detection:** The Sobel operator is a popular **3x3 edge extractor** used to find the **gradient** of an image. It is a separable combination of a **central difference** (to find the derivative) in one direction and a **box filter** (to smooth the result) in the perpendicular direction. This process generates **horizontal and vertical gradients**, emphasizing edges where intensity changes abruptly,.
*   **Convolution Engine and Padding:** When a kernel extends beyond image boundaries, the "engine" must use **padding** to prevent darkening or artifacts at the edges,. Common modes include **zero padding** (setting outside pixels to 0), **clamp/replicate** (repeating edge pixels), and **mirroring** (reflecting pixels across the edge). Proper padding ensures the linear system behaves consistently across the entire image domain,.

#### **Priority 2: Foundational Understanding**

*   **Convolution as a Weighted Sum:** Mathematically, a linear neighborhood operator determines the value of an output pixel by calculating a **weighted sum of the input pixels** in the vicinity of that location. The "weights" are defined by the **filter coefficients** within the kernel or mask. This operation is **linear shift-invariant (LSI)**, meaning it obeys the superposition principle and behaves the same way at every pixel location.
*   **Separability for Efficiency:** A 2D convolution kernel of size $K \times K$ normally requires **$K^2$ multiply-add operations** per pixel. If a kernel is **separable**, it can be decomposed into the outer product of a vertical and horizontal kernel ($K = vh^T$), allowing the operation to be performed in **$2K$ operations**,. This optimization significantly increases processing speed and often influences the design of kernels used in computer vision.
*   **Edge Detection as Taking Derivatives:** Finding edges in an image is mathematically equivalent to **taking derivatives** of the image function. First-order derivatives (like Sobel) identify the **gradient field**, while second-order derivatives (like the **Laplacian**) respond to rapid changes in the gradient, such as corners or lines,. Because differentiation linearly magnifies higher frequencies, it effectively highlights the sharp transitions that define object boundaries.


# Day 1 - Canny Edge Detector

## Implementation
- Location: `code/classical_cv/edge_detection.py`
- Components:
  - Gaussian blur (from filters.py)
  - Sobel gradients (from filters.py)
  - Non-maximum suppression
  - Double threshold
  - Hysteresis

## Key learnings
- Hysteresis connects weak edges to strong edges
- Sliding windows for efficient neighbor operations
- Ratios for adaptive thresholding

## Performance
- Match rate vs OpenCV: ~XX%
- Works on test patterns: checkerboard, gradient, circle