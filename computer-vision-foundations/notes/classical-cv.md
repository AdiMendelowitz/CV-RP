# Core Image Processing: Core Operations and Principles

## Priority 1: Core Operations

**Gaussian Blur from Scratch.**  
A smooth blur is obtained by convolving the image with a **Gaussian kernel**, which acts as a low‑pass filter and suppresses high‑frequency noise and detail. In practice, small‑sigma Gaussians are often implemented using a discrete **binomial kernel** such as the 5‑tap filter \`/ 16\`; this corresponds to the coefficients of a row of Pascal’s triangle and provides a very good approximation to a discrete Gaussian. Because the 2D Gaussian is separable, the filter is applied efficiently as two 1D convolutions—first horizontally, then vertically—rather than as a full 2D kernel.

**Sobel Edge Detection.**  
The Sobel operator is a classic **3×3 gradient filter** used to estimate horizontal and vertical intensity derivatives. It can be interpreted as a separable combination of a **central difference** in one direction `to approximate the derivative` and a **
** smoothing filter in the perpendicular direction `a small binomial/box‑like filter`, which reduces noise in the gradient estimate. Applying Sobel in x and y yields gradient components that emphasise edges where intensity changes sharply.

**Convolution Engine and Padding.**  
When the convolution kernel extends beyond image boundaries, some form of **padding** is required to avoid artifacts and unintended darkening near edges. Common padding modes include:  
- **Zero padding:** outside pixels treated as 0 `can darken borders`.  
- **Replicate/clamp:** extend the nearest border value outward.  
- **Mirror/reflect:** mirror the image content at the border.  

Choosing appropriate padding ensures the linear system behaves consistently across the image, especially for repeated filtering stages `e.g., multiple blurs or derivatives`.

---

## Priority 2: Foundational Understanding

**Convolution as a Weighted Sum.**  
In discrete 2D images, a linear neighbourhood operator defines each output pixel as a **weighted sum of nearby input pixels**, with weights given by the kernel coefficients. This operation is linear and shift‑invariant `LSI`: it obeys superposition and behaves identically at every spatial location, which is why convolution is the canonical model for linear filtering in computer vision.

**Separability for Efficiency.**  
A general \`K \times K\` kernel requires \`K^2\` multiply‑adds per output pixel. If the kernel is **separable**, it can be written as the outer product of a column and row vector, \`K = v \, h^\top\`. In that case, 2D convolution can be implemented as a 1D convolution with \`h\` followed by a 1D convolution with \`v\`, reducing the cost to \`2K\` multiplies per pixel. This is a major reason why Gaussian and binomial filters `which are separable` are favoured in many classical CV pipelines.

**Edge Detection as Taking Derivatives.**  
Edges correspond to locations where the image intensity function has large gradients. First‑order derivative filters `e.g., Sobel, Prewitt` approximate the **gradient field** and highlight edges where intensity changes rapidly. Second‑order derivatives `e.g., the **Laplacian**` respond to changes in the gradient itself and are sensitive to fine structures such as corners and line crossings. In the frequency domain, differentiation amplifies high frequencies, which is why derivative operators emphasize sharp transitions.

---

# Canny Edge Detector

## Implementation

- Location: `code/classical_cv/edge_detection.py`  
- Components:  
  - Gaussian blur `from `filters.py``  
  - Sobel gradients `from `filters.py``  
  - Non‑maximum suppression `NMS`  
  - Double threshold  
  - Hysteresis

This matches the standard five‑stage Canny pipeline: Gaussian smoothing → gradient computation → non‑maximum suppression → double threshold → edge tracking by hysteresis.

## Key Learnings

- **Hysteresis connects weak edges to strong edges.**  
  Weak gradient responses are retained only if they are connected `via 8‑connected paths` to strong edge pixels; otherwise they are suppressed as noise. This stabilises edge maps across varying contrast.

- **Sliding windows for efficient neighbourhood operations.**  
  Implementing Sobel, NMS, and hysteresis as sliding‑window operations allows you to reuse local computations and enforce consistent neighbourhood definitions across the image.

- **Ratios for adaptive thresholding.**  
  Canny commonly uses thresholds defined as fractions of the maximum gradient magnitude `e.g., high threshold at 0.2–0.3 of max, low threshold at a smaller fraction of the high threshold`, which adapts to overall image contrast.

## Performance

- Match rate vs OpenCV: **quantitative validation against OpenCV is pending** `not yet recorded`  
- Qualitative tests: works on synthetic patterns such as checkerboards, intensity gradients, and simple shapes `circles`, which are standard sanity checks for edge detectors.

Once you measure agreement with a reference implementation `e.g., OpenCV’s `Canny`` on standard images, you can add quantitative metrics such as pixel‑wise agreement or F1 on labelled edge maps.

---

# Geometric Transformations

## Implementation

- Location: `code/classical_cv/transforms.py`  
- **Affine transform:** 2×3 matrix; 3 non‑collinear point correspondences are sufficient to solve for the parameters.  
- **Perspective `projective` transform:** 3×3 homography; typically solved from 4 point correspondences in general position.  
- **Bilinear interpolation** used when sampling non‑integer locations to obtain smooth warps.

Affine transforms include translation, rotation, scaling, and shear; perspective transforms additionally model convergence of parallel lines and foreshortening.

## Results

- Affine matrix numerical error: ~\`5 \times 10^{-14}\` `relative to analytical solution or library reference`.  
- Perspective matrix numerical error: ~\`1 \times 10^{-12}\`.  
- Visual inspection indicates warps that match OpenCV’s results for the same control points.

These error levels are consistent with double‑precision solutions of small linear systems and indicate a correct implementation.

## Key Learnings

- **Inverse mapping prevents holes.**  
  For image warping, mapping from output pixels back to source coordinates `inverse warping` avoids gaps `“holes”` that occur when pushing source pixels forward onto a discrete output grid.

- **Homogeneous coordinates for projective geometry.**  
  Representing pixels as \`(x, y, 1`\) and transformations as 3×3 matrices `or 2×3 for affine` unifies translation, rotation, scaling, shear, and perspective into simple matrix multiplication.

- **SVD solves overconstrained systems.**  
  When you have more than the minimum number of point correspondences, using least‑squares with SVD yields a robust estimate of the transform that balances noise across all points.

---

# Unit Tests

## Coverage

- Filters: Gaussian blur, Sobel operator, generic convolution.  
- Edge detection: non‑maximum suppression, thresholding, hysteresis, full Canny pipeline.  
- Transforms: affine, perspective, rotation, resize `including boundary behaviour`.

## Test Results

- 32 tests pass.  
- Edge cases covered explicitly:  
  - Shape preservation `output dimensions correct`.  
  - Identity transforms `no‑op operations behave correctly`.  
  - Boundary conditions for padding and interpolation.  
  - Value ranges `no unexpected clipping or overflow`.

## Run Tests

```bash
pytest test_unit.py -v
```

This test suite gives you a solid regression harness for future refactors `e.g., optimisations, vectorisation, or GPU ports` while ensuring classical CV behaviour remains correct.
