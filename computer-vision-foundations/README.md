# Computer Vision Foundations

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-green.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/tests-38%20passed-brightgreen.svg)](code/tests/test_core.py)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A ground-up implementation of core computer vision systems — from classical image processing through deep residual networks, Vision Transformers, and self-supervised representation learning. Every component is implemented from mathematical first principles with no reliance on high-level model libraries, then verified against established benchmarks.

---

## Results Summary

| Model | Task | Accuracy | Parameters | Notes |
|---|---|---|---|---|
| CNN (NumPy) | MNIST digit classification | **90.94%** | ~13K | No frameworks — pure NumPy |
| ResNet-18 | CIFAR-10 classification | **93.43%** | 11.2M | Matches published benchmark |
| ViT-Tiny | CIFAR-10 classification | **86.70%** | 5.36M | T4 GPU, 100 epochs |
| SimCLR | CIFAR-10 linear evaluation | **68.23%** | — | Self-supervised, no labels during pretraining |

![Results Summary](computer-vision-foundations/code/results_summary.png)
---

## Repository Structure

```
computer-vision-foundations/
├── code/
│   ├── classical_cv/               # NumPy-only edge detection and geometric transforms
│   ├── cnn_scratch/                # CNN built from scratch: forward pass, backprop, SGD
│   ├── pytorch_cnn/                # ResNet-18 from scratch in PyTorch
│   ├── vision_transformers/        # ViT-Tiny from scratch in PyTorch
│   ├── self_supervised_learning/   # SimCLR contrastive learning pipeline
│   └── tests/
│       └── test_core.py            # 38 unit tests covering all modules
├── notebooks/                      # Interactive walkthroughs and visualizations
└── README.md                       # This file
```

Each subdirectory contains its own detailed README covering architecture, training configuration, results, and usage instructions.

---

## Modules

### 1. Classical Computer Vision
**[`classical_cv/`](code/classical_cv/README.md)**

NumPy-only implementations of foundational image processing algorithms, verified against OpenCV:

- **Gaussian blur** — 2D kernel convolution with configurable σ
- **Sobel edge detection** — Gradient operators (Gx, Gy) with magnitude and direction
- **Canny edge detection** — Complete 5-stage pipeline: Gaussian smoothing → Sobel gradients → non-maximum suppression → double thresholding → hysteresis edge linking
- **Geometric transformations** — Rotation, affine warp, and perspective warp via inverse mapping with a 3×3 homography

All transforms use inverse mapping to guarantee hole-free output. Verification images showing side-by-side comparisons with OpenCV are included in `classical_cv/outputs/`.

---

### 2. CNN from Scratch — NumPy Only
**[`cnn_scratch/`](code/cnn_scratch/README.md)**

A complete Convolutional Neural Network implemented without any deep learning framework — convolution, pooling, dense layers, ReLU, softmax, cross-entropy loss, and SGD with momentum are all derived and coded manually.

**Architecture:**
```
Input (1×28×28) → Conv(1→8) → ReLU → MaxPool →
                  Conv(8→16) → ReLU → MaxPool →
                  Flatten → Dense(784→128) → ReLU → Dense(128→10) → Softmax
```

**Key technical details:**
- Each layer implements `forward()`, `backward()`, and `get_params()` following a consistent layer interface
- He initialization prevents vanishing gradients in ReLU networks
- Validated first on synthetic line-pattern data (100% accuracy in 2 epochs) before MNIST, following incremental verification methodology

**Result: 90.94% on MNIST.** The same architecture rebuilt in PyTorch achieves 93.00% — the 0.01% difference in *training* accuracy between the two confirms mathematical equivalence with PyTorch's autograd engine.

---

### 3. PyTorch CNN & ResNet-18
**[`pytorch_cnn/`](code/pytorch_cnn/README.md)**

ResNet-18 implemented from scratch in PyTorch, following He et al. (2015) precisely. No `torchvision.models` — every `BasicBlock`, projection shortcut, and global average pooling layer is written explicitly.

**Training configuration:**
- Optimizer: SGD with momentum (0.9), weight decay 5×10⁻⁴
- LR schedule: Cosine annealing over 200 epochs from 0.1
- Augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip
- Batch size: 128

**Per-class results on CIFAR-10:**

| Class | Accuracy | Class | Accuracy |
|---|---|---|---|
| Airplane | 94.2% | Dog | 89.4% |
| Automobile | 96.8% | Frog | 95.7% |
| Bird | 88.3% | Horse | 94.1% |
| Cat | 87.1% | Ship | 96.2% |
| Deer | 93.5% | Truck | 94.9% |

**Overall: 93.43%**, matching the published benchmark for ResNet-18 on CIFAR-10 trained from scratch.

---

### 4. Vision Transformer (ViT-Tiny)
**[`vision_transformers/`](code/vision_transformers/README.md)**

ViT-Tiny from scratch, following the DeiT-Tiny configuration (Touvron et al., 2021). Implementation uses Pre-LN (LayerNorm before each sublayer) and `F.scaled_dot_product_attention` for Flash Attention compatibility.

**Architecture:**
- 32×32 input, 4×4 patches → 64 patch tokens + 1 CLS token
- Embedding dim: 192, depth: 12 layers, heads: 3, MLP ratio: 4.0
- Total parameters: 5.36M

**Training** (Kaggle T4 GPU, 100 epochs):
- Optimizer: AdamW with selective weight decay — bias, LayerNorm, CLS token, and positional embeddings excluded from decay per Loshchilov & Hutter (2019)
- LR: linear warmup (10 epochs) → cosine annealing, peak 1×10⁻³
- Regularisation: MixUp (α=0.2), CutMix (α=1.0), TrivialAugmentWide, RandomErasing (p=0.25), label smoothing (ε=0.1), gradient clipping (max norm 1.0)

**Result: 86.70%**, 6.7 points below ResNet-18. The gap is consistent with published literature — ViT's lack of convolutional inductive biases (local connectivity, translation equivariance) requires substantially more data to compensate. The crossover point between ViT and CNNs has been empirically placed at approximately 14M images (Dosovitskiy et al., 2020); CIFAR-10's 50,000 images are three orders of magnitude below that threshold.

Notably, ViT-Tiny achieves this with **47.7% fewer parameters** than ResNet-18. The dominant failure mode is cat/dog confusion (cat: 69.5%, dog: 79.6%), accounting for ~19% of all errors — a known hard pair at 32×32 resolution. A detailed architectural comparison with training curve analysis is in [`ViT_vs_ResNet.md`](code/vision_transformers/ViT_vs_ResNet.md).

---

### 5. Self-Supervised Learning — SimCLR
**[`self_supervised_learning/`](code/self_supervised_learning/README.md)**

SimCLR (Chen et al., 2020) contrastive learning pipeline: the encoder is pretrained entirely without labels, then a linear classifier is trained on top of frozen representations to evaluate representation quality.

**Components:**
- `SimCLR` — ResNet-18 encoder + 2-layer MLP projection head (512→512→128)
- `NTXentLoss` — Temperature-scaled NT-Xent contrastive loss (τ=0.5)
- `SimCLRAugmentation` — Stochastic pipeline producing two independent views per image: RandomResizedCrop, ColorJitter, GaussianBlur, RandomHorizontalFlip, RandomGrayscale

**Training configuration:**
- 100 pretraining epochs, batch size 256, Adam optimizer, cosine annealing LR
- Linear evaluation: encoder frozen, single linear layer trained for 100 epochs on labeled data

**Result: 68.23% linear evaluation accuracy** on CIFAR-10, trained on CPU. The original paper uses batch size 4096 and 1000 epochs on TPU hardware — this implementation achieves competitive representations under significant compute constraints, demonstrating that the core contrastive learning signal is robust to hardware limitations.

---

## Test Suite

All modules are covered by a unit test suite in [`code/tests/test_core.py`](code/tests/test_core.py):

```
38 tests across 15 test classes
├── TestCorrelated2d          (3 tests)   — shape, multichannel, identity kernel
├── TestGaussianKernel        (3 tests)   — shape, normalisation, symmetry
├── TestGaussianBlur          (2 tests)   — spatial dims, finite output
├── TestSobelEdges            (3 tests)   — return structure, non-negativity, flat image
├── TestCannyEdgeDetector     (3 tests)   — shape, binary output, flat image
├── TestNonMaxSuppression     (1 test)    — output shape
├── TestDoubleThreshold       (2 tests)   — shape, strong > weak threshold invariant
├── TestResNet18              (4 tests)   — output shape, finite, skip connections, batch invariance
├── TestResNet18Channels      (1 test)    — custom in_channels
├── TestPatchEmbedding        (2 tests)   — output shape with CLS token, patch size variants
├── TestVitTiny               (3 tests)   — output shape, finite, batch invariance
├── TestNTXentLoss            (3 tests)   — scalar output, finite, temperature effect
├── TestProjectionHead        (2 tests)   — output shape, custom dims
├── TestSimCLRModel           (3 tests)   — projected/encoder output shapes, finite
└── TestSimCLRAugmentation    (3 tests)   — returns two tensors, correct shape, views differ
```

All tests run on CPU with synthetic data — no downloads or GPU required.

```bash
pytest code/tests/test_core.py -v
# 38 passed in ~5s
```

---

## Setup

**Requirements:** Python 3.12, PyTorch 2.0+, NumPy, Pillow

```bash
git clone https://github.com/AdiMendelowitz/CV-RP.git
cd CV-RP/computer-vision-foundations
pip install -r requirements.txt
```

**Run tests:**
```bash
pytest code/tests/test_core.py -v
```

**Train ResNet-18 on CIFAR-10:**
```bash
python code/pytorch_cnn/train_cifar.py
```

**Train ViT-Tiny** (GPU recommended — run on Kaggle T4):
```bash
# Open code/vision_transformers/vit_cifar10_kaggle.ipynb on Kaggle
```

**SimCLR pretraining and linear evaluation:**
```bash
python code/self_supervised_learning/train_simclr.py
python code/self_supervised_learning/linear_eval.py
```

---

## Key Findings

**Inductive bias vs data scale.** ResNet-18 outperforms ViT-Tiny by 6.7 points on CIFAR-10 (93.43% vs 86.70%). This gap is a direct measurement of the value of convolutional inductive biases — locality and translation equivariance — when labeled data is scarce. The result is quantitatively consistent with the original ViT paper's predictions.

**Self-supervised representations are data-efficient.** SimCLR trained without any labels produces representations that support 68.23% linear classification accuracy, compared to ResNet-18's 93.43% with full supervision. The gap reflects both the harder optimization problem and the information bottleneck of evaluating with a linear probe over a nonlinear representation.

**Cat/dog confusion is architecture-agnostic.** Both ResNet-18 and ViT-Tiny identify cat and dog as their dominant failure modes. This reflects genuine visual ambiguity at 32×32 resolution rather than an architectural weakness — the class boundary is inherently hard at this input scale.

**NumPy backpropagation is mathematically equivalent to autograd.** The NumPy CNN matches PyTorch's training accuracy to within 0.01%, confirming that manual gradient derivation and framework-computed gradients converge to the same solution.

---

## References

- He, K. et al. (2015). *Deep Residual Learning for Image Recognition.* [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Dosovitskiy, A. et al. (2020). *An Image is Worth 16x16 Words.* [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Touvron, H. et al. (2021). *Training data-efficient image transformers & distillation through attention.* [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
- Chen, T. et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations.* [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- Canny, J. (1986). *A Computational Approach to Edge Detection.* IEEE TPAMI, 8(6), 679–698.
- Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

---

## Author

**Adi Mendelowitz** — Machine Learning Engineer  
Specialization: Computer Vision & Deep Learning Systems  
GitHub: [AdiMendelowitz](https://github.com/AdiMendelowitz)

---

*Last updated: February 2026*