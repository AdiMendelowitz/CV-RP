# Computer Vision Research Portfolio

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-38%20passed-brightgreen.svg)](computer-vision-foundations/code/tests/test_core.py)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Implementations of core computer vision systems , from classical image processing through deep residual networks, Vision Transformers, self-supervised representation learning, model compression, and object detection. Every component is built from mathematical first principles against published papers and validated against reported benchmarks.

---

## Results

### Classification (trained from scratch on CIFAR-10 / MNIST)

| Model | Task | Accuracy | Params | Notes                               |
|-------|------|----------|--------|-------------------------------------|
| NumPy CNN | MNIST digit classification | **90.94%** | ~13K | No frameworks , pure NumPy backprop |
| ResNet-18 | CIFAR-10 classification | **93.43%** | 11.2M | Matches published benchmark         |
| ViT-Tiny | CIFAR-10 classification | **86.70%** | 5.36M | 47.7% fewer params than ResNet-18   |
| SimCLR | CIFAR-10 linear evaluation | **68.23%** | — | Unsupervised pretraining, no labels |

### Architecture comparison (linear probe, ImageNet pretrained weights)

| Model | Top-1 Acc | Params | Inference |
|-------|-----------|--------|-----------|
| ResNet-18 | 83.83% | 11.2M | 26.8 ms/img |
| ViT-Tiny | 80.72% | 5.5M | 26.9 ms/img |
| EfficientNet-B0 | 90.06% | 4.0M | 29.2 ms/img |
| ConvNeXt-Tiny | **95.08%** | 27.8M | 90.6 ms/img |

Protocol: frozen backbone + logistic regression on CIFAR-10, CPU, batch=1.

### Model compression (ResNet-18, CIFAR-10)

| Method | Accuracy gain/drop | Notes |
|--------|-------------------|-------|
| Knowledge distillation (SmallCNN student) | +4% over unguided baseline | T=4, α=0.7, 22x parameter reduction |
| L1 unstructured pruning (20%) | < 1% drop | Fine-tuned 5 epochs post-pruning |
| Static INT8 quantization | < 1% drop | >2x size reduction |

### Object detection (YOLOv8n, PCB defect detection)

| Metric | Value |
|--------|-------|
| mAP@0.5 | **0.9896** |
| mAP@0.5:0.95 | 0.6025 |
| Precision | 0.9769 |
| Recall | 0.9837 |

6-class industrial defect detection, 8,001 images, Tesla T4.

---

## Repository Structure

```
ml-research-12weeks/
├── computer-vision-foundations/         # Classification: CNN, ResNet-18, ViT-Tiny, SimCLR
│   ├── code/
│   │   ├── classical_cv/                # NumPy: convolution, Canny edges, geometric transforms
│   │   ├── cnn_scratch/                 # NumPy CNN with manual backprop (90.94% MNIST)
│   │   ├── pytorch_cnn/                 # ResNet-18 from scratch (93.43% CIFAR-10)
│   │   ├── vision_transformers/         # ViT-Tiny from scratch (86.70% CIFAR-10)
│   │   ├── self_supervised_learning/    # SimCLR (68.23% CIFAR-10 linear eval)
│   │   └── tests/                       # 38 unit tests, CPU-only, synthetic data
│   └── notes/                           # Paper summaries and architecture notes
├── Advanced CV & Efficient Models/      # Efficient architectures, compression, detection
│   ├── code/
│   │   ├── efficient_architectures/     # EfficientNet-B0, ConvNeXt-Tiny from scratch
│   │   ├── compression/                 # Knowledge distillation, quantization, pruning
│   │   └── detection/                   # IoU/NMS/mAP from scratch + YOLOv8 fine-tuning
│   ├── experiments/                     # Benchmark results and analysis
│   └── notes/                           # Paper summaries
└── data/                                # Shared dataset cache (CIFAR-10)
```

---

---

## Modules

### Computer Vision Foundations

[Full documentation](computer-vision-foundations/README.md)

Classical CV implements Gaussian blur, Sobel gradients, Canny edge detection (5-stage pipeline), and geometric transforms in NumPy, verified against OpenCV. The CNN from scratch covers convolution, pooling, dense layers, ReLU, softmax, cross-entropy, and SGD with momentum derived and coded manually without any framework, reaching **90.94% on MNIST**. ResNet-18 follows He et al. (2015) with BasicBlock skip connections and 200-epoch cosine annealing, reaching **93.43% on CIFAR-10**. ViT-Tiny follows Dosovitskiy et al. (2020) with Pre-LN architecture, AdamW with selective weight decay, MixUp/CutMix augmentation, and attention rollout visualization, reaching **86.70% on CIFAR-10**. SimCLR follows Chen et al. (2020) with NT-Xent contrastive loss and a two-view augmentation pipeline; linear evaluation on frozen representations reaches **68.23% on CIFAR-10**.

### Efficient Architectures and Model Compression


EfficientNet-B0 (Tan and Le, 2019) implements compound scaling across depth, width, and resolution via a single coefficient, with Squeeze-and-Excitation and MBConv blocks. ConvNeXt-Tiny (Liu et al., 2022) uses a transformer-inspired ConvNet design with depthwise 7x7 convolutions, inverted-bottleneck MLP, layer scale, and stochastic depth, reaching **95.08%** in the linear probe benchmark.

Knowledge distillation (Hinton et al., 2015) trains a SmallCNN student (170K params, 3 conv blocks) against a frozen ResNet-18 teacher using combined soft-target KL and hard-label CE loss. At 65.6x compression the student reaches 78.33%; the CE-only baseline reaches 78.97%, consistent with the capacity-gap analysis: at this compression ratio the student cannot fully absorb the soft-target structure from the teacher.

Post-training static INT8 quantization achieves 3.95x size reduction and 1.52x speedup with no measurable accuracy cost and no retraining. L1 unstructured pruning at 40% sparsity with 5-epoch fine-tuning recovers to within 0.16pp of the dense baseline; runtime is unchanged without sparse kernel support.

### Object Detection

Detection metrics (`compute_iou`, `non_max_suppression`, `compute_ap` with 11-point PASCAL VOC 2007 interpolation, `compute_map`) are implemented in NumPy and unit-tested. YOLOv8n fine-tuned on a PCB defect dataset (6 classes, 8,001 images, 50 epochs) reaches mAP@0.5 = 0.9896 with all six classes above 0.984 AP@0.5.

---

## Setup

```bash
# Activate virtual environment (Windows)
source .venv/Scripts/activate

# Install dependencies
pip install -e .

# Run tests
pytest computer-vision-foundations/code/tests/test_core.py -v
# 38 passed

# Train ResNet-18 on CIFAR-10
python "computer-vision-foundations/code/pytorch_cnn/train_cifar.py"

# SimCLR pretraining and linear evaluation
python "computer-vision-foundations/code/self_supervised_learning/train_simclr.py"
python "computer-vision-foundations/code/self_supervised_learning/linear_eval.py"

# Knowledge distillation
python "Advanced CV & Efficient Models/code/compression/train_distillation.py"

# Architecture linear probe benchmark
python "Advanced CV & Efficient Models/experiments/benchmark_architecture.py"
```

All data scripts resolve to the shared `data/` directory at repo root via absolute paths anchored to `__file__`, with no dependency on the working directory.

---

## References

- He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- Tan, M. and Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., and Xie, S. (2022). A ConvNet for the 2020s. CVPR. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
- Hinton, G., Vinyals, O., and Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- Han, S., Pool, J., Tran, J., and Dally, W. (2015). Learning both Weights and Connections for Efficient Neural Networks. NeurIPS. [arXiv:1506.02626](https://arxiv.org/abs/1506.02626)
- Redmon, J., Divvala, S., Girshick, R., and Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. CVPR. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
- Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., and Zisserman, A. (2010). The PASCAL Visual Object Classes Challenge. IJCV, 88(2), 303-338.
- Canny, J. (1986). A Computational Approach to Edge Detection. IEEE TPAMI, 8(6), 679-698.