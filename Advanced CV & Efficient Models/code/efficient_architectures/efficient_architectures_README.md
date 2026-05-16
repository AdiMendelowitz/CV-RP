# Efficient Architectures

PyTorch implementations of EfficientNet and ConvNeXt, plus a receptive field
analysis tool that compares theoretical and empirical RF across both architectures.

All architectures are built from `nn.Module` primitives with no use of `torchvision.models`.

---

## Files

| File | Description |
|---|---|
| `efficientnet.py` | EfficientNet-B0, B1, B4 |
| `convnext.py` | ConvNeXt-Tiny, Small, Base |
| `receptive_field_analysis.py` | Theoretical and empirical RF measurement |

---

## EfficientNet

**Reference:** Tan, M. and Le, Q. V. Rethinking Model Scaling for Convolutional Neural
Networks. ICML 2019. https://arxiv.org/abs/1905.11946

### Design

EfficientNet scales network width, depth, and input resolution jointly using compound
coefficients (phi_w, phi_d, phi_r) derived from a neural architecture search baseline
(B0). All variants share the same block structure; only the number of channels, number
of block repetitions per stage, and input resolution change.

The fundamental building block is MBConv (Mobile Inverted Bottleneck):

```
Input
  |-- 1x1 pointwise conv (expand channels by expand_ratio; omitted when ratio=1)
  |-- BN + SiLU
  |-- kxk depthwise conv (k=3 or 5, stride as specified per stage)
  |-- BN + SiLU
  |-- Squeeze-Excitation (se_ratio=0.25)
  |-- 1x1 pointwise conv (project back to out_channels)
  |-- BN
  `-- residual add (only when stride=1 and in_channels == out_channels)
```

Squeeze-Excitation applies a channel-wise attention gate: global average pool to a scalar
per channel, then a two-layer MLP bottleneck (se_ratio=0.25) that produces per-channel
multipliers via Sigmoid. The gate is applied by element-wise multiplication.

Stochastic depth (drop path) is applied linearly from 0.0 at the first block to 0.2 at
the last, following the schedule in the original paper.

### Variants

| Variant | Width mult. | Depth mult. | Dropout | Approx. Params |
|---|---|---|---|---|
| B0 | 1.0 | 1.0 | 0.2 | 5.3M |
| B1 | 1.0 | 1.1 | 0.2 | 7.8M |
| B4 | 1.4 | 1.8 | 0.4 | 19M |

### Usage

```python
from efficient_architectures.efficientnet import efficientnet_b0

model = efficientnet_b0(num_classes=1000)
```

---

## ConvNeXt

**Reference:** Liu, Z. et al. A ConvNet for the 2020s. CVPR 2022.
https://arxiv.org/abs/2201.03545

### Design

ConvNeXt starts from a ResNet-50 and applies a series of training and architecture
modifications to close the gap with vision transformers, each independently motivated.
The result is a pure ConvNet that matches or exceeds Swin Transformer accuracy with
lower inference complexity.

**ConvNeXt block:**

```
Input (C channels, H x W)
  |-- 7x7 depthwise conv (groups=C)
  |-- LayerNorm2d (channel-wise, NCHW format)
  |-- 1x1 pointwise conv (expand to 4*C)
  |-- GELU
  |-- 1x1 pointwise conv (project back to C)
  |-- multiply by layer_scale (per-channel learnable scalar, init=1e-6)
  `-- residual add (with stochastic depth on the block output)
```

Key differences from ResNet:
- 7x7 depthwise convolution replaces 3x3 standard convolution (larger receptive field
  per parameter, matching the window size in Swin Transformer).
- Layer Normalization replaces Batch Normalization (normalization per sample, not per
  batch; implemented as LayerNorm2d by permuting to NHWC and back).
- GELU replaces ReLU throughout.
- Inverted bottleneck (expand-then-contract) replaces standard bottleneck.
- Layer scale initializes residual branch outputs near zero to stabilize training.
- Stochastic depth on the residual branch (drop_path_rate increases with depth).

The stem uses a non-overlapping 4x4 stride-4 patchify convolution, matching ViT's
patch embedding. Downsampling between stages uses LayerNorm followed by a 2x2 stride-2
convolution.

### Variants

| Variant | Stage depths | Stage dims | Drop path rate | Approx. Params |
|---|---|---|---|---|
| Tiny | 3, 3, 9, 3 | 96, 192, 384, 768 | 0.1 | 28M |
| Small | 3, 3, 27, 3 | 96, 192, 384, 768 | 0.4 | 50M |
| Base | 3, 3, 27, 3 | 128, 256, 512, 1024 | 0.5 | 89M |

### Usage

```python
from efficient_architectures.convnext import convnext_tiny

model = convnext_tiny(num_classes=1000)
```

---

## Receptive Field Analysis

**Source:** `receptive_field_analysis.py`

The script computes both theoretical and empirical receptive fields for EfficientNet-B0
and ConvNeXt-Tiny, then produces a side-by-side comparison plot
(`receptive_field_comparison.png`).

### Theoretical RF

The theoretical RF at each layer is computed from the recurrence:

```
RF_new = RF_prev + (kernel_size - 1) * cumulative_stride
```

where `cumulative_stride` is the product of all strides from the input to the current
layer. This formula assumes no padding loss and gives an upper bound on the spatial
region that can influence a single output neuron.

Final theoretical RFs on a 224x224 input:
- EfficientNet-B0: 1003 px (448% of input width, across 16 spatial layers)
- ConvNeXt-Tiny: 1421 px (634% of input width, across 24 spatial layers)

Both architectures have theoretical RFs that exceed the input size, meaning every output
neuron has the potential to be influenced by the entire input. The larger RF of ConvNeXt
reflects its 7x7 depthwise convolutions: each such layer contributes
(7 - 1) * cumulative_stride pixels to the RF, versus (3 - 1) * cumulative_stride for
EfficientNet's 3x3 depthwise layers.

### Empirical RF

The empirical RF is measured via gradient attribution: for each hook point, a fresh
input is constructed (seed 0, scaled by 0.1), a forward pass is run, the gradient is
computed with respect to the activation at the center neuron, and the spatial extent of
non-negligible gradients (magnitude > peak * 1e-3) is measured as a bounding box.

This gives the actual region that the center neuron responds to in practice, which is
typically smaller than the theoretical upper bound because early-layer gradients decay
rapidly with distance from the center.

### Running the Analysis

```bash
python "Advanced CV & Efficient Models/code/efficient_architectures/receptive_field_analysis.py"
```

Output: `receptive_field_comparison.png` written to the same directory.

---

## References

- Tan, M. and Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional
  Neural Networks. ICML 2019. https://arxiv.org/abs/1905.11946
- Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., and Xie, S. (2022).
  A ConvNet for the 2020s. CVPR 2022. https://arxiv.org/abs/2201.03545
- Howard, A. G. et al. (2017). MobileNets: Efficient Convolutional Neural Networks for
  Mobile Vision Applications. arXiv:1704.04861.
- Hu, J., Shen, L., and Sun, G. (2018). Squeeze-and-Excitation Networks. CVPR 2018.