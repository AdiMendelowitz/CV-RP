# Quantization & Pruning

**References:**
- Jacob et al. (2018) *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* (https://arxiv.org/abs/1712.05877)
- Nagel et al. (2021) *A White Paper on Neural Network Quantization* (https://arxiv.org/abs/2106.08295)
- Han et al. (2015) *Deep Compression* (https://arxiv.org/abs/1510.00149)
- PyTorch Quantization docs https://pytorch.org/docs/stable/quantization.html

---

## 1. Why Quantization

Full-precision (FP32) models are expensive to deploy. A standard ResNet-50 holds ~25M parameters at 4 bytes each, totalling 100 MB just for weights. Quantization reduces numerical precision to shrink model size and accelerate inference, with the goal of preserving accuracy within a tolerable margin.

The core tradeoff: lower bit-width compresses the value range that can be represented. Representable values are spaced further apart, so rounding errors accumulate across layers. How much accuracy degrades depends on the bit-width chosen, the calibration method, and how sensitive the model's activations are to rounding.

---

## 2. INT8 vs FP32

| Property              | FP32                          | INT8                                |
|-----------------------|-------------------------------|-------------------------------------|
| Bits per value        | 32                            | 8                                   |
| Memory (weights)      | Baseline                      | ~4x reduction                       |
| Representable range   | +/- 3.4e38 (floating)        | -128 to 127 (signed) / 0-255 (unsigned) |
| Precision             | ~7 decimal digits             | ~2-3 decimal digits                 |
| Typical accuracy drop | -                             | 0.1-1% top-1 on ImageNet (well-calibrated) |
| Inference speedup     | Baseline                      | 2-4x on supported hardware          |
| Hardware support      | Universal                     | NVIDIA TensorRT, ARM NEON, Intel VNNI, ONNX Runtime |

FP16 sits between the two: half the memory of FP32 with negligible accuracy loss, no calibration required, and supported natively on NVIDIA GPUs. INT8 is preferred for edge/mobile deployment where memory bandwidth and power are constrained.

The 2-4x speedup is not automatic. It requires hardware with INT8 matrix-multiply units (e.g., NVIDIA's DP4A instruction on Turing/Volta, ARM's SDOT). On hardware without native INT8 support, quantized inference can actually be slower due to dequantization overhead.

---

## 3. The Quantization Map

Quantization maps a floating-point value `x_float` to an integer `x_int` via:

```
x_int   = clamp(round(x_float / scale) + zero_point,  q_min,  q_max)
x_float = scale * (x_int - zero_point)          # dequantization
```

`scale` and `zero_point` are per-tensor or per-channel constants determined during calibration. Together they define the affine mapping between the integer grid and the real-valued range. The choice of these two values defines whether the scheme is **symmetric** or **asymmetric**.

---

## 4. Symmetric vs Asymmetric Quantization

### Symmetric

The quantization range is centered at zero: `[-a, a]` maps to `[-(2^(b-1)-1), 2^(b-1)-1]` for signed INT8.

```
scale      = a / (2^(b-1) - 1)
zero_point = 0
```

Because `zero_point = 0`, the multiply-accumulate operation in a linear layer carries no zero-point correction term. This makes symmetric quantization faster in hardware and simpler to implement.

**Best for:** weights, which typically have near-symmetric distributions around zero (e.g., Conv2D, Linear layers after batch normalisation folding).

**Weakness:** wastes range when the true distribution is asymmetric. If activations live in `[0, 6]` (post-ReLU), a symmetric range of `[-6, 6]` uses half its representable integers for negative values that never appear.

### Asymmetric

The range `[x_min, x_max]` maps to `[0, 2^b - 1]` for unsigned INT8.

```
scale      = (x_max - x_min) / (2^b - 1)
zero_point = round(-x_min / scale)          # an integer offset
```

The zero-point shifts the integer grid to align with the actual value distribution, so all 256 levels are used efficiently. This matters most for activations, which are often strictly non-negative after ReLU.

**Best for:** activations, which commonly have asymmetric distributions.

**Cost:** each multiply-accumulate in a linear layer carries a zero-point correction term, adding a small amount of compute overhead. In practice this overhead is negligible compared to the accuracy benefit from better range utilisation.

### In Practice

PyTorch's default quantization uses **symmetric for weights** and **asymmetric (unsigned INT8) for activations**, consistent with Jacob et al. (2018) and the ONNX quantization spec.

---

## 5. Calibration Process

Calibration determines `scale` and `zero_point` for each tensor in the model. It only applies to **Post-Training Quantization (PTQ)**; in QAT, fake-quantization nodes learn these parameters during training.

### Steps

1. **Collect a representative calibration dataset.** Typically 100-1000 samples drawn from the training distribution. This is not a held-out test set; its purpose is solely to profile activation statistics. Using an unrepresentative subset (e.g., only one class) produces poor calibration.

2. **Forward pass in FP32.** Run the calibration data through the full-precision model and record the distribution of every weight tensor and activation tensor that will be quantized.

3. **Determine the clipping range.** Choose `[x_min, x_max]` for each tensor using one of the methods below.

4. **Compute `scale` and `zero_point`** from the chosen range.

5. **Optional: evaluate accuracy** on a validation set to confirm calibration quality before converting.

### Calibration Methods

| Method          | How it works                                                   | Notes                                          |
|-----------------|----------------------------------------------------------------|------------------------------------------------|
| **Min-max**     | `x_min = min(activations)`, `x_max = max(activations)`        | Simple; sensitive to outliers                  |
| **Percentile**  | Clip at e.g. 99.99th percentile to discard outliers            | More robust; percentile is a hyperparameter    |
| **Entropy (KL)**| Minimise KL-divergence between FP32 and INT8 distributions     | Used by TensorRT; tends to give best accuracy  |
| **MSE**         | Minimise mean-squared error between FP32 and quantized values  | Strong for weights; computationally heavier    |

Min-max is the easiest to implement and is the default in PyTorch's `MinMaxObserver`. Entropy/KL calibration (TensorRT's default) consistently produces better results at the cost of slightly more calibration compute.

### PyTorch Calibration Skeleton

```python
import torch
from torch.quantization import prepare, convert, get_default_qconfig

model.eval()
model.qconfig = get_default_qconfig("x86")   # or "qnnpack" for ARM
prepare(model, inplace=True)                   # inserts observers

with torch.no_grad():
    for x, _ in calibration_loader:           # ~100-1000 samples
        model(x)

convert(model, inplace=True)                   # replaces observers with quantized ops
```

After `prepare`, each observer accumulates statistics during calibration forward passes. `convert` then reads those statistics to compute `scale` and `zero_point` and replaces FP32 ops with their INT8 equivalents.

---

## 6. Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)

PTQ requires no retraining and is practical when a pre-trained checkpoint is available. For most classification models, PTQ with good calibration holds within 1% of FP32 accuracy.

QAT inserts **fake-quantization nodes** (quantize then immediately dequantize, still in FP32) during training. The model learns to be robust to quantization noise. This recovers accuracy in cases where PTQ degrades too much, particularly for smaller models (MobileNet-class), low bit-widths (INT4), or tasks with fine-grained predictions (detection, segmentation).

Rule of thumb: start with PTQ and entropy calibration. If accuracy degrades more than 1%, switch to QAT.

---

## 7. Per-Tensor vs Per-Channel Quantization

A single `scale` per entire weight tensor (per-tensor) is fast but imprecise when filter magnitudes vary widely across channels, which they always do. Per-channel quantization assigns an independent `scale` to each output channel of a Conv or Linear layer, dramatically reducing quantization error on weights at negligible runtime cost (scales are absorbed into bias during inference). PyTorch and TensorRT both default to per-channel for weights.

Activations are typically quantized per-tensor because their channel-wise statistics change with each input, making per-channel calibration impractical without QAT.

---

## 8. Key Numbers to Remember

- INT8 quantization typically achieves 2-4x memory reduction and 2-4x latency improvement on supported hardware.
- Entropy/KL calibration on a 1000-sample dataset typically brings PTQ accuracy within 0.5% of FP32 for ResNet/EfficientNet class models (Nagel et al., 2021).
- Per-channel weight quantization alone recovers a significant portion of per-tensor accuracy loss at no inference overhead.
- Below INT8 (e.g., INT4), PTQ accuracy degrades sharply and QAT or more advanced techniques (GPTQ, SmoothQuant) become necessary.