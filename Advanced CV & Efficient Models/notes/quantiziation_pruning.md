# Quantization & Pruning

**References:**
- Jacob et al. (2018) *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* (https://arxiv.org/abs/1712.05877)
- Nagel et al. (2021) *A White Paper on Neural Network Quantization* (https://arxiv.org/abs/2106.08295)
- Han et al. (2015/2016) *Deep Compression* (https://arxiv.org/abs/1510.00149)
- PyTorch quantization docs (https://pytorch.org/docs/stable/quantization.html)

---

## 1. Why Quantization

Full-precision (FP32) models are expensive to deploy. A standard ResNet-50 has around 25.6M parameters; at 4 bytes per parameter this is roughly 100 MB of weight storage. Quantization reduces numerical precision (e.g., from 32-bit float to 8-bit integer) to shrink model size and accelerate inference, aiming to preserve accuracy within a small, acceptable margin.

Lower bit-width compresses the set of representable values, increasing the spacing between quantization levels; rounding errors then accumulate across layers. How much accuracy degrades depends on bit-width, calibration strategy, model architecture, and the sensitivity of activations to rounding.

---

## 2. INT8 vs FP32

| Property              | FP32                            | INT8                                         |
|-----------------------|----------------------------------|---------------------------------------------|
| Bits per value        | 32                              | 8                                           |
| Memory (weights)      | Baseline                        | ~4× reduction                               |
| Representable range   | ~±3.4e38                        | -128 to 127 (signed) / 0–255 (unsigned)     |
| Precision             | ~7 decimal digits               | about 2–3 decimal digits                    |
| Typical accuracy drop | –                               | ≈0–1% top‑1 on ImageNet (good PTQ/QAT)      |
| Inference speedup     | Baseline                        | ≈2–4× on hardware with INT8 units           |
| Hardware support      | Universal                       | TensorRT, ARM NEON, Intel VNNI, ONNX Runtime, TFLite |



FP16 sits in between: half the memory of FP32, often negligible accuracy loss, and native support on modern GPUs; it generally requires no calibration. INT8 is usually preferred for edge/mobile deployment where memory bandwidth and power are constrained, provided the hardware exposes efficient INT8 MAC instructions (e.g., NVIDIA DP4A / INT8 Tensor Cores, ARM NEON SDOT, Intel VNNI).

On hardware without native INT8 acceleration, quantized inference may be no faster—or even slower—once dequantization and conversion overheads are included.

---

## 3. The Quantization Map

A standard affine quantization map between float and integer values is:

```text
x_int   = clamp(round(x_float / scale) + zero_point, q_min, q_max)
x_float = scale * (x_int - zero_point)    # dequantization
```

- `scale` > 0: real-valued step size between adjacent integer levels.  
- `zero_point`: integer that should represent real value 0.  
- `q_min`, `q_max`: integer bounds (e.g., 0 and 255 for uint8).  

`scale` and `zero_point` may be defined **per tensor** (one pair per layer) or **per channel** (one pair per output channel for weights). The choice between symmetric and asymmetric schemes is determined by how these parameters are chosen.

---

## 4. Symmetric vs Asymmetric Quantization

### Symmetric

The quantization range is centred at zero. For signed INT8, we map `[-a, a]` to `[-(2^(b-1)-1), 2^(b-1)-1]`:

```text
scale      = a / (2^(b-1) - 1)
zero_point = 0
```

Because `zero_point = 0`, there is no zero-point correction term inside the MAC; the linear layer uses pure integer multiplications and accumulations, which simplifies hardware kernels.

**Best for:** weights, which often have approximately zero-centred distributions after BN folding.

**Weakness:** for non-negative or highly skewed distributions (e.g., ReLU activations in [0, 6]), half the symmetric representable range is wasted on values that never appear.

### Asymmetric

The range `[x_min, x_max]` maps to `[0, 2^b - 1]` for unsigned INT8:

```text
scale      = (x_max - x_min) / (2^b - 1)
zero_point = round(-x_min / scale)
```

The zero-point shifts the integer grid to align with the actual data range, using all levels efficiently. This is particularly beneficial for activations, which are often strictly non-negative after ReLU.

**Cost:** the MAC must incorporate a small zero-point correction term, adding a bit of extra arithmetic, though this is typically negligible compared to the core GEMM cost.

### In practice

Most modern toolchains adopt: **symmetric per-channel** quantization for weights and **asymmetric per-tensor** quantization for activations, matching Jacob et al. and the ONNX/TFLite specs.

---

## 5. Calibration Process

Calibration determines `scale` and `zero_point` for each quantized tensor in **post‑training quantization (PTQ)**. In QAT, these parameters are learned/updated during training via observers or fake‑quant nodes.

### Steps

1. **Collect a representative calibration dataset.**  
   Typically O(100–1000) samples drawn from the training distribution; the goal is to capture typical activation ranges, not to evaluate performance.

2. **Run FP32 forward passes.**  
   Execute the model in evaluation mode and record activation/weight statistics (min/max, histograms) for tensors that will be quantized.

3. **Choose clipping ranges.**  
   For each tensor, choose `[x_min, x_max]` using a calibration heuristic (min–max, percentile, KL, etc.).

4. **Compute `scale` and `zero_point`.**  
   Apply the selected quantization scheme (symmetric/asymmetric, per‑tensor/per‑channel).

5. **Convert and evaluate.**  
   Replace FP32 ops with quantized counterparts and evaluate accuracy to confirm that the drop is acceptable.

### Calibration methods

| Method         | How it works                                                 | Notes                                   |
|----------------|--------------------------------------------------------------|-----------------------------------------|
| Min–max        | Use global min and max of activations                        | Very simple; sensitive to outliers      |
| Percentile     | Clip at, e.g., 99.9–99.99th percentile                       | Robust; percentile is a tunable hyperparameter |
| KL / entropy   | Choose range minimising KL divergence between FP32 and INT8  | Used in TensorRT; often best accuracy   |
| MSE (L2)       | Choose range minimising mean‑squared error                   | Strong especially for weights; costlier |



Min–max is the default in PyTorch’s `MinMaxObserver`. Entropy/KL-based calibration is the default in TensorRT and generally yields better PTQ accuracy at modest extra cost.

### PyTorch calibration skeleton

```python
import torch
from torch.ao.quantization import prepare, convert, get_default_qconfig

model.eval()
model.qconfig = get_default_qconfig("x86")  # or "qnnpack" for ARM
prepare(model, inplace=True)

with torch.no_grad():
    for x, _ in calibration_loader:  # ~100–1000 samples
        model(x)

convert(model, inplace=True)
```

After `prepare`, observers accumulate activation statistics; `convert` uses them to compute scales/zero-points and swap in quantized kernels.

---

## 6. PTQ vs QAT

**Post‑Training Quantization (PTQ):**  

- Quantize a pre‑trained FP32 model using calibration only (no weight updates).  
- Works very well for many classification CNNs at 8‑bit when combined with good calibration and per‑channel weight quantization.  
- Typical ImageNet accuracy loss for ResNet/EfficientNet-scale models is often within ≈0–1 percentage point under strong PTQ.

**Quantization‑Aware Training (QAT):**  

- Insert fake‑quantization operators (quantize–dequantize in FP32) during training so that the model learns to be robust to quantization noise.  
- Necessary when PTQ drop is too large, e.g., compact models (MobileNet family), tasks beyond classification (detection, segmentation), or bit‑widths lower than 8 bits.

Practical strategy:

- Start with PTQ (with per‑channel weights and Entropy/MSE calibration).  
- If accuracy drop >1–2 percentage points, consider QAT or more advanced methods (SmoothQuant, GPTQ, etc. for transformers and LLMs).

---

## 7. Per-Tensor vs Per-Channel Quantization

**Per-tensor:** one `(scale, zero_point)` per tensor (e.g., per weight matrix). Fast and simple but vulnerable when different channels have very different magnitudes.

**Per-channel:** one `(scale, zero_point)` per output channel for Conv/Linear weights. This significantly reduces quantization error on weights and is now default in many libraries.

- For weights, per-channel adds negligible runtime overhead because scale factors can be folded into biases or scaling operations.  
- For activations, per-channel quantization is more complex since activation ranges depend on the input; PTQ typically uses per‑tensor activations, while QAT can support per‑channel activation quantization if the framework implements it.

Nagel et al. (2021) and Krishnamoorthi (2018) show that 8‑bit **per‑channel weight quantization** with per‑tensor activations can keep accuracy within ≈2% of FP32 across many CNNs using PTQ alone.

---

## 8. Key Numbers to Remember

- **Memory:** FP32 → INT8 gives ≈4× reduction in weight storage; overall model size is typically close to this (biases and scales are a small fraction).  
- **Latency:** 8‑bit integer inference can be ≈2–4× faster on hardware with dedicated INT8 units (Tensor Cores, VNNI, NEON, etc.). On CPUs without these, speedups may be modest.  
- **Accuracy:** Well-calibrated 8‑bit PTQ or QAT on standard CNNs usually stays within ≈1 percentage point of FP32 top‑1 accuracy.  
- **Below 8 bits:** 4‑bit PTQ often incurs large drops (10–14 points) without QAT or specialised schemes; robust 4‑bit or lower precision generally requires more advanced training and quantization methods.

---

## 9. Pruning (High-Level)

Although this note focuses on quantization, **pruning** is the complementary axis in Deep Compression:

- **Magnitude pruning:** remove weights below a threshold and retrain. Han et al. report ≈9× parameter reduction on AlexNet and ≈13× on VGG‑16 from pruning alone.  
- **Structured pruning:** remove entire channels/filters instead of individual weights to obtain dense, smaller models that accelerate naturally on standard hardware.

A realistic deployment pipeline combines **pruning + quantization** (often followed by simple entropy coding), achieving dramatic size and bandwidth reductions with minimal accuracy loss.

---

## 10. Practical Takeaways

- Use **symmetric per-channel** quantization for weights and **asymmetric per-tensor** for activations as a strong default.  
- For CNNs: start with **8‑bit PTQ with entropy/MSE calibration and per‑channel weights**; if that fails, move to QAT.  
- For MobileNet/EfficientNet-scale models and low bits (≤4‑bit), plan on QAT or newer methods expressly designed for low-bit regimes.  
- Combine pruning and quantization for maximum compression; ensure you retrain between steps to recover accuracy.
