# Model Compression

Three compression techniques applied to a ResNet-18 baseline (93.43% top-1, 11.17M
parameters, 42.70MB) trained from scratch on CIFAR-10. All benchmarks run on CPU.
Latency is single-sample median over 100 runs.

---

## Files

| File | Description |
|---|---|
| `distillation.py` | SmallCNN student, distillation loss, build_student factory |
| `train_distillation.py` | Training script: distillation and CE baseline modes |
| `quantization.py` | Dynamic and static INT8 post-training quantization |
| `inference_benchmark.py` | Teacher vs student latency and accuracy benchmark |
| `visualize_soft_targets.py` | Soft target distributions at T=1, 2, 4, 8 |
| `distillation-kaggle.ipynb` | Distillation training notebook (Kaggle T4) |
| `pruning_kaggle.ipynb` | L1 unstructured pruning notebook (Kaggle T4) |

---

## Knowledge Distillation

**Reference:** Hinton, G., Vinyals, O., and Dean, J. Distilling the Knowledge in a Neural
Network. arXiv:1503.02531. https://arxiv.org/abs/1503.02531

A SmallCNN student (170K parameters) was trained to match the output distribution of the
ResNet-18 teacher using the combined KD and cross-entropy loss:

```
L = alpha * T^2 * KL(softmax(z_t/T) || softmax(z_s/T)) + (1 - alpha) * CE(z_s, y)
```

where `z_s` and `z_t` are student and teacher logits, `T` is temperature, and `alpha`
weights the soft-target KL term. The `T^2` factor rescales gradient magnitudes to keep
`alpha` a stable hyperparameter across temperature values.

### Student Architecture: SmallCNN

```
Block 1: Conv(3->32,  3x3) + BN + ReLU + MaxPool(2)       -> 16x16
Block 2: Conv(32->64, 3x3) + BN + ReLU + MaxPool(2)       ->  8x8
Block 3: Conv(64->256,3x3) + BN + ReLU + AdaptiveAvgPool  ->  1x1
Linear(256, 10)
```

170,378 parameters. 65.6x smaller than the ResNet-18 teacher. No skip connections.

### Hyperparameters

| Parameter | Value |
|---|---|
| Temperature T | 4.0 |
| KL weight alpha | 0.3 |
| CE weight (1 - alpha) | 0.7 |
| Epochs | 30 |
| Batch size | 128 |
| Optimizer | Adam (lr=1e-3) |
| Scheduler | CosineAnnealingLR |

### Results

| Model | Parameters | Size (MB) | Latency (ms) | Top-1 (%) |
|---|---|---|---|---|
| ResNet-18 (teacher) | 11,173,962 | 42.63 | 12.44 | 93.43 |
| SmallCNN (distilled) | 170,378 | 0.65 | 0.86 | 78.33 |
| SmallCNN (CE baseline) | 170,378 | 0.65 | 0.84 | 78.97 |

The distilled student achieves 65.6x parameter reduction and 14.4x speedup at a 15.10pp
accuracy cost versus the teacher. The distillation gain over the CE-only baseline is
-0.64pp: the baseline marginally outperforms the distilled student with these
hyperparameters.

This outcome is consistent with the capacity-gap analysis. At 65x compression, student
capacity is the binding constraint rather than the quality of the training signal. With
alpha=0.3, the KL term (weighted 30%) introduces noise relative to the dominant CE signal
(70%) that is not offset by the additional inter-class similarity information in the soft
targets at this compression ratio. A larger student or a higher alpha value would be
expected to widen the distillation gap in the positive direction.

### Soft Target Visualization

`visualize_soft_targets.py` plots the teacher's softmax distribution at T=1, 2, 4, and 8
for a sample of CIFAR-10 images. At T=1 the distribution is near-degenerate (the correct
class receives close to 100% probability). At T=4 inter-class similarity emerges: a cat
sample assigns non-trivial probability to dog and deer; a ship sample spreads probability
to airplane and truck. At T=8 the distribution approaches uniform and the class signal
degrades. This is the empirical justification for T=4 as the operating temperature.

### Running Distillation

```bash
# Distillation + CE baseline (Kaggle T4 recommended, ~25 min)
# Run distillation-kaggle.ipynb on Kaggle with the resnet_input dataset attached.

# Inference benchmark (local CPU)
python "Advanced CV & Efficient Models/code/compression/inference_benchmark.py"

# Soft target visualization (local CPU, requires teacher checkpoint)
python "Advanced CV & Efficient Models/code/compression/visualize_soft_targets.py"
```

---

## Post-Training Quantization

**Reference:** Jacob, B. et al. Quantization and Training of Neural Networks for Efficient
Integer-Arithmetic-Only Inference. CVPR 2018. https://arxiv.org/abs/1712.05877

PTQ was applied to the ResNet-18 checkpoint using `torch.ao.quantization`. Both dynamic
and static schemes were evaluated on CPU.

**Quantization map:**

```
x_int = clamp(round(x_float / scale) + zero_point, q_min, q_max)
```

**Dynamic quantization** quantizes weights offline (INT8) and activations at runtime per
forward pass. PyTorch does not support dynamic quantization for Conv2d; only the final
Linear layer (5,120 of 11.17M parameters) is quantized, so no meaningful size reduction
occurs. The latency increase reflects dequantization overhead on a convolution-dominated
model.

**Static PTQ** applies Conv-BN fusion before calibration to eliminate intermediate
requantization steps, then calibrates activation observers over 100 training batches
(12,800 samples). Per-channel weight quantization and per-tensor unsigned INT8 activation
quantization are used, following Jacob et al. (2018).

### Results

| Method | Size (MB) | Latency (ms) | Top-1 (%) | Delta (pp) |
|---|---|---|---|---|
| FP32 baseline | 42.70 | 9.47 | 93.43 | -- |
| INT8 dynamic (Linear only) | 42.69 | 18.80 | 93.44 | +0.01 |
| INT8 static (PTQ) | 10.80 | 6.21 | 93.44 | +0.01 |

Static PTQ achieves 3.95x size reduction and 1.52x speedup with no measurable accuracy
degradation, consistent with well-calibrated INT8 PTQ results on ResNet-class models
reported in Nagel et al. (2021).

### Running Quantization

```bash
python "Advanced CV & Efficient Models/code/compression/quantization.py"
```

---

## Unstructured Pruning

**Reference:** Han, S., Pool, J., Tran, J., and Dally, W. Learning both Weights and
Connections for Efficient Neural Networks. NeurIPS 2015.
https://arxiv.org/abs/1506.02626

L1 magnitude pruning was applied to all Conv2d weight tensors using
`torch.nn.utils.prune.l1_unstructured`, followed by 5 epochs of fine-tuning with
SGD (lr=1e-3, momentum=0.9, weight_decay=5e-4) with masks held fixed. Masks were then
made permanent via `prune.remove`.

### Results

| Target sparsity | Actual sparsity | Top-1 before FT (%) | Top-1 after FT (%) | Delta vs baseline (pp) |
|---|---|---|---|---|
| 0% (baseline) | 0.0% | 93.43 | 93.43 | -- |
| 20% | 20.0% | 93.45 | 93.17 | -0.26 |
| 40% | 40.0% | 93.21 | 93.27 | -0.16 |
| 60% | 60.0% | 91.56 | 93.16 | -0.27 |

5-epoch fine-tuning recovers accuracy to within 0.27pp of the dense baseline at all three
sparsity levels. At 60% sparsity, the pre-fine-tuning degradation is 1.87pp; fine-tuning
closes this to 0.27pp below baseline.

These results do not translate to runtime improvements. L1 unstructured pruning zeroes
individual weight values within a dense float32 tensor without changing its shape.
Standard CPU matrix multiplication kernels process zero-valued weights identically to
non-zeros, so size and latency remain equal to the FP32 baseline after mask removal.
Structured pruning (filter or channel removal) would be required to achieve actual size
and latency reductions.

### Running Pruning

```bash
# Kaggle T4 recommended
# Run pruning_kaggle.ipynb on Kaggle with the resnet_input dataset attached.
```

---

## Compression Summary

All techniques applied to the same ResNet-18 checkpoint. Distillation produces a
different architecture (SmallCNN, 170K parameters) and is not directly comparable to the
quantization and pruning rows, which all operate on ResNet-18 weights.

| Method | Params (M) | Size (MB) | Latency (ms) | Top-1 (%) |
|---|---|---|---|---|
| ResNet-18 (FP32 baseline) | 11.17 | 42.70 | 9.47 | 93.43 |
| + Static INT8 (PTQ) | 11.17 | 10.80 | 6.21 | 93.44 |
| + Dynamic INT8 | 11.17 | 42.69 | 18.80 | 93.44 |
| + Pruning 40% (L1 unstructured) | 11.17 | 42.70 | 9.47* | 93.27 |
| SmallCNN (distilled, KD) | 0.17 | 0.65 | 0.86 | 78.33 |
| SmallCNN (CE baseline) | 0.17 | 0.65 | 0.84 | 78.97 |

*Pruning latency equals FP32 baseline: unstructured sparsity has no CPU speedup
without sparse compute kernels.

Static PTQ is the most practical technique for this architecture: 3.95x size reduction
and 1.52x speedup with no accuracy cost and no retraining. Knowledge distillation yields
65.6x parameter reduction and 14.4x speedup but at a 15.10pp accuracy cost, and at this
compression ratio produces no benefit over CE-only training.

---

## References

- Hinton, G., Vinyals, O., and Dean, J. (2015). Distilling the Knowledge in a Neural
  Network. arXiv:1503.02531.
- Han, S., Pool, J., Tran, J., and Dally, W. (2015). Learning both Weights and
  Connections for Efficient Neural Networks. NeurIPS 2015.
- Jacob, B. et al. (2018). Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference. CVPR 2018.
- Nagel, M. et al. (2021). A White Paper on Neural Network Quantization.
  arXiv:2106.08295.