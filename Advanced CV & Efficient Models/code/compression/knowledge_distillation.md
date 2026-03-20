# Knowledge Distillation

**Paper:** "Distilling the Knowledge in a Neural Network" — Hinton, Vinyals & Dean, 2015
**Link:** https://arxiv.org/abs/1503.02531
**Week:** 2 — Model Compression
**Implementation:** `code/compression/distillation.py`, `code/compression/train_distillation.py`

---

## Core Idea

A large, accurate model (teacher) contains more information in its output distribution than the hard one-hot labels alone. When the teacher predicts, say, 85% cat / 12% dog / 3% other, the 12% dog probability is not noise — it encodes the teacher's learned belief that cats and dogs share visual features. Training a smaller model (student) to match these soft distributions transfers that structured knowledge, not just the final decision.

This is the central insight: **hard labels tell you what the answer is; soft labels tell you how similar the wrong answers are to each other.**

---

## The Loss Function

The combined distillation loss has two terms:

```
L = α · T² · KL(softmax(z_s/T) ‖ softmax(z_t/T)) + (1 - α) · CE(z_s, y)
```

Where:
- `z_s` — student logits
- `z_t` — teacher logits (frozen, no gradient)
- `T` — temperature
- `α` — weight on the soft-target KL term
- `y` — ground truth hard labels

### Temperature T

Dividing logits by T before softmax flattens the probability distribution. At T=1 the model is confident (peaked distribution). At T=4 the distribution is softer, making the small inter-class probabilities (e.g. the 12% dog) much more visible and learnable.

Without temperature scaling, the teacher's distribution is so peaked that all non-maximum probabilities are effectively zero after softmax — the soft target degenerates into a hard label, and KL divergence gives no additional signal beyond CE.

### The T² Multiplier

The full combined loss from the paper (equation 4) is:
```
L = (1 - α) · H(y, σ(z_s; T=1)) + α · H(σ(z_t; T), σ(z_s; T))
```
Where H is cross-entropy and σ is softmax at temperature T. When you divide logits by T, the softmax gradient scales by 1/T². Without compensating, the KL term's gradients are T² times smaller than the CE term's gradients, making α an unstable hyperparameter where increasing T would silently downweight the distillation signal. Multiplying by T² restores gradient magnitudes to be comparable, so α retains its intended meaning regardless of temperature.

In implementation this becomes the KL form shown at the top of this section, with the T² scaling factor applied explicitly. The two formulations are equivalent -
KL divergence between two distributions equals their cross-entropy minus the entropy of the target, and since the teacher distribution is fixed, minimising KL and
minimising cross-entropy are the same optimisation problem.


### Alpha α

Controls the balance between learning from the teacher (soft targets) and the ground truth labels (hard targets). Hinton et al. found α=0.7 works well in practice, meaning the distillation signal dominates. Setting α=0 reduces to standard CE training with no teacher — this is the baseline condition.

---

## Why It Works: Dark Knowledge

The teacher's soft predictions encode **inter-class similarity structure** learned from the full dataset. A model trained on ImageNet learns that Persian cats look more like Siamese cats than like trucks — this similarity is embedded in the off-diagonal probabilities of its output distribution. The student absorbs this structure during training, effectively getting access to a richer training signal than one-hot labels provide.

This is analogous to how a student learns more from a detailed explanation of *why* an answer is correct than from just being told the correct answer.

---

## Architecture

### Teacher: ResNet-18
- Parameters: ~11.2M
- CIFAR-10 accuracy: 93.43% (Week 1 checkpoint)
- Frozen during all distillation training — no gradients computed

### Student: SmallCNN
- Parameters: ~170K
- Architecture: 4 depthwise conv blocks + adaptive avg pool + linear
- Compression ratio: **65×** fewer parameters than teacher

```
Block 1: Conv(3→32)    + BN + ReLU + MaxPool(2) → 16×16
Block 2: Conv(32→64)   + BN + ReLU + MaxPool(2) →  8×8
Block 3: Conv(64→128)  + BN + ReLU + MaxPool(2) →  4×4
Block 4: Conv(128→256) + BN + ReLU + AdaptiveAvgPool(1) → 1×1
Linear(256, 10)
```

---

## Hyperparameters

| Parameter | Value             | Rationale |
|---|-------------------|---|
| Temperature T | 4.0               | Hinton et al. recommendation for vision tasks |
| Alpha α | 0.3               | KL dominates; paper's default |
| Optimizer | Adam              | lr=1e-3 |
| Scheduler | CosineAnnealingLR | Smooth decay, no manual LR tuning |
| Epochs | 30                | Sufficient for student convergence on CIFAR-10 |
| Batch size | 128               | Standard for CIFAR-10 |

---

## Experiment Results

### Run 1 — Diagnostic (broken)

Training stalled at ~27% val_acc. Two bugs: `return` statement inside the batch
loop caused the student to train on only one batch per epoch (1/390th of the data),
and fixed lr=1e-3 with no scheduler caused oscillation. Demonstrates that training
dynamics matter as much as distillation hyperparameters.

### Run 2 — Partial fix (wrong teacher, wrong normalisation)

Fixed the return bug and added CosineAnnealingLR. Used wrong teacher checkpoint
(72.93% accuracy) and mismatched normalisation stats between teacher and students
(0.2023 vs 0.2470 std). Distillation reached 77.77% vs baseline 78.38% (Δ=-0.61pp).
The negative gap was a symptom of the weak teacher — soft targets from a 72.93%
model carry less structured inter-class information.

### Run 3 — Final (correct teacher, unified normalisation)

| | Distillation (T=4, α=0.3) | Baseline (CE only) |
|---|---|---|
| Best val_acc | 78.34% | 78.27% |
| Gap | **+0.07pp** | — |
| val_acc at epoch 1 | 45.61% | 48.26% |
| val_acc at epoch 5 | 61.89% | 62.42% |
| val_acc at epoch 10 | 68.59% | 70.02% |
| val_acc at epoch 20 | 75.91% | 76.61% |

**Interpretation:** with a strong teacher (93.43%) distillation correctly edges above
the baseline. The margin (+0.07pp) is small because student capacity (170K params,
65.6× compression) is the binding constraint — the student cannot fully absorb the
teacher's richer soft distributions. The convergence curves are noisier than the
baseline because the KD loss signal is harder to optimise than plain CE at this
compression ratio.

### Inference Benchmark (CPU, batch_size=128)

| Model | Parameters | Size | Val Acc | Latency/batch | Throughput |
|---|---|---|---|---|---|
| ResNet-18 (teacher) | 11,173,962 | 42.6 MB | 93.43% | 1117ms | 115 img/s |
| SmallCNN — distilled | 170,378 | 0.6 MB | 78.34% | 76ms | 1,676 img/s |
| SmallCNN — baseline CE | 170,378 | 0.6 MB | 78.27% | 67ms | 1,905 img/s |

**Compression ratio:** 65.6× parameters, 71× size  
**Speedup vs teacher:** 14.6× (distilled), 16.6× (baseline)  
**Accuracy cost:** 15.09pp vs teacher at 14.6× speedup

### Plots

Loss breakdown (distillation run):
![Distillation loss breakdown](../../../Advanced%20CV%20%26%20Efficient%20Models/code/compression/plots/distillation/distill_loss_breakdown.png)

Validation accuracy comparison:
![Val accuracy comparison](../../../Advanced%20CV%20%26%20Efficient%20Models/code/compression/plots/distillation/val_accuracy_comparison.png)

---

## Key Observations

**On the KD vs CE loss gap:** throughout training, `kd_loss > ce_loss`. This is expected — the KL divergence between two distributions over 10 classes is naturally larger in scale than cross-entropy against a one-hot target. The ratio between them should decrease as the student's distribution converges toward the teacher's.

**On the 65× compression:** the student at 170K parameters is not expected to match the teacher's 93.43%. The question distillation answers is: *does the student trained with soft targets outperform the same student trained with hard labels alone?* That gap is the measure of what the teacher's knowledge contributes.

**On frozen teacher:** the teacher must be in `eval()` mode with all gradients disabled during distillation training. If the teacher were trainable, backpropagating through both models simultaneously would update the teacher toward the student's poor predictions — the signal would degrade rather than improve.

---

## Variants and Extensions (not implemented)

**Feature-level distillation (FitNets, Romero et al., 2015):** instead of matching only the final logits, intermediate feature maps from the teacher are used as targets for corresponding student layers. More information transfer but requires aligned architectures.

**Progressive distillation (Hinton et al., 2022):** cascade of distillation steps — teacher → medium model → small model — rather than jumping directly from large to tiny. Each step is an easier learning problem.

**Self-distillation (Zhang et al., 2019):** the model distills into itself across epochs, using earlier checkpoints as the teacher. No separate teacher model required.

**Task-Agnostic Distillation (DistilBERT, Sanh et al., 2019):** applied to transformers in NLP; the same principles transfer directly — the architecture is different but the loss formulation is identical.

---

## References

- Hinton, G., Vinyals, O., Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. https://arxiv.org/abs/1503.02531
- Romero, A. et al. (2015). *FitNets: Hints for Thin Deep Nets*. https://arxiv.org/abs/1412.6550
- Sanh, V. et al. (2019). *DistilBERT*. https://arxiv.org/abs/1910.01108
- Gou, J. et al. (2021). *Knowledge Distillation: A Survey*. https://arxiv.org/abs/2006.05525