# Benchmark Results

All experiments use ETTh1 with the standard 12/4/4 month chronological split. MSE and MAE are
averaged element-wise over all variates and all predicted time steps on the held-out test split.
Training was conducted on a Kaggle T4 GPU. The best checkpoint per run was selected by validation
MSE. Per-epoch metrics are logged to the corresponding CSV files in this directory.

---

## PatchTST (Nie et al., ICLR 2023)

Configuration: PatchTST/64, seq_len=512, patch_size=16, stride=8, d_model=128, num_heads=16,
num_layers=3, dropout=0.2, AdamW lr=1e-4, cosine schedule with 10-epoch linear warmup, seed=42.

| Horizon | MSE (ours) | MAE (ours) | MSE (paper) | MAE (paper) | MSE gap |
|---------|-----------|-----------|------------|------------|---------|
| 96 | [fill] | [fill] | 0.370 | 0.400 | -- |
| 192 | [fill] | [fill] | 0.413 | 0.422 | -- |
| 336 | [fill] | [fill] | 0.422 | 0.440 | -- |
| 720 | [fill] | [fill] | 0.447 | 0.468 | -- |

Paper reference: Table 3, ETTh1 multivariate, PatchTST/64.

### Per-channel test MSE

Values are averaged over batch and time dimensions per batch, then divided by the number of
batches. These are relative ranking metrics for identifying the hardest channels; they are not
directly comparable to the scalar MSE reported above.

| Variate | pred=96 | pred=192 | pred=336 | pred=720 |
|---------|---------|----------|----------|----------|
| HUFL | 0.8055 | 0.9080 | 0.9092 | 0.9951 |
| HULL | 0.2077 | 0.2394 | 0.2477 | 0.2773 |
| MUFL | 0.8126 | 0.9170 | 0.9114 | 0.9515 |
| MULL | 0.1578 | 0.1794 | 0.1879 | 0.2153 |
| LUFL | 0.5912 | 0.6235 | 0.7058 | 0.9484 |
| LULL | 0.1330 | 0.1462 | 0.1579 | 0.1689 |
| OT | 0.0894 | 0.1195 | 0.1602 | 0.2423 |
| **Hardest** | MUFL | MUFL | MUFL | HUFL |

The full-load variates (HUFL, MUFL) are consistently the hardest to forecast. OT (oil temperature)
is the easiest at all horizons, contrary to expectations sometimes stated in commentary on this
benchmark. This likely reflects the strong seasonal regularity of temperature relative to the more
volatile load measurements, which are driven by less predictable consumption behaviour.

---

## iTransformer (Liu et al., ICLR 2024)

*To be completed after Thursday training runs.*

Configuration: seq_len=96, d_model=512, num_heads=8, num_layers=4, dropout=0.1, AdamW lr=1e-4,
cosine schedule with 10-epoch linear warmup, seed=42.

| Horizon | MSE (ours) | MAE (ours) | MSE (paper) | MAE (paper) | MSE gap |
|---------|-----------|-----------|------------|------------|---------|
| 96 | -- | -- | -- | -- | -- |
| 192 | -- | -- | -- | -- | -- |
| 336 | -- | -- | -- | -- | -- |
| 720 | -- | -- | -- | -- | -- |
| avg | -- | -- | 0.454 | 0.447 | -- |

Paper reference: Table 1, ETTh1 multivariate, averaged over horizons.

---

## TimeMixer (Wang et al., ICLR 2024)

*To be completed after Friday training runs.*

Configuration: seq_len=512, d_model=16, num_scales=3, decomp_kernel=25, dropout=0.1, AdamW
lr=1e-4, cosine schedule with 10-epoch linear warmup, seed=42.

| Horizon | MSE (ours) | MAE (ours) | MSE (paper) | MAE (paper) | MSE gap |
|---------|-----------|-----------|------------|------------|---------|
| 96 | -- | -- | -- | -- | -- |
| 192 | -- | -- | -- | -- | -- |
| 336 | -- | -- | -- | -- | -- |
| 720 | -- | -- | -- | -- | -- |
| avg | -- | -- | 0.446 | 0.434 | -- |

Paper reference: Table 1, ETTh1 multivariate, averaged over horizons.

---

## Linear Baseline

*To be completed after Monday training runs.*

Channel-independent linear model: one `nn.Linear(seq_len, pred_len)` applied identically per
channel with shared weights. This is DLinear without the trend-seasonal decomposition and serves
as a sanity check -- any transformer implementation that underperforms this baseline is incorrect.

| Horizon | MSE (ours) | MAE (ours) |
|---------|-----------|-----------|
| 96 | -- | -- |
| 336 | -- | -- |

---

## CI vs CD Ablation (PatchTST, ETTh1)

*To be completed after Saturday training runs.*

Channel-independent (CI) mode is the default PatchTST design. Channel-dependent (CD) mode passes
all channel patches jointly through the encoder, allowing attention to operate across variates.
Both variants use identical hyperparameters and seed.

| Mode | pred=96 MSE | pred=96 MAE |
|------|------------|------------|
| CI (default) | -- | -- |
| CD | -- | -- |

Expected finding: CI outperforms CD on ETTh1. With only 7 variates, cross-variate attention
introduces more noise than signal.

---

## Full Comparison Table

| Model | Horizon | MSE (ours) | MAE (ours) | MSE (paper) | Gap |
|-------|---------|-----------|-----------|------------|-----|
| Linear baseline | 96 | -- | -- | -- | -- |
| Linear baseline | 336 | -- | -- | -- | -- |
| PatchTST | 96 | [fill] | [fill] | 0.370 | -- |
| PatchTST | 192 | [fill] | [fill] | 0.413 | -- |
| PatchTST | 336 | [fill] | [fill] | 0.422 | -- |
| PatchTST | 720 | [fill] | [fill] | 0.447 | -- |
| iTransformer | avg | -- | -- | 0.454 | -- |
| TimeMixer | avg | -- | -- | 0.446 | -- |
| PatchTST (CI) | 96 | -- | -- | -- | -- |
| PatchTST (CD) | 96 | -- | -- | -- | -- |

---

## Anomaly Detection

*To be completed after Friday implementation.*

The anomaly detection module applies a trained PatchTST forecasting model (pred_len=1, seq_len=512)
to unsupervised anomaly detection via reconstruction error thresholding. The anomaly score at each
timestep is the MSE between the one-step-ahead prediction and the observed value. The detection
threshold is set at the 95th percentile of validation reconstruction errors, fitted on the
validation split to avoid leakage into the test set.

Two synthetic anomaly types are injected into the test split for evaluation:

- **Point anomaly**: Gaussian noise with std = 5 * channel_std added at 20 random timesteps.
- **Contextual anomaly**: a contiguous 24-hour window zeroed out at 3 random locations.

Precision, recall, and F1 are reported per anomaly type. Full results in `anomaly_ettch1.md`.

This approach is not state-of-the-art. Methods such as Anomaly Transformer (Xu et al., ICLR 2022)
and TranAD use anomaly-specific training objectives. The purpose here is to demonstrate that
forecasting quality and anomaly signal are connected -- a property directly relevant to inverter
telemetry monitoring in industrial IoT settings.