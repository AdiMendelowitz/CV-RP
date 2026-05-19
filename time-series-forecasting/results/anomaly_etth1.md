# Anomaly Detection on ETTh1 -- PatchTST Reconstruction Errors

## Method

Reconstruction-error-based anomaly detection using a PatchTST model trained for long-horizon
forecasting (seq_len=512, pred_len=96) on ETTh1. For each position in the test split, a
sliding window of 512 timesteps is passed to the model, which predicts 96 steps ahead. The
first predicted step (index 0) serves as the one-step-ahead reconstruction of the timestep
immediately following the window. The anomaly score at each reconstructable timestep is the
MSE between this prediction and the observed value, averaged across all 7 ETTh1 channels.

The anomaly threshold is set at the 95th percentile of reconstruction errors computed on the
validation split. The threshold is fit on the validation split exclusively and applied to
the test split -- no leakage into the test set.

This approach does not use a learned anomaly-specific objective. SOTA methods such as Anomaly
Transformer (Xu et al., ICLR 2022) and TranAD (Tuli et al., VLDB 2022) use training objectives
designed specifically for anomaly detection. The purpose of this experiment is to demonstrate
that forecasting quality and anomaly signal are connected: a model trained purely on next-step
prediction produces elevated reconstruction errors at anomalous timesteps without any
anomaly-specific supervision. This principle applies directly to inverter telemetry monitoring,
where a forecasting model trained on normal operating conditions can flag anomalous sensor
readings by reconstruction error thresholding.

---

## Configuration

| Parameter               | Value                                      |
|-------------------------|--------------------------------------------|
| Model                   | PatchTST (seq_len=512, pred_len=96)        |
| Checkpoint              | patchtst_pred96_best.pt                    |
| Dataset split           | ETTh1 test split (2880 timesteps)          |
| Reconstructable range   | 2368 timesteps (indices 512 through 2879)  |
| Threshold               | 95th percentile of val errors = 1.166358   |
| Point anomalies         | 20 timesteps, Gaussian noise std = 5 * channel_std |
| Contextual anomalies    | 3 contiguous 24-hour windows zeroed to 0.0 (72 timesteps total) |
| Seed                    | 42                                         |

---

## Results

### Point Anomalies

Gaussian noise with standard deviation equal to five times the per-channel training standard
deviation was injected at 20 randomly selected timesteps within the reconstructable region.

| Metric    | Value  |
|-----------|--------|
| Precision | 0.1227 |
| Recall    | 1.0000 |
| F1        | 0.2186 |

### Contextual Anomalies

Three contiguous 24-hour windows were zeroed out at random locations within the reconstructable
region, producing 72 anomalous timesteps in total.

| Metric    | Value  |
|-----------|--------|
| Precision | 0.0420 |
| Recall    | 0.0694 |
| F1        | 0.0524 |

---

## Analysis

### Point anomalies

Recall of 1.0 confirms that every injected point anomaly produced a reconstruction error above
the threshold. The large noise magnitude (5 * channel_std) creates spikes that are
unambiguously outside the normal range, so the forecasting model cannot predict them from
context and the reconstruction error is correspondingly high.

Precision of 0.12 reflects the base rate of false positives introduced by the threshold choice.
The 95th percentile threshold flags 5% of normal timesteps by construction. On 2368
reconstructable test timesteps, this produces approximately 118 false positives in expectation,
which substantially outnumber the 20 injected anomalies. This is not a model failure -- it is
an inherent property of operating at the 95th percentile threshold with a small anomaly count
relative to the normal region.

### Contextual anomalies

Both precision and recall are near zero, indicating that the model does not detect the zeroed
windows reliably. The mechanistic explanation is that zeroing values to 0.0 in the normalized
space produces values close to the data mean, since z-score normalization centers the training
distribution at approximately zero. A window of zeros is therefore not a strong perturbation
from the model's perspective -- the forecasting model can predict near-zero values from context
and the reconstruction error does not spike above the threshold.

This is a meaningful finding: the choice of anomaly injection strategy matters as much as the
detection method. Contextual anomalies that are indistinguishable from the normal distribution
in normalized space cannot be detected by reconstruction error thresholding regardless of model
quality. An alternative injection strategy -- such as zeroing in the original (unnormalized)
scale, or replacing the window with a sustained offset beyond the normal range -- would produce
a more informative evaluation.

### Implications for industrial monitoring

In the inverter telemetry setting, the relevant anomaly types are sensor faults (abrupt spikes
or drops) and gradual degradation (sustained deviation from expected operating conditions).
Point anomalies in this experiment approximate the first type, where reconstruction-error
detection performs well at the cost of false positives. The contextual anomaly result
highlights that the effectiveness of this approach depends critically on how anomalies manifest
relative to the normalized training distribution, which varies by sensor type and fault mode.

---

## Reproduction

```bash
python time-series-forecasting/anomaly/detect.py
```

Requires `patchtst_pred96_best.pt` in `time-series-forecasting/results/checkpoints/` and
`ETTh1.csv` in `time-series-forecasting/data/`. Seed 42 is fixed throughout; results are
fully reproducible.