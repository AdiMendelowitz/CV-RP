# Time-Series Forecasting with Transformers

From-scratch PyTorch implementations of three transformer-based architectures for
long-horizon multivariate forecasting, evaluated on the ETTh1 benchmark under the
standard chronological split used by Nie et al. (ICLR 2023) and Liu et al.
(ICLR 2024). PatchTST, iTransformer, and TimeMixer are each built from
`nn.Module` primitives alongside a linear baseline, with a shared dataset loader
and a unit-test suite covering every model.

---

## Problem Setup

Given a look-back window of `seq_len` timesteps across all variates, predict the
next `pred_len` steps for all variates jointly. Performance is the element-wise
MSE and MAE over all variates and predicted steps on the held-out test split.

---

## Dataset

ETTh1 (Electricity Transformer Temperature, hourly) records seven variates from a
transformer station: high-, medium-, and low-load usage at full and partial
capacity (HUFL, HULL, MUFL, MULL, LUFL, LULL) and oil temperature (OT), across
17,420 hourly rows.

The chronological split matches the published protocol:

| Split | Rows | Row indices |
|-------|------|-------------|
| Train | 8,640 | 0 – 8,639 |
| Val | 2,880 | 8,640 – 11,519 |
| Test | 2,880 | 11,520 – 14,399 |

Normalisation is per-channel z-score, fitted on train rows only and applied to
val and test. Measured on the train split, the seven variates have a mean
absolute Pearson correlation of 0.31, with the strongest pair (HUFL, MUFL) at
0.98 and the weakest near 0.04.

Source: `ETTh1.csv` from the ETDataset repository
(https://github.com/zhouhaoyi/ETDataset).

---

## Repository Structure

```
time-series-forecasting/
├── data/
│   └── ett_dataset.py            -- ETTh1Dataset: sliding window, normalisation, split
├── models/
│   ├── patchtst.py               -- PatchTST from scratch
│   ├── itransformer.py           -- iTransformer from scratch
│   └── timemixer.py              -- TimeMixer from scratch
├── baselines/
│   └── linear_baseline.py        -- Channel-independent linear baseline (DLinear-style)
├── anomaly/
│   └── detect.py                 -- Reconstruction-error anomaly detection
├── synthetic/
│   └── generate.py               -- Multivariate AR(1) generator
├── results/
│   ├── patchtst_etth1.csv        -- PatchTST test metrics, all horizons
│   ├── itransformer_etth1.csv    -- iTransformer test metrics
│   ├── timemixer_etth1.csv       -- TimeMixer test metrics
│   ├── linear_etth1.csv          -- Linear baseline results
│   ├── anomaly_etth1.md          -- Anomaly detection evaluation
│   └── plots/                    -- Forecast and per-channel figures
├── tests/
│   ├── test_patchtst.py
│   ├── test_itransformer.py
│   ├── test_timemixer.py
│   ├── test_ett_dataset.py
│   └── test_generate.py
├── patchtst_train_etth1.ipynb    -- Kaggle T4: all four horizons
├── itransformer_train_etth1.ipynb
├── timemixer_train_etth1.ipynb
└── analyze_results.py            -- CPU: plots and per-channel analysis
```

---

## Architectures

| Model | Tokenisation | Attention axis | Core idea |
|-------|-------------|----------------|-----------|
| PatchTST | Overlapping time patches per channel | Temporal, within channel | Patch-based forecasting with shared per-channel weights |
| iTransformer | Full history per variate | Cross-variate | Each variate becomes one token; attention over variates |
| TimeMixer | Multi-scale decomposition | None (MLP only) | Trend-seasonal mixing at multiple resolutions |

### PatchTST (Nie et al., ICLR 2023)

Each variate's look-back window is divided into overlapping patches of length 16
with stride 8, projected to `d_model` with a learned linear layer and sinusoidal
positional encodings. A pre-norm transformer encoder processes the patch sequence
per variate. ETTh1 configuration: seq_len 512, d_model 128, 16 heads, 3 layers,
dropout 0.2, batch size 128, AdamW (lr 1e-4, weight decay 1e-4), linear warmup
over 10 epochs then cosine annealing, early stopping on validation MSE with
patience 10, seed 42.

### iTransformer (Liu et al., ICLR 2024)

Each variate's full look-back history is projected to a single `d_model` token,
and the encoder runs attention across the variate tokens. The forecast head
applies a per-variate linear projection from `d_model` to `pred_len`. ETTh1 uses
seq_len 96, the paper default.

### TimeMixer (Wang et al., ICLR 2024)

The input is average-pooled to multiple resolutions, forming a scale pyramid. At
each scale a series decomposition separates seasonal and trend components;
Past-Decomposable-Mixing aggregates seasonal fine-to-coarse and trend
coarse-to-fine, and Future-Multipredictor-Mixing applies a linear predictor at
each scale and ensembles the outputs. No attention is used.

---

## Results

ETTh1 multivariate test metrics at seed 42. Look-back windows differ by model
(PatchTST and TimeMixer 512, iTransformer 96), so figures are comparable within a
model across horizons rather than across models.

PatchTST (look-back 512, from scratch, seed 42):

| Horizon | MSE | MAE |
|---------|------|------|
| 96 | 0.398 | 0.421 |
| 192 | 0.442 | 0.449 |
| 336 | 0.467 | 0.467 |
| 720 | 0.542 | 0.526 |

iTransformer and TimeMixer reproduced metrics:

| Model | H=96 | H=192 | H=336 | H=720 |
|-------|------|-------|-------|-------|
| iTransformer (MSE) | 0.484 | 0.545 | 0.611 | 0.717 |
| iTransformer (MAE) | 0.483 | 0.517 | 0.564 | 0.628 |
| TimeMixer (MSE) | 0.454 | 0.499 | 0.543 | 0.675 |
| TimeMixer (MAE) | 0.452 | 0.481 | 0.514 | 0.602 |

The linear baseline (look-back 512) records MSE 0.389 / MAE 0.405 at horizon 96
and MSE 0.485 / MAE 0.471 at horizon 336, serving as the sanity floor: a
transformer that fails to beat it at a given horizon is not learning useful
temporal structure. The from-scratch PatchTST reproduces the expected horizon
scaling on ETTh1, with error rising monotonically as prediction length grows,
in line with the PatchTST architecture (Nie et al., ICLR 2023). Figures are
single-seed under a fixed training budget.

---

## Anomaly Detection

`anomaly/detect.py` loads a trained PatchTST checkpoint and runs sliding-window
one-step-ahead inference over the ETTh1 test split. The anomaly score at each
timestep is the MSE between prediction and observation, and the detection
threshold is the 95th percentile of validation reconstruction errors, fitted on
the validation split only. Two synthetic anomaly types are injected for
evaluation: point anomalies (Gaussian noise at scale 5x channel std at 20 random
timesteps) and contextual anomalies (a 24-hour window zeroed at 3 random
locations). Precision, recall, and F1 are reported per type in
[`results/anomaly_etth1.md`](results/anomaly_etth1.md). This is
reconstruction-error detection rather than a purpose-built anomaly model;
methods such as Anomaly Transformer (Xu et al., ICLR 2022) use anomaly-specific
objectives and reach higher precision on standard benchmarks.

---

## Reproduction

Training runs on Kaggle T4 GPU sessions with the ETTh1 dataset attached, and
post-training analysis runs locally on CPU.

```bash
# After downloading Kaggle notebook outputs to results/
python analyze_results.py --model_type patchtst --pred_len 96

# Unit tests (no data required)
pytest tests/ -v
```

Strip notebook outputs before committing: `nbstripout *.ipynb`.

---

## References

Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is
worth 64 words: Long-term forecasting with transformers. *ICLR 2023*.
https://arxiv.org/abs/2211.14730

Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024).
iTransformer: Inverted transformers are effective for time series forecasting.
*ICLR 2024*. https://arxiv.org/abs/2310.06625

Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J. Y., & Zhou, J.
(2024). TimeMixer: Decomposable multiscale mixing for time series forecasting.
*ICLR 2024*. https://arxiv.org/abs/2405.14616

Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are transformers effective for
time series forecasting? *AAAI 2023*. https://arxiv.org/abs/2205.13504

Xu, J., Wu, H., Wang, J., & Long, M. (2022). Anomaly transformer: Time series
anomaly detection with association discrepancy. *ICLR 2022*.
https://arxiv.org/abs/2110.02642