# Time Series Forecasting with Transformers

Reproduction and comparative study of transformer-based architectures for long-horizon multivariate
time series forecasting. All models are implemented from scratch in PyTorch and evaluated on the
ETTh1 benchmark under the standard chronological split used in the PatchTST (Nie et al., ICLR 2023)
and iTransformer (Liu et al., ICLR 2024) papers.

Full benchmark results and per-channel analysis are in [`results/RESULTS.md`](results/RESULTS.md).
Architecture analysis is in [`notes/architecture-comparison.md`](notes/architecture-comparison.md).

---

## Problem Setup

The task is long-horizon multivariate forecasting: given a look-back window of `seq_len` timesteps
across all variates, predict the next `pred_len` steps for all variates simultaneously. Performance
is measured by MSE and MAE averaged element-wise over all variates and all predicted steps on the
held-out test split.

---

## Dataset

**ETTh1** (Electricity Transformer Temperature, hourly) contains readings from a transformer station
in China. The dataset comprises 7 variates: high-, medium-, and low-load usage at full and partial
capacity (HUFL, HULL, MUFL, MULL, LUFL, LULL), and oil temperature (OT). The dataset has 17,420
rows at hourly frequency.

**Split protocol** (matching Nie et al., ICLR 2023 and Liu et al., ICLR 2024):

| Split | Rows | Row indices |
|-------|------|-------------|
| Train | 8,640 | 0 -- 8,639 |
| Val | 2,880 | 8,640 -- 11,519 |
| Test | 2,880 | 11,520 -- 14,399 |

Normalisation is per-channel z-score, with mean and standard deviation fitted on the train rows
only and applied to val and test.

**Source:** `ETTh1.csv` from the iTransformer repository
(https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small).

---

## Repository Structure

```
time-series-forecasting/
├── data/
│   └── ett_dataset.py                  -- ETTh1Dataset: sliding window, normalisation, split
├── models/
│   ├── patchtst.py                     -- PatchTST from scratch (CI and CD modes)
│   ├── itransformer.py                 -- iTransformer from scratch
│   └── timemixer.py                    -- TimeMixer from scratch
├── baselines/
│   └── linear_baseline.py              -- Channel-independent linear baseline (DLinear-style)
├── anomaly/
│   └── detect.py                       -- Reconstruction-error anomaly detection
├── synthetic/
│   └── generate.py                     -- Multivariate AR(1) generator
├── experiments/
│   └── results_grid.csv                -- CI vs CD synthetic grid results
├── results/
│   ├── RESULTS.md                      -- All benchmark results and analysis
│   ├── patchtst_etth1.csv             -- PatchTST per-run metrics, all horizons
│   ├── itransformer_etth1.csv         -- iTransformer per-run metrics
│   ├── timemixer_etth1.csv            -- TimeMixer per-run metrics
│   ├── linear_etth1.csv               -- Linear baseline results
│   ├── ci_cd_etth1.csv                -- CI vs CD ablation results
│   ├── anomaly_etth1.md               -- Anomaly detection evaluation
│   ├── checkpoints/                    -- Best model state dicts (gitignored)
│   └── plots/                          -- Figures from analyze_results.py
├── tests/
│   ├── test_patchtst.py
│   ├── test_itransformer.py
│   └── test_timemixer.py
├── notes/
│   └── architecture-comparison.md
├── patchtst_train_etth1.ipynb         -- Kaggle T4: all four horizons
├── itransformer_train_etth1.ipynb     -- Kaggle T4: all four horizons
├── timemixer_train_etth1.ipynb        -- Kaggle T4: all four horizons
├── ci_cd_train_etth1.ipynb            -- Kaggle T4: CI vs CD ablation
├── analyze_results.py                  -- CPU: plots and per-channel analysis
└── README.md
```

---

## Architectures

| Model | Tokenisation | Attention axis | Inductive bias |
|-------|-------------|----------------|----------------|
| PatchTST | Overlapping time patches per channel | Temporal (within channel) | Channel independence; local temporal context via patching |
| iTransformer | Full history per variate | Cross-variate | Cross-channel correlation; treats each variate as a token |
| TimeMixer | Multi-scale decomposition | None (MLP only) | Trend-seasonal decomposition at multiple resolutions |

### PatchTST (Nie et al., ICLR 2023)

Each variate's look-back window is divided into overlapping patches of length `P` with stride `S`.
Each patch is projected to `d_model` via a learned linear layer, and fixed sinusoidal positional
encodings are added across patch positions. A pre-norm transformer encoder then processes the patch
sequence independently per variate -- no information crosses channel boundaries. This
channel-independence (CI) design forces shared weights to learn patterns that generalise across all
channels, acting as implicit regularisation.

ETTh1 training configuration:

| Hyperparameter | Value |
|----------------|-------|
| seq_len | 512 |
| patch_size | 16 |
| stride | 8 |
| num_patches | 63 (unpadded; paper uses 64 with right-padding) |
| d_model | 128 |
| num_heads | 16 |
| num_layers | 3 |
| dropout | 0.2 |
| batch_size | 128 |
| optimiser | AdamW, lr=1e-4, weight_decay=1e-4 |
| schedule | Linear warmup (10 epochs) + cosine annealing |
| early stopping | patience=10, monitor=val_mse |
| seed | 42 |

### iTransformer (Liu et al., ICLR 2024)

Each variate's full look-back history is projected to a single `d_model`-dimensional token. The
transformer encoder runs attention over the resulting `C` variate tokens, capturing cross-channel
correlations. The forecast head applies a per-variate linear projection from `d_model` to
`pred_len`. ETTh1 uses `seq_len=96` (paper default).

### TimeMixer (Wang et al., ICLR 2024)

The input is downsampled to multiple resolutions via average pooling, producing a scale pyramid.
At each scale, a series decomposition separates seasonal and trend components. Past-Decomposable-
Mixing (PDM) aggregates seasonal components fine-to-coarse and trend components coarse-to-fine.
Future-Multipredictor-Mixing (FMM) applies a linear predictor at each scale and ensembles the
outputs. No attention is used.

---

## Results

Full results with published benchmark comparisons are in [`results/RESULTS.md`](results/RESULTS.md).

Summary (ETTh1, multivariate, test MSE). PatchTST uses seq_len=512; iTransformer uses seq_len=96;
TimeMixer uses seq_len=512. Look-back windows differ across models and numbers are not directly
comparable.

| Model | H=96 | H=192 | H=336 | H=720 | Paper target (H=96) |
|-------|------|-------|-------|-------|---------------------|
| Linear baseline | 0.389 | -- | 0.485 | -- | -- |
| PatchTST | 0.3984 | 0.4417 | 0.4673 | 0.5423 | 0.370 |
| iTransformer | 0.4841 | 0.5450 | 0.6110 | 0.7167 | ~0.454 (avg) |
| TimeMixer | 0.4539 | 0.4990 | 0.5431 | 0.6753 | ~0.446 (avg) |

Linear baseline was trained at H=96 and H=336 only.

### CI vs CD Ablation (ETTh1, pred_len=96)

PatchTST was trained in channel-independent (CI) and channel-dependent (CD) modes under identical
hyperparameters and seed. In CD mode, patches from all variates are concatenated along the sequence
dimension before the encoder, allowing cross-variate attention.

| Mode | Test MSE | Test MAE |
|------|----------|----------|
| PatchTST CI | -- | -- |
| PatchTST CD | -- | -- |

---

## Anomaly Detection

The anomaly detection module (`anomaly/detect.py`) loads a trained PatchTST checkpoint and runs
sliding window inference over the ETTh1 test split (step size 1, seq_len=512, pred_len=1). The
anomaly score at each timestep is the MSE between the one-step-ahead prediction and the observed
value. The detection threshold is the 95th percentile of validation reconstruction errors, fitted
on the validation split only.

Two synthetic anomaly types are injected into the test split for evaluation:

- Point anomaly: Gaussian noise with std = 5 * channel_std at 20 random timesteps.
- Contextual anomaly: a contiguous 24-hour window zeroed out at 3 random locations.

Precision, recall, and F1 are reported per anomaly type. Full results in
[`results/anomaly_etth1.md`](results/anomaly_etth1.md).

This is reconstruction-error-based detection, not a purpose-built anomaly model. Methods such as
Anomaly Transformer (Xu et al., ICLR 2022) use anomaly-specific training objectives and achieve
higher precision on standard benchmarks.

---

## Reproduction

Training runs on Kaggle T4 GPU sessions with the ETTh1 dataset attached. Post-training analysis
runs locally on CPU.

```bash
# After downloading Kaggle notebook outputs to results/

# Plots and per-channel analysis
python analyze_results.py --model_type patchtst --pred_len 96
python analyze_results.py --model_type patchtst --pred_len 192
python analyze_results.py --model_type patchtst --pred_len 336
python analyze_results.py --model_type patchtst --pred_len 720

# Unit tests (no data required)
pytest tests/ -v
```

Strip notebook outputs before committing: `nbstripout *.ipynb`.

---

## References

Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words:
Long-term forecasting with transformers. *ICLR 2023*. https://arxiv.org/abs/2211.14730

Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024). iTransformer: Inverted
transformers are effective for time series forecasting. *ICLR 2024*.
https://arxiv.org/abs/2310.06625

Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J. Y., & Zhou, J. (2024). TimeMixer:
Decomposable multiscale mixing for time series forecasting. *ICLR 2024*.
https://arxiv.org/abs/2405.14616

Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are transformers effective for time series
forecasting? *AAAI 2023*. https://arxiv.org/abs/2205.13504

Xu, J., Wu, H., Wang, J., & Long, M. (2022). Anomaly transformer: Time series anomaly detection
with association discrepancy. *ICLR 2022*. https://arxiv.org/abs/2110.02642