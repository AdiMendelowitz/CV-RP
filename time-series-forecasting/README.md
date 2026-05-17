# Time Series Forecasting with Transformers

Reproduction and comparative study of transformer-based architectures for long-horizon multivariate
time series forecasting. All models are implemented from scratch in PyTorch and evaluated on the
ETTh1 benchmark under the standard chronological split used in the PatchTST (Nie et al., ICLR 2023)
and iTransformer (Liu et al., ICLR 2024) papers.

Full benchmark results, per-channel analysis, and ablation findings are in
[`results/RESULTS.md`](results/benchmark_results.md).

---

## Problem Setup

The task is long-horizon multivariate forecasting: given a look-back window of length `seq_len`
containing observations across all variates, predict the next `pred_len` steps for all variates
simultaneously. Performance is measured by mean squared error (MSE) and mean absolute error (MAE)
averaged element-wise over all variates and all predicted time steps on the held-out test split.

---

## Dataset

**ETTh1** (Electricity Transformer Temperature, hourly) contains readings from an electricity
transformer station in China at one-hour resolution. The dataset comprises 7 variates: high-,
medium-, and low-load usage at full and partial capacity (HUFL, HULL, MUFL, MULL, LUFL, LULL),
and oil temperature (OT).

**Split protocol** (matching Nie et al., ICLR 2023 and Liu et al., ICLR 2024):

| Split | Rows | Calendar period |
|-------|------|-----------------|
| Train | 0 -- 8,640 | First 12 months |
| Val | 8,640 -- 11,520 | Next 4 months |
| Test | 11,520 -- 14,400 | Final 4 months |

Normalisation is per-channel z-score, with mean and standard deviation fitted on the train split
only and applied to val and test.

**Source:** Long Horizon Datasets, Kaggle (thuml/iTransformer repository, datasets folder).

---

## Repository Structure

```
time-series-forecasting/
├── data/
│   └── ett_dataset.py                        -- ETTh1Dataset: sliding window, normalisation, split
├── models/
│   ├── patchtst.py                           -- PatchTST from scratch
│   ├── itransformer.py                       -- iTransformer from scratch
│   └── timemixer.py                          -- TimeMixer from scratch
├── baselines/
│   └── linear_baseline.py                    -- Channel-independent linear baseline
├── anomaly/
│   └── detect.py                             -- Reconstruction-error anomaly detection
├── synthetic/
│   └── generate.py                           -- Multivariate AR(1) generator (Week 4B)
├── experiments/
│   └── results_grid.csv                      -- CI vs CD synthetic grid results (Week 4B)
├── results/
│   ├── RESULTS.md                            -- All benchmark results and analysis
│   ├── patchtst_ettch1.csv                   -- PatchTST per-epoch metrics, all horizons
│   ├── itransformer_ettch1.csv               -- iTransformer per-epoch metrics
│   ├── timemixer_ettch1.csv                  -- TimeMixer per-epoch metrics
│   ├── linear_ettch1.csv                     -- Linear baseline results
│   ├── ci_cd_ettch1.csv                      -- CI vs CD ablation results
│   ├── anomaly_ettch1.md                     -- Anomaly detection evaluation
│   ├── checkpoints/                          -- Best model state dicts (gitignored)
│   └── plots/                               -- Figures produced by analyze_results.py
├── tests/
│   ├── test_patchtst.py
│   ├── test_itransformer.py
│   └── test_timemixer.py
├── notes/
│   └── architecture-comparison.md
├── patchtst_train_ettch1.ipynb               -- Kaggle T4: trains all four horizons
├── itransformer_train_ettch1.ipynb           -- Kaggle T4
├── timemixer_train_ettch1.ipynb              -- Kaggle T4
├── ci_cd_train_ettch1.ipynb                  -- Kaggle T4: CI vs CD ablation
├── analyze_results.py                        -- CPU: plots and per-channel analysis
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

PatchTST tokenises each variate's look-back window into overlapping patches of length `P` with
stride `S`, producing `num_patches = floor((seq_len + pad - P) / S) + 1` tokens per channel.
Each patch is projected to `d_model` via a learned linear layer, and fixed sinusoidal positional
encodings are added across patch positions. A standard pre-norm transformer encoder then processes
the patch sequence independently for each variate -- no information crosses channel boundaries at
any point in the forward pass. This channel-independence (CI) design acts as implicit
regularisation by forcing shared weights to learn patterns that generalise across all channels.

ETTh1 configuration (PatchTST/64, matching Table 3 of Nie et al., ICLR 2023):

| Hyperparameter | Value |
|----------------|-------|
| seq_len | 512 |
| patch_size | 16 |
| stride | 8 |
| num_patches | 64 |
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

*To be completed after Thursday training runs.*

### TimeMixer (Wang et al., ICLR 2024)

*To be completed after Friday training runs.*

---

## Reproduction

All training notebooks are designed to run on a Kaggle T4 GPU session with the Long Horizon
Datasets dataset attached. Post-training analysis runs locally on CPU.

```bash
# After downloading Kaggle outputs to results/

# Plots and per-channel analysis
python analyze_results.py --model_type patchtst --pred_len 96
python analyze_results.py --model_type patchtst --pred_len 192
python analyze_results.py --model_type patchtst --pred_len 336
python analyze_results.py --model_type patchtst --pred_len 720

# Unit tests (CPU, no data required)
pytest tests/ -v
```

---

## References

Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words:
Long-term forecasting with transformers. *International Conference on Learning Representations
(ICLR 2023)*. https://arxiv.org/abs/2211.14730

Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024). iTransformer: Inverted
transformers are effective for time series forecasting. *International Conference on Learning
Representations (ICLR 2024)*. https://arxiv.org/abs/2310.06625

Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J. Y., & Zhou, J. (2024). TimeMixer:
Decomposable multiscale mixing for time series forecasting. *International Conference on Learning
Representations (ICLR 2024)*. https://arxiv.org/abs/2405.14616

Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are transformers effective for time series
forecasting? *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2023)*.
https://arxiv.org/abs/2205.13504

Xu, J., Wu, H., Wang, J., & Long, M. (2022). Anomaly transformer: Time series anomaly detection
with association discrepancy. *International Conference on Learning Representations (ICLR 2022)*.
https://arxiv.org/abs/2110.02642