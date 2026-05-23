"""
Synthetic multivariate AR(1) dataset with controlled inter-channel correlation.

Generate multivariate time series with two independently controllable properties:
    - C: number of variates
    - rho: uniform off-diagonal Pearson correlation (cross-variate structure)

No explicit cross-channel lagged coupling is introduced. Channels share the same AR coefficient phi and are correlated
only through contemporaneous innovation.
Temporal autocorrelation is controlled by phi, stationarity requires |phi|<1.

The generation model is a multivariate AR(1) process with diagonal coefficient matrix:
    x_t = phi * x_{t-1} + epsilon_t,    epsilon_t ~ N(0, Sigma)

Sigma is a uniform correlation (compound symmetry) covariance matrix:
    Sigma_ij = rho for i != j , else 1

This matrix is positive definite for rho in the open interval (-1/(C-1) , 1) and singular at the boundary values.

Because A = phi * I is a scalar multiple of identity, the stationary covariance satisfies
P = phi^2 * P + Sigma  =>  P = Sigma / (1 - phi^2)
The initial state x_0 is drawn from N(0, I), not the stationary distribution. Burn-in substantially reduces the
effect of this initialization, geometrically at rate phi^t. Exact stationarity from time zero would require
initiallizing from N(0, Sigma / (1 - phi^2)).

Split protocol matches ETTh1 fractions (60 / 20 / 20 of usable timesteps):
    usable  = total_len - _BURN_IN        (default: 14400 - 1000 = 13400)
    train   = usable * 8640 // 14400      (default: 8040)
    val     = usable * 2880 // 14400      (default: 2680)
    test    = usable * 2880 // 14400      (default: 2680)

Normalization: StandardScaler fit on train split, applied to all three splits.

Usage:
    from synthetic.generate import SyntheticARDataset

    ds = SyntheticARDataset(C=7, rho=0.5, phi=0.8, seq_len=512, pred_len=96, split="train", seed=42)
    x, y = ds[0]  # x: (512, 7), y: (96, 7)
"""
from typing import Literal

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

_BURN_IN: int = 1000

# ETTh1 split row counts used as fraction numerators over 14400 total rows.
# Applied proportionally to the usable series length after burn-in.
_TRAIN_NUM: int = 8640
_VAL_NUM: int = 2880
_DENOM: int = 14400

type Split = Literal["train", "val", "test"]


def build_covariance(C: int, rho: float) -> np.ndarray:
    """
    Construct a uniform off-diagonal (compound-symmetry) correlation matrix.
    The matrix has ones on the diagonal and rho on the rest:
        Sigma_ij = rho for i != j , else 1
        Equivalent to: rho * ones(C, C) + (1 - rho) * eye(C)
    Args:
        C: number of variates, must be >= 2.
        rho: Off-diagonal correlation coefficient.

    Returns:
        Float64 array of shape (C, C).

    Raises:
        ValueError: If C < 2 or rho is outside the valid positive defined range.
    """

    if C < 2:
        raise ValueError(f"C must be >=2, got {C}")
    lower = -1.0 / (C - 1)
    if not (lower < rho < 1.0):
        raise ValueError(
            f"rho={rho} is outside the valid range ({lower:.6f}, 1.0) for C={C}. "
            "The covariance matrix would not be positive definite."
        )
    diag = (1.0 - rho) * np.eye(C, dtype=np.float64)
    off_diag = rho * np.ones((C, C), dtype=np.float64)
    return off_diag + diag

def generate_ar1(T: int, C: int, phi: float, cov: np.ndarray, seed: int) -> np.ndarray:
    """
    Generate multivariate AR(1) time series and discard burn-in steps.

    The recurrence: x_t = phi * x_{t-1} + epsilon_t
    Where epsilon_t ~ N(0, cov), x_0 is drawn from N(0, I) via rng.standard_normal.
    Burn-in substantially reduces the effect of this initialization.
    All noise samples are pre-generated in one vectorized call. The recurrence loop is the only sequential step.

    Args:
        T: Total timesteps to generate before discarding burn-in, must be > _BURN_IN.
        C: Number of variates.
        phi: Scalar AR(1) coefficient, must satisfy |phi| < 1 for stationarity.
        cov: Positive-definite covariance matrix of shape (C, C).
        seed: Integer seed for the NumPy generator. No global rng state is modified.

    Returns:
        Float64 array of shape (T - _BURN_IN, C).

    Raises:
        ValueError: if T <= _BURN_IN.
    """

    if T <= _BURN_IN:
        raise ValueError(f"T={T} must be > _BURN_IN={_BURN_IN}")

    rng = np.random.default_rng(seed)

    x = np.empty((T, C), dtype=np.float64)
    x[0] = rng.standard_normal(C)  # N(0, I). Burn-in reduces initialization effect
    noise = rng.multivariate_normal(np.zeros(C), cov, size=T-1) # (T-1, C)
    for t in range(1, T):
        x[t] = phi * x[t-1] + noise[t-1]
    return x[_BURN_IN:]  # Discard burn-in steps

class SyntheticARDataset(Dataset):
    """
    Sliding-window dataset over a synthetic multivariate AR(1) time series.

    Generates the full series at construction time, applies a chronological 60/20/20 split matching ETTh1 fractions,
    normalizes using StandardScaler fit on the train split only and exposes sliding window via __getitem__.

    Args:
        C: Number of variates, paper experiments are {7, 21, 84}.
        rho: Off-diagonal correlation coefficient, paper experiments are {0.1, 0.5, 0.9}.
        phi: AR(1) coefficient, fixed at 0.8.
        seq_len: Number of input timesteps per sample.
        pred_len: Number of target timesteps immediately following the input.
        split: One of "train", "val" or "test".
        seed: Integer seed for reproducible series generation.
        total_len: Total timesteps to generate before burn-in discard. Defualt 14,400 => 13,400 usable timesteps
                   splits as 8040 / 2680 / 2680.
    """

    def __init__(self, C: int, rho: float, phi: float, seq_len: int, pred_len: int,
                 split: Split, seed: int, total_len: int = 14400) -> None:
        if split not in ["train", "val", "test"]:
            raise ValueError(f"split must be one of 'train', 'val' or 'test', got {split}")
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if pred_len <1:
            raise ValueError(f"pred_len must be >= 1, got {pred_len}")

        cov = build_covariance(C, rho)
        series = generate_ar1(total_len, C, phi, cov, seed)  # (usable_len, C)

        usable = len(series)
        train_end = usable * _TRAIN_NUM // _DENOM
        val_end = train_end + usable * _VAL_NUM // _DENOM

        scaler = StandardScaler()
        scaler.fit(series[:train_end])
        normalized = scaler.transform(series).astype(np.float32)    # (usable, C)

        match split:
            case "train":
                self._data = normalized[:train_end]
            case "val":
                self._data = normalized[train_end:val_end]
            case "test":
                self._data = normalized[val_end:]

        window = seq_len + pred_len
        if len(self._data) < window:
            raise ValueError(
                f"Split {split} has {len(self._data)} rows but seq_len + pred_len = {window}. "
                "Reduce seq_len or pred_len, or increase total_len."
            )

        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self._data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._data[idx : idx + self.seq_len]          # (seq_len, C)
        y = self._data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)