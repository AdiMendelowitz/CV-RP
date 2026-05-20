"""
ETTh1 dataset for long-horizon multivariate time series forecasting.

Dataset: Electricity Transformer Temperature, hourly (ETTh1).
Source: https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh1.csv

Split protocol: (matches PatchTST, iTransformer, TimeMixer papers):
    Train: first 12 months -> rows [0, 8640)
    Val: next 4 months -> rows [8640, 11520)
    Test: final 4 months -> rows [11520, 14400)

Normalization: per-channel z-score, scaler fit on train split only.

References: Nie et al., "A Time Series is Worth 64 Words", ICLR 2023.
            Liu et al., "iTransformer", ICLR 2024.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Hourly rows per calendar month (365.25 / 12 * 24, rounded to match paper convention).
# 12 months = 8640 hours
_TRAIN_END: int = 8640
_VAL_END: int = 11520
_TEST_END: int = 14400

Split = Literal["train", "val", "test"]


class ETTh1Dataset(Dataset):
    """
    Sliding window dataset over the ETTh1 time series.

    Args:
        csv_path: Path to ETTh1.csv.
        split: One of ['train', 'val', 'test'].
        seq_len: Number of input timesteps.
        pred_len: Number of target timesteps immediately following the input.
    """

    def __init__(self, csv_path: str | Path, split: Split, seq_len: int, pred_len: int) -> None:
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of 'train', 'val' or 'test', got {split}")
        if seq_len < 1:
            raise ValueError(f"seq_len must be >=1, got {seq_len}")
        if pred_len < 1:
            raise ValueError(f"pred_len must be >=1, got {pred_len}")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split

        raw = self._load_csv(Path(csv_path))  # (T, C)
        self._validate_length(raw)

        # Fit scaler on train rows only, then apply to all rows before slicing
        train_rows = raw[:_TRAIN_END]
        self._mean = train_rows.mean(axis=0)  # (C,)
        self._std = train_rows.std(axis=0, ddof=0).clip(min=1e-8)  # (C,)
        normalized = (raw - self._mean) / self._std  # (T, C)

        start, end = self._split_bounds(split)
        self._data = normalized[start:end].astype(np.float32)  # (split_len, C)

        window = seq_len + pred_len
        if len(self._data) < window:
            raise ValueError(
                f"Split '{split}' has {len(self._data)} rows but seq_len + pred_len = {window}."
                "Reduce seq_len or pred_len."
            )
        self._num_samples = len(self._data) - window + 1

    # ----------------------------------------------------------------------------
    # Dataset protocol
    # ----------------------------------------------------------------------------

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self._num_samples})")
        x = self._data[idx : idx + self.seq_len]  # (seq_len, C)
        y = self._data[idx + self.seq_len : idx + self.seq_len + self.pred_len]  # (pred_len, C)
        return torch.from_numpy(x), torch.from_numpy(y)

    # ----------------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------------

    @property
    def num_features(self) -> int:
        """Number of channels / variates."""
        return self._data.shape[1]

    @property
    def train_mean(self) -> np.ndarray:
        """Per-channel mean fitted on train split, shape (C,)."""
        return self._mean.copy()

    @property
    def train_std(self) -> np.ndarray:
        """Per-channel std fitted on train split, shape (C,)."""
        return self._std.copy()

    # ----------------------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------------------

    @staticmethod
    def _load_csv(path: Path) -> np.ndarray:
        """Load ETTh1.csv and return numeric columns as a float64 array."""
        if not path.exists():
            raise FileNotFoundError(
                f"ETTh1.csv not found {path}. Please download from " "https://github.com/zhouhaoyi/ETDataset"
            )
        df = pd.read_csv(path)
        numeric = df.drop(columns=["date"])
        if numeric.isnull().any().any():
            raise ValueError("ETTh1.csv contains NaN values, preprocess required.")
        return numeric.to_numpy(dtype=np.float64)

    @staticmethod
    def _validate_length(data: np.ndarray) -> None:
        if len(data) < _TEST_END:
            raise ValueError(
                f"ETTh1.csv has {len(data)} rows but at least {_TEST_END} are required for"
                " the standard 12/4/4 month split"
            )

    @staticmethod
    def _split_bounds(split: Split) -> tuple[int, int]:
        bounds: dict[Split, tuple[int, int]] = {
            "train": (0, _TRAIN_END),
            "val": (_TRAIN_END, _VAL_END),
            "test": (_VAL_END, _TEST_END),
        }
        return bounds[split]
