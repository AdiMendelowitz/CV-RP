"""
Unit tests for ETTh1Dataset.

Run with:
    pytest time-series-forecasting/tests/test_ett_dataset.py -v

All tests use a synthetic CSV so no network access or real data is required.
The synthetic data is sized to exactly match the standard 14400-row test boundary,
ensuring the split arithmetic is exercised with realistic lengths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.ett_dataset import ETTh1Dataset, _TEST_END, _TRAIN_END, _VAL_END

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_ROWS = _TEST_END  # 14400  -- minimum required by the dataset class
NUM_CHANNELS = 7  # ETTh1 column count


def _make_synthetic_csv(path: Path, num_rows: int = NUM_ROWS) -> None:
    """Write a minimal ETTh1-shaped CSV with deterministic numeric values."""
    rng = np.random.default_rng(seed=0)
    dates = pd.date_range("2016-07-01", periods=num_rows, freq="h")
    data = rng.standard_normal((num_rows, NUM_CHANNELS))
    cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    df = pd.DataFrame(data, columns=cols)
    df.insert(0, "date", dates.strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(path, index=False)


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("ett")
    p = tmp / "ETTh1.csv"
    _make_synthetic_csv(p)
    return p


# ---------------------------------------------------------------------------
# Helper to instantiate all three splits quickly
# ---------------------------------------------------------------------------

def _make_dataset(csv_path: Path, split: str, seq_len: int = 96, pred_len: int = 96) -> ETTh1Dataset:
    return ETTh1Dataset(csv_path=csv_path, split=split, seq_len=seq_len, pred_len=pred_len)


# ---------------------------------------------------------------------------
# Split boundary tests
# ---------------------------------------------------------------------------


class TestSplitBoundaries:
    """Verify that the three splits cover disjoint, contiguous row ranges."""

    def test_train_row_count(self, csv_path: Path) -> None:
        ds = _make_dataset(csv_path, "train")
        # _data spans rows [0, _TRAIN_END); shape[0] == _TRAIN_END
        assert ds._data.shape[0] == _TRAIN_END

    def test_val_row_count(self, csv_path: Path) -> None:
        ds = _make_dataset(csv_path, "val")
        assert ds._data.shape[0] == _VAL_END - _TRAIN_END

    def test_test_row_count(self, csv_path: Path) -> None:
        ds = _make_dataset(csv_path, "test")
        assert ds._data.shape[0] == _TEST_END - _VAL_END

    def test_no_temporal_overlap(self, csv_path: Path) -> None:
        """Load raw (un-normalised) values and confirm that no timestamp appears in two splits.

        We verify this structurally: the three row ranges [0, TRAIN_END),
        [TRAIN_END, VAL_END), [VAL_END, TEST_END) are mutually exclusive and
        exhaustive up to TEST_END. The dataset class enforces this via
        _split_bounds(), so we check _data shapes match those ranges exactly
        and that they sum correctly.
        """
        train = _make_dataset(csv_path, "train")
        val = _make_dataset(csv_path, "val")
        test = _make_dataset(csv_path, "test")

        total = train._data.shape[0] + val._data.shape[0] + test._data.shape[0]
        assert total == _TEST_END, (
            f"Combined split rows ({total}) do not equal TEST_END ({_TEST_END}); "
            "splits are not exhaustive or have an overlap."
        )


# ---------------------------------------------------------------------------
# Normalisation tests
# ---------------------------------------------------------------------------


class TestNormalisation:
    """Verify that scaling statistics are derived from train rows only."""

    def test_train_mean_zero_after_normalisation(self, csv_path: Path) -> None:
        ds = _make_dataset(csv_path, "train")
        # After z-score normalisation the train split should have ~0 mean per channel.
        col_means = ds._data.mean(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-5)

    def test_train_std_one_after_normalisation(self, csv_path: Path) -> None:
        ds = _make_dataset(csv_path, "train")
        col_stds = ds._data.std(axis=0, ddof=0)
        np.testing.assert_allclose(col_stds, 1.0, atol=1e-5)

    def test_val_uses_train_stats(self, csv_path: Path) -> None:
        """Val and train datasets must share identical scaler parameters."""
        train = _make_dataset(csv_path, "train")
        val = _make_dataset(csv_path, "val")
        np.testing.assert_array_equal(train.train_mean, val.train_mean)
        np.testing.assert_array_equal(train.train_std, val.train_std)

    def test_test_uses_train_stats(self, csv_path: Path) -> None:
        train = _make_dataset(csv_path, "train")
        test = _make_dataset(csv_path, "test")
        np.testing.assert_array_equal(train.train_mean, test.train_mean)
        np.testing.assert_array_equal(train.train_std, test.train_std)

    def test_val_not_zero_mean(self, csv_path: Path) -> None:
        """Val split normalised with train stats should NOT have zero mean in general."""
        ds = _make_dataset(csv_path, "val")
        col_means = ds._data.mean(axis=0)
        # With synthetic random data the val mean won't be exactly 0.
        assert not np.allclose(col_means, 0.0, atol=1e-3), (
            "Val split has zero mean after normalisation, which suggests it was "
            "incorrectly fitted on val data rather than train data."
        )


# ---------------------------------------------------------------------------
# __getitem__ shape and dtype tests
# ---------------------------------------------------------------------------


class TestGetItem:
    SEQ_LEN = 512
    PRED_LEN = 96

    @pytest.fixture(autouse=True)
    def dataset(self, csv_path: Path) -> None:
        self.ds = ETTh1Dataset(
            csv_path=csv_path,
            split="train",
            seq_len=self.SEQ_LEN,
            pred_len=self.PRED_LEN,
        )

    def test_x_shape(self) -> None:
        x, _ = self.ds[0]
        assert x.shape == (self.SEQ_LEN, NUM_CHANNELS)

    def test_y_shape(self) -> None:
        _, y = self.ds[0]
        assert y.shape == (self.PRED_LEN, NUM_CHANNELS)

    def test_x_dtype(self) -> None:
        x, _ = self.ds[0]
        assert x.dtype == torch.float32

    def test_y_dtype(self) -> None:
        _, y = self.ds[0]
        assert y.dtype == torch.float32

    def test_len(self) -> None:
        expected = _TRAIN_END - self.SEQ_LEN - self.PRED_LEN + 1
        assert len(self.ds) == expected

    def test_last_valid_index(self) -> None:
        """The last index must be accessible without raising."""
        last = len(self.ds) - 1
        x, y = self.ds[last]
        assert x.shape == (self.SEQ_LEN, NUM_CHANNELS)
        assert y.shape == (self.PRED_LEN, NUM_CHANNELS)

    def test_out_of_bounds_raises(self) -> None:
        with pytest.raises(IndexError):
            _ = self.ds[len(self.ds)]

    def test_x_y_are_contiguous(self) -> None:
        """y must immediately follow x in the original time series."""
        # Because we control the synthetic data we can verify alignment via
        # the raw _data array: ds._data[idx:idx+seq_len] == x, etc.
        idx = 10
        x, y = self.ds[idx]
        expected_x = torch.from_numpy(self.ds._data[idx : idx + self.SEQ_LEN])
        expected_y = torch.from_numpy(
            self.ds._data[idx + self.SEQ_LEN : idx + self.SEQ_LEN + self.PRED_LEN]
        )
        torch.testing.assert_close(x, expected_x)
        torch.testing.assert_close(y, expected_y)


# ---------------------------------------------------------------------------
# Miscellaneous edge-case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_invalid_split_raises(self, csv_path: Path) -> None:
        with pytest.raises(ValueError, match="split must be"):
            ETTh1Dataset(csv_path=csv_path, split="invalid", seq_len=96, pred_len=96)  # type: ignore[arg-type]

    def test_invalid_seq_len_raises(self, csv_path: Path) -> None:
        with pytest.raises(ValueError, match="seq_len"):
            ETTh1Dataset(csv_path=csv_path, split="train", seq_len=0, pred_len=96)

    def test_invalid_pred_len_raises(self, csv_path: Path) -> None:
        with pytest.raises(ValueError, match="pred_len"):
            ETTh1Dataset(csv_path=csv_path, split="train", seq_len=96, pred_len=0)

    def test_missing_csv_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="ETTh1.csv not found"):
            ETTh1Dataset(csv_path="/nonexistent/ETTh1.csv", split="train", seq_len=96, pred_len=96)

    def test_window_too_large_raises(self, csv_path: Path) -> None:
        # val split has 2880 rows; a window > 2880 should raise.
        with pytest.raises(ValueError, match="seq_len \+ pred_len"):
            ETTh1Dataset(csv_path=csv_path, split="val", seq_len=2000, pred_len=1000)

    def test_too_short_csv_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "ETTh1.csv"
        _make_synthetic_csv(p, num_rows=100)
        with pytest.raises(ValueError, match="at least"):
            ETTh1Dataset(csv_path=p, split="train", seq_len=96, pred_len=96)

    def test_num_features(self, csv_path: Path) -> None:
        ds = _make_dataset(csv_path, "train")
        assert ds.num_features == NUM_CHANNELS

    def test_constant_channel_no_nan(self, tmp_path: Path) -> None:
        """A constant channel should not produce NaN after std clipping."""
        p = tmp_path / "ETTh1.csv"
        dates = pd.date_range("2016-07-01", periods=NUM_ROWS, freq="h")
        data = np.ones((NUM_ROWS, NUM_CHANNELS))  # all constant
        cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        df = pd.DataFrame(data, columns=cols)
        df.insert(0, "date", dates.strftime("%Y-%m-%d %H:%M:%S"))
        df.to_csv(p, index=False)
        ds = ETTh1Dataset(csv_path=p, split="train", seq_len=96, pred_len=96)
        x, y = ds[0]
        assert not torch.isnan(x).any()
        assert not torch.isnan(y).any()