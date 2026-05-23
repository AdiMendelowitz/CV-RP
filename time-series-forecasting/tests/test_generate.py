"""Tests for synthetic/generate.py.

All tests use synthetic data only -- no external downloads required.
Run with: pytest tests/test_generate.py -v
"""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synthetic"))
from generate import (
    SyntheticARDataset,
    _BURN_IN,
    _DENOM,
    _TRAIN_NUM,
    _VAL_NUM,
    build_covariance,
    generate_ar1,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_C = 7
_PHI = 0.8
_SEED = 42
_TOTAL_LEN = 14400
_USABLE = _TOTAL_LEN - _BURN_IN  # 13400
_SEQ_LEN = 512
_PRED_LEN = 96


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_series_hi() -> np.ndarray:
    """Raw AR(1) series with rho=0.9, shape (13400, 7)."""
    return generate_ar1(_TOTAL_LEN, _C, _PHI, build_covariance(_C, 0.9), _SEED)


@pytest.fixture(scope="module")
def raw_series_lo() -> np.ndarray:
    """Raw AR(1) series with rho=0.1, shape (13400, 7)."""
    return generate_ar1(_TOTAL_LEN, _C, _PHI, build_covariance(_C, 0.1), _SEED)


@pytest.fixture(scope="module")
def datasets() -> dict[str, SyntheticARDataset]:
    """One dataset instance per split, shared across tests."""
    kwargs = dict(C=_C, rho=0.5, phi=_PHI, seq_len=_SEQ_LEN, pred_len=_PRED_LEN, seed=_SEED)
    return {
        split: SyntheticARDataset(split=split, **kwargs)
        for split in ("train", "val", "test")
    }


# ---------------------------------------------------------------------------
# Test 1: covariance positive definiteness
# ---------------------------------------------------------------------------


def test_build_covariance_positive_definite() -> None:
    """build_covariance(7, 0.9) must be strictly positive definite.

    The compound-symmetry matrix is positive definite for rho strictly inside
    (-1/(C-1), 1). All eigenvalues must be strictly positive, not merely non-negative.
    """
    cov = build_covariance(_C, 0.9)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert eigenvalues.min() > 0, (
        f"Expected all eigenvalues > 0 for a positive definite matrix; "
        f"got min eigenvalue {eigenvalues.min():.6f}."
    )


# ---------------------------------------------------------------------------
# Test 2: correlation structure
# ---------------------------------------------------------------------------


def test_correlation_structure_high_rho(raw_series_hi: np.ndarray) -> None:
    """Series generated with rho=0.9 must have mean pairwise |Pearson| above 0.7."""
    row_idx, col_idx = np.triu_indices(_C, k=1)
    mean_abs_r = np.abs(np.corrcoef(raw_series_hi.T)[row_idx, col_idx]).mean()
    assert mean_abs_r > 0.7, (
        f"Expected mean pairwise |Pearson| > 0.7 for rho=0.9; got {mean_abs_r:.4f}."
    )


def test_correlation_structure_low_rho(raw_series_lo: np.ndarray) -> None:
    """Series generated with rho=0.1 must have mean pairwise |Pearson| below 0.3."""
    row_idx, col_idx = np.triu_indices(_C, k=1)
    mean_abs_r = np.abs(np.corrcoef(raw_series_lo.T)[row_idx, col_idx]).mean()
    assert mean_abs_r < 0.3, (
        f"Expected mean pairwise |Pearson| < 0.3 for rho=0.1; got {mean_abs_r:.4f}."
    )


# ---------------------------------------------------------------------------
# Test 3: no-leakage split
# ---------------------------------------------------------------------------


def test_split_no_overlap_and_full_coverage(datasets: dict) -> None:
    """Raw data rows assigned to each split must be disjoint and cover all usable rows.

    The three _data arrays must sum to exactly _USABLE rows. This confirms no
    rows are duplicated across splits and none are silently dropped.
    """
    total = sum(len(ds._data) for ds in datasets.values())
    assert total == _USABLE, (
        f"Expected train + val + test rows == {_USABLE}; got {total}."
    )


def test_split_boundaries_are_distinct(datasets: dict) -> None:
    """The last row of train and the first row of val must differ.

    If the split boundary were applied incorrectly (e.g., off by one or
    duplicated), adjacent splits would share rows.
    """
    last_train = datasets["train"]._data[-1]
    first_val  = datasets["val"]._data[0]
    assert not np.allclose(last_train, first_val), (
        "Last row of train equals first row of val -- split boundary may be incorrect."
    )


# ---------------------------------------------------------------------------
# Test 4: normalization
# ---------------------------------------------------------------------------


def test_train_split_normalized_mean_and_std(datasets: dict) -> None:
    """Train split must have per-channel mean ~0 and std ~1 after normalization."""
    data = datasets["train"]._data
    assert np.abs(data.mean(axis=0)).max() < 0.01, (
        f"Train mean not close to 0: max abs = {np.abs(data.mean(axis=0)).max():.6f}"
    )
    assert np.abs(data.std(axis=0) - 1.0).max() < 0.01, (
        f"Train std not close to 1: max |std-1| = {np.abs(data.std(axis=0) - 1.0).max():.6f}"
    )


def test_val_normalized_with_train_statistics() -> None:
    """Val split must be normalized with train statistics, not its own.

    If val were self-normalized its mean would be ~0. Because it is normalized
    with train statistics (which differ from val statistics), the val mean after
    normalization will be measurably nonzero.
    """
    raw = generate_ar1(_TOTAL_LEN, _C, _PHI, build_covariance(_C, 0.5), _SEED)
    usable = len(raw)
    train_end = usable * _TRAIN_NUM // _DENOM
    val_end   = train_end + usable * _VAL_NUM // _DENOM

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(raw[:train_end])
    val_train_normalized = scaler.transform(raw[train_end:val_end])

    # If val were self-normalized, max abs mean would be ~0.
    # Under train normalization it is measurably nonzero.
    val_mean_under_train_scaler = np.abs(val_train_normalized.mean(axis=0)).max()
    assert val_mean_under_train_scaler > 0.01, (
        f"Val mean after train-scaler normalization is {val_mean_under_train_scaler:.6f}; "
        "expected > 0.01, indicating train and val have different distributions."
    )

    # Confirm the dataset's val _data matches train-scaler normalization.
    ds_val = SyntheticARDataset(
        C=_C, rho=0.5, phi=_PHI, seq_len=1, pred_len=1, split="val", seed=_SEED
    )
    actual_val_mean = np.abs(ds_val._data.mean(axis=0)).max()
    assert np.isclose(actual_val_mean, val_mean_under_train_scaler, atol=1e-5), (
        f"Dataset val mean {actual_val_mean:.6f} does not match train-scaler val mean "
        f"{val_mean_under_train_scaler:.6f}; normalization may be using val statistics."
    )


# ---------------------------------------------------------------------------
# Test 5: __getitem__ shapes and dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_getitem_shapes_and_dtypes(split: str, datasets: dict) -> None:
    """__getitem__ must return (x, y) with correct shapes and float32 dtype."""
    ds = datasets[split]
    x, y = ds[0]
    assert x.shape == (_SEQ_LEN, _C), f"{split}: expected x shape ({_SEQ_LEN}, {_C}), got {tuple(x.shape)}"
    assert y.shape == (_PRED_LEN, _C), f"{split}: expected y shape ({_PRED_LEN}, {_C}), got {tuple(y.shape)}"
    assert x.dtype == torch.float32, f"{split}: expected float32, got {x.dtype}"
    assert y.dtype == torch.float32, f"{split}: expected float32, got {y.dtype}"


# ---------------------------------------------------------------------------
# Test 6: reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_series() -> None:
    """Two SyntheticARDataset instances with the same seed must be identical."""
    kwargs = dict(C=_C, rho=0.5, phi=_PHI, seq_len=_SEQ_LEN, pred_len=_PRED_LEN,
                  split="train", seed=_SEED)
    ds1 = SyntheticARDataset(**kwargs)
    ds2 = SyntheticARDataset(**kwargs)
    assert np.array_equal(ds1._data, ds2._data), (
        "Two instances with the same seed produced different _data arrays."
    )


def test_different_seeds_produce_different_series() -> None:
    """Two SyntheticARDataset instances with different seeds must differ."""
    kwargs = dict(C=_C, rho=0.5, phi=_PHI, seq_len=_SEQ_LEN, pred_len=_PRED_LEN, split="train")
    ds1 = SyntheticARDataset(seed=42,  **kwargs)
    ds2 = SyntheticARDataset(seed=99,  **kwargs)
    assert not np.array_equal(ds1._data, ds2._data), (
        "Two instances with different seeds produced identical _data arrays."
    )