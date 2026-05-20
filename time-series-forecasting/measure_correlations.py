"""
Measure inter-channel Pearson correlation on the training split of ETTh1 and ECL.

Computes the CxC Pearson correlation matrix on the raw and unnormalized training split for each data set, logs
summary statistics to results/dataset_correlations.csv and saves the full correlation matrix as a NumPy array
for manual inspection.

This script operates on raw CSV data rather than ETTh1Dataset because that class yields windowed (x, y) pairs, while
correlation measurement requires the full raw time series matrix.
Split boundaries match the ett_dataset.py exactly so correlation is measured on the same training rows the model sees.

ETTh1 split boundary: rows [0, 8640)  -- matches _TRAIN_END in ett_dataset.py
ECL split boundary: rows [0, 15840) -- matches iTransformer paper canonical split

Usage:
    python measure_correlations.py

Output:
    results/dataset_correlations.csv   -- summary statistics, one row per dataset
    results/etth1_corr_matrix.npy      -- full 7x7 correlation matrix
    results/ecl_corr_matrix.npy        -- full 321x321 correlation matrix (if ECL available)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent
_DATA_DIR = _REPO_ROOT / "data"

CONFIG = {
    "datasets": [
        {"name": "ETTh1", "csv_path": _DATA_DIR / "ETTh1.csv", "train_rows": 8640, "date_col": "date"},
        {"name": "ECL", "csv_path": _DATA_DIR / "electricity.csv", "train_rows": 15840, "date_col": "date"},
    ],
    "results_dir": _REPO_ROOT / "results",
}

# ---------------------------------------------------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------------------------------------------------

def compute_correlation_stats(csv_path: Path, train_rows: int, date_col: str) -> dict:
    """Load the training split and compute Pearson correlation summary statistics.

    Args:
        csv_path: Path to the dataset CSV file.
        train_rows: Number of rows to use as the training split.
        date_col: Name of the timestamp column to drop before computing correlation.

    Returns:
        Dict with keys:
            corr_matrix (DataFrame, shape (C, C)),
            off_diag_abs (Series of unique absolute off-diagonal values),
            mean_abs, median_abs, min_abs, max_abs (all float).

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If the CSV contains NaN values or fewer rows than train_rows.
    """
    df = pd.read_csv(csv_path, usecols=lambda c: c != date_col)

    if df.isnull().any().any():
        raise ValueError(f"{csv_path.name}: contains NaN values; preprocess required.")
    if len(df) < train_rows:
        raise ValueError(f"{csv_path.name}: expected at least {train_rows} rows, got {len(df)}.")

    corr = df.iloc[:train_rows].corr(method="pearson")

    n = corr.shape[0]
    row_idx, col_idx = np.triu_indices(n, k=1)
    off_diag_abs = pd.Series(np.abs(corr.values[row_idx, col_idx]))

    return {
        "corr_matrix": corr,
        "off_diag_abs": off_diag_abs,
        "mean_abs": off_diag_abs.mean(),
        "median_abs": off_diag_abs.median(),
        "min_abs": off_diag_abs.min(),
        "max_abs": off_diag_abs.max(),
    }


if __name__ == "__main__":
    results_dir: Path = CONFIG["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for ds in CONFIG["datasets"]:
        name: str = ds["name"]
        csv_path: Path = ds["csv_path"]

        if not csv_path.exists():
            print(f"\n[SKIP] {name}: CSV not found at {csv_path}")
            print("       Download it and update CONFIG before running.")
            continue

        print(f"\nProcessing {name}...")
        stats = compute_correlation_stats(csv_path, ds["train_rows"], ds["date_col"])

        corr: pd.DataFrame = stats["corr_matrix"]
        off_diag_abs: pd.Series = stats["off_diag_abs"]
        print(f"  Train split shape: ({ds['train_rows']}, {corr.shape[0]})  (timesteps x variates)")

        npy_path = results_dir / f"{name.lower()}_corr_matrix.npy"
        np.save(npy_path, corr.values)
        print(f"  Correlation matrix saved to {npy_path}")

        n = corr.shape[0]
        print(f"\n{name} Pearson correlation matrix (train split, {n} variates):")
        if n <= 10:
            print(corr.round(3).to_string())
        else:
            print(f"  Matrix is {n}x{n}; printing summary only (full matrix saved to .npy).")
            print(f"  mean   |r|: {off_diag_abs.mean():.4f}")
            print(f"  median |r|: {off_diag_abs.median():.4f}")
            print(f"  min    |r|: {off_diag_abs.min():.4f}")
            print(f"  max    |r|: {off_diag_abs.max():.4f}")

        summary_rows.append({
            "dataset": name,
            "num_variates": n,
            "mean_abs_pearson": round(stats["mean_abs"], 6),
            "median_abs_pearson": round(stats["median_abs"], 6),
            "min_abs_pearson": round(stats["min_abs"], 6),
            "max_abs_pearson": round(stats["max_abs"], 6),
        })

    if not summary_rows:
        print("\nNo datasets processed. Check that CSV paths in CONFIG are correct.")
    else:
        output_csv = results_dir / "dataset_correlations.csv"
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_csv, index=False)
        print(f"\nSummary written to {output_csv}")
        print("\n--- Final summary ---")
        print(summary_df.to_string(index=False))