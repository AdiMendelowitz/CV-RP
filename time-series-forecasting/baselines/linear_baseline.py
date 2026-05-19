"""
Channel-independent linear baseline for long-horizon time series forecasting.

Architecture: one nn.Linear(seq_len, pred_len) applied identically and independently to each input channel.
Weights are shared across channels by construction, no cross-channel interactions occurs.
This is the channel-independent (CI) variant of DLinear (Zeng et al., AAAI 2023) without the trend-seasonal
decomposition.

Purpose: establish the lower bound for models. If PatchTST underperforms this baseline on ETTh1,
the transformer implementation is incorrect.

Reference: Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023.
  https://arxiv.org/abs/2205.13504
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.ett_dataset import ETTh1Dataset  # noqa: PLC0415

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --------------------------------------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------------------------------------

_BASELINES_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _BASELINES_DIR.parent
_DATA_DIR = _PROJECT_DIR.parent / "data"
_RESULTS_DIR = _PROJECT_DIR / "results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_CSV_PATH = _DATA_DIR / "ETTh1.csv"
_RESULTS_CSV = _RESULTS_DIR / "linear_ettch1.csv"
_CHECKPOINT_DIR = _RESULTS_DIR / "checkpoints"
_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------------------------

CONFIG: dict = {
    "seq_len": 512,
    "pred_lens": [96, 336],
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 50,
    "patience": 10,
    "seed": 42,
    "num_workers": 0,
}

# --------------------------------------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------------------------


class ChannelIndependentLinear(nn.Module):
    """
    Linear forecaster with shared weights applied independently per channel.

    Input shape (B, seq_len, C), output shape (B, pred_len, C).

    The same nn.Linear(seq_len, pred_len) applied to every channel's history by transposing so that seq_len
    occupies the last dimension. No cross-channel interaction takes place.

    Args:
        seq_len: Length of the input lookback window.
        pred_len: Number of future steps to forecast.
    """

    def __init__(self, seq_len: int, pred_len: int) -> None:
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, C)
        x = x.transpose(-1, -2)  # (B, C, seq_len)
        out = self.linear(x)  # (B, C, pred_len)
        return out.permute(0, 2, 1)  # (B, pred_len, C)


# --------------------------------------------------------------------------------------------------------------------
# Training helpers
# --------------------------------------------------------------------------------------------------------------------


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one full pass over the loader.

    Args:
        optimizer: If none, runs in eval mode (no gradient updates).

    Returns:
        (mean_mse, mean_mae) averaged over all samples in the loader.
    """

    training = optimizer is not None
    model.train(training)

    total_mse, total_mae, total_samples = 0.0, 0.0, 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            mse = criterion(pred, y)
            mae = (pred - y).abs().mean()

            if training:
                optimizer.zero_grad()
                mse.backward()
                optimizer.step()

            batch_size = x.size(0)
            total_mse += mse.item() * batch_size
            total_mae += mae.item() * batch_size
            total_samples += batch_size

    return total_mse / total_samples, total_mae / total_samples


def _train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    checkpoint: Path,
) -> None:
    """Train with early stopping on val MSE, checkpoint best model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.MSELoss()

    best_val_mse = float("inf")
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        train_mse, train_mae = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_mse, val_mae = _run_epoch(model, val_loader, criterion, None, device)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            patience_counter += 1

        print(
            f"  epoch {epoch:3d} | train MSE {train_mse:.4f} MAE {train_mae:.4f}"
            f" | val MSE {val_mse:.4f} MAE {val_mae:.4f}"
            + (" [best]" if patience_counter == 0 else f" [patience {patience_counter}/{config['patience']}]")
        )

        if patience_counter >= config["patience"]:
            print(f"  Early stopping at epoch {epoch}")
            break


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, checkpoint_path: Path) -> tuple[float, float]:
    """Load the best checkpoint and evaluate on the given loader."""
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    criterion = nn.MSELoss()
    mse, mae = _run_epoch(model, loader, criterion, None, device)
    return mse, mae


# --------------------------------------------------------------------------------------------------------------------
# Results logging
# --------------------------------------------------------------------------------------------------------------------


def _append_results(results_path: Path, row: dict) -> None:
    """Append one result row to the CSV. Write the header if the file is new."""
    fieldnames = ["seq_len", "pred_len", "test_mse", "test_mae", "timestamp"]
    write_header = not results_path.exists()
    with results_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    _seed_everything(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seq_len = CONFIG["seq_len"]

    for pred_len in CONFIG["pred_lens"]:
        print(f"\n{'='*60}")
        print(f"Linear baseline | seq_len={seq_len} | pred_len={pred_len}")
        print(f"\n{'=' * 60}")

        train_ds = ETTh1Dataset(_CSV_PATH, split="train", seq_len=seq_len, pred_len=pred_len)
        val_ds = ETTh1Dataset(_CSV_PATH, split="val", seq_len=seq_len, pred_len=pred_len)
        test_ds = ETTh1Dataset(_CSV_PATH, split="test", seq_len=seq_len, pred_len=pred_len)

        loader_kwargs = {
            "batch_size": CONFIG["batch_size"],
            "num_workers": CONFIG["num_workers"],
            "pin_memory": device.type == "cuda",
        }

        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

        model = ChannelIndependentLinear(seq_len=seq_len, pred_len=pred_len).to(device)
        checkpoint_path = _CHECKPOINT_DIR / f"linear_seq{seq_len}_pred_len{pred_len}.pt"

        _train(model, train_loader, val_loader, CONFIG, device, checkpoint_path)

        test_mse, test_mae = _evaluate(model, test_loader, device, checkpoint_path)
        print(f"\nTest | MSE {test_mse:.4f} | MAE {test_mae:.4f}")

        _append_results(
            _RESULTS_CSV,
            {
                "seq_len": seq_len,
                "pred_len": pred_len,
                "test_mse": round(test_mse, 6),
                "test_mae": round(test_mse, 6),
                "timestamp": time.strftime("%Y_%m_%d_%H_%M_%S"),
            },
        )

        print(f"Results appended to {_RESULTS_CSV}")
