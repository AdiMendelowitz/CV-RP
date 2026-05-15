# analyze_results.py
"""Post-training analysis for Week 4 ETTh1 experiments.

Produces three figures saved to results/plots/:
  1. forecast_vs_truth_pred<N>.png  -- 3 representative test Windows
  2. val_mse_curves.png             -- validation MSE learning curves for all logged runs
  3. per_channel_mse_pred<N>.png    -- per-channel test MSE bar chart

Usage:
    python analyze_results.py --model_type patchtst --pred_len 96 (or 192, 336, 720)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.ett_dataset import ETTh1Dataset  # noqa: E402
from models.patchtst import PatchTST  # noqa: E402

# ETTh1 variate names in column order (excluding the date column).
VARIATE_NAMES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

SEQ_LEN_DEFAULTS = {"patchtst": 512, "itransformer": 96, "timemixer": 512}


def _load_model(model_type: str, pred_len: int, num_variates: int, seq_len: int, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load a model from a checkpoint.

    Args:
        model_type: Architecture name.
        pred_len: Forecast horizon.
        num_variates: Number of input channels.
        seq_len: Input sequence length.
        checkpoint_path: Path to the saved state_dict.
        device: Compute device.

    Returns:
        Model with loaded weights in eval mode.
    """
    if model_type == "patchtst":
        model = PatchTST(seq_len=seq_len, pred_len=pred_len, num_variates=num_variates)
    elif model_type == "itransformer":
        from models.itransformer import iTransformer

        model = iTransformer(seq_len=seq_len, pred_len=pred_len, num_variates=num_variates)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def _plot_forecast_windows(model: nn.Module, test_ds: ETTh1Dataset, pred_len: int, model_type: str,
                           plots_dir: Path, device: torch.device) -> None:
    """Plot 3 representative forecast windows (start, middle, end of test split).

    Args:
        model: Trained model in eval mode.
        test_ds: Test split dataset.
        pred_len: Forecast horizon.
        model_type: Architecture name (used in filename).
        plots_dir: Directory to save the figure.
        device: Compute device.
    """
    n = len(test_ds)
    indices = [0, n // 2, n - 1]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            x, y = test_ds[idx]
            pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            truth = y.numpy()

            # Plot only the OT (oil temperature) channel -- index 6 -- as the primary target.
            ax.plot(truth[:, 6], label="Ground truth (OT)", color="steelblue")
            ax.plot(pred[:, 6], label="Forecast (OT)", color="darkorange", linestyle="--")
            ax.set_title(f"Test window {idx} (steps {idx} to {idx + pred_len})")
            ax.set_xlabel("Steps ahead")
            ax.set_ylabel("Normalised value")
            ax.legend(fontsize=8)

    fig.suptitle(f"{model_type.upper()} | pred_len={pred_len} | OT channel forecast", fontsize=12)
    plt.tight_layout()
    out = plots_dir / f"forecast_vs_truth_pred{pred_len}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _plot_val_mse_curves(results_dir: Path, plots_dir: Path, model_type: str) -> None:
    """Overlay validation MSE learning curves for all logged runs of a model type.

    Args:
        results_dir: Directory containing per-run CSV files.
        plots_dir: Directory to save the figure.
        model_type: Architecture name -- used to glob matching CSV files.
    """
    run_csvs = sorted(results_dir.glob(f"{model_type}_pred*.csv"))
    if not run_csvs:
        print(f"No result CSVs found for {model_type} in {results_dir}. Skipping learning curve plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for csv_path in run_csvs:
        df = pd.read_csv(csv_path)
        pred_len = csv_path.stem.split("pred")[-1]
        ax.plot(df["epoch"], df["val_mse"], label=f"pred_len={pred_len}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE")
    ax.set_title(f"{model_type.upper()} validation MSE across horizons")
    ax.legend()
    plt.tight_layout()
    out = plots_dir / "val_mse_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _plot_per_channel_mse(model: nn.Module, test_ds: ETTh1Dataset, pred_len: int, model_type: str, plots_dir: Path, device: torch.device) -> None:
    """Compute and plot per-channel test MSE.

    Args:
        model: Trained model in eval mode.
        test_ds: Test split dataset.
        pred_len: Forecast horizon.
        model_type: Architecture name.
        plots_dir: Directory to save the figure.
        device: Compute device.
    """
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=False)
    channel_mse_sum = np.zeros(len(VARIATE_NAMES))
    n_samples = 0

    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).cpu().numpy()
            truth = y.numpy()
            diff_sq = (pred - truth) ** 2  # (B, pred_len, C)
            channel_mse_sum += diff_sq.mean(axis=(0, 1))  # mean over B and pred_len
            n_samples += 1

    per_channel_mse = channel_mse_sum / n_samples

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(VARIATE_NAMES, per_channel_mse, color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel("Variate")
    ax.set_ylabel("Test MSE")
    ax.set_title(f"{model_type.upper()} per-channel test MSE | pred_len={pred_len}")
    plt.tight_layout()
    out = plots_dir / f"per_channel_mse_pred{pred_len}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    print("\nPer-channel test MSE:")
    for name, mse in zip(VARIATE_NAMES, per_channel_mse):
        print(f"  {name}: {mse:.4f}")
    hardest = VARIATE_NAMES[int(np.argmax(per_channel_mse))]
    print(f"Hardest channel: {hardest}")


def analyze(model_type: str, pred_len: int) -> None:
    """Run full post-training analysis for one model/horizon combination.

    Args:
        model_type: Architecture name.
        pred_len: Forecast horizon.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = SEQ_LEN_DEFAULTS[model_type]

    results_dir = ROOT / "results"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = results_dir / "checkpoints" / f"{model_type}_pred{pred_len}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run train.py first.")

    data_path = ROOT / "data" / "ETTh1.csv"
    test_ds = ETTh1Dataset(data_path, split="test", seq_len=seq_len, pred_len=pred_len)
    num_variates = test_ds.num_features

    model = _load_model(model_type, pred_len, num_variates, seq_len, checkpoint_path, device)

    _plot_forecast_windows(model, test_ds, pred_len, model_type, plots_dir, device)
    _plot_val_mse_curves(results_dir, plots_dir, model_type)
    _plot_per_channel_mse(model, test_ds, pred_len, model_type, plots_dir, device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-training analysis for ETTh1 experiments.")
    parser.add_argument("--model_type", type=str, required=True, choices=["patchtst", "itransformer", "timemixer"])
    parser.add_argument("--pred_len", type=int, required=True, choices=[96, 192, 336, 720])
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    analyze(model_type=args.model_type, pred_len=args.pred_len)