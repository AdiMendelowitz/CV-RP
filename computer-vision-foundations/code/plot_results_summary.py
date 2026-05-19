"""
Generate results summary figure for computer-vision-foundations.

Two panels:
  Left  — ResNet-18 training curves (loss + accuracy) from training_history.json
  Right — Accuracy comparison across all four models

Usage:
    python plot_results_summary.py
    python plot_results_summary.py --history path/to/training_history.json
    python plot_results_summary.py --output path/to/output.png

Output: results_summary.png (saved alongside this script by default)
"""

import argparse
import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Known results — update if you re-run any model
# ---------------------------------------------------------------------------
FINAL_RESULTS: list[dict] = [
    {"model": "CNN\n(NumPy)", "accuracy": 90.94, "color": "#6C8EBF"},
    {"model": "ResNet-18", "accuracy": 93.43, "color": "#82B366"},
    {"model": "ViT-Tiny", "accuracy": 86.70, "color": "#D6A520"},
    {"model": "SimCLR\n(linear eval)", "accuracy": 68.23, "color": "#AE4132"},
]

# Keys expected in training_history.json
_REQUIRED_KEYS = ("train_loss", "test_loss", "train_acc", "test_acc")


def load_history(path: Path) -> dict[str, list[float]]:
    """Load and validate training history JSON.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if required keys are missing or list lengths are inconsistent.
    """
    if not path.exists():
        raise FileNotFoundError(f"Training history not found: {path}")

    with open(path) as f:
        history: dict[str, list[float]] = json.load(f)

    missing = [k for k in _REQUIRED_KEYS if k not in history]
    if missing:
        raise ValueError(f"training_history.json is missing keys: {missing}")

    lengths = {k: len(history[k]) for k in _REQUIRED_KEYS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"History lists have inconsistent lengths: {lengths}")

    return history


def plot_training_curves(ax_loss: plt.Axes, ax_acc: plt.Axes, history: dict[str, list[float]]) -> None:
    n_epochs = len(history["train_loss"])
    epochs = range(1, n_epochs + 1)
    color = "#82B366"

    # --- Loss ---
    ax_loss.plot(epochs, history["train_loss"], color=color, linewidth=1.8, label="Train")
    ax_loss.plot(epochs, history["test_loss"], color=color, linewidth=1.8, linestyle="--", alpha=0.7, label="Test")
    ax_loss.set_ylabel("Loss", fontsize=11)
    ax_loss.set_title("ResNet-18 — Training Curves", fontsize=12, fontweight="bold", pad=10)
    ax_loss.legend(fontsize=9)
    ax_loss.set_xlim(1, n_epochs)
    ax_loss.grid(True, alpha=0.3, linestyle=":")
    ax_loss.spines[["top", "right"]].set_visible(False)

    # --- Accuracy ---
    train_acc_pct = [v * 100 for v in history["train_acc"]]
    test_acc_pct = [v * 100 for v in history["test_acc"]]
    best_test_acc = max(test_acc_pct)

    ax_acc.plot(epochs, train_acc_pct, color=color, linewidth=1.8, label="Train")
    ax_acc.plot(epochs, test_acc_pct, color=color, linewidth=1.8, linestyle="--", alpha=0.7, label="Test")
    ax_acc.axhline(y=best_test_acc, color="#555555", linewidth=0.8, linestyle=":", alpha=0.6)
    ax_acc.annotate(
        f"Best: {best_test_acc:.2f}%",
        xy=(n_epochs, best_test_acc),
        xytext=(-8, 6),
        textcoords="offset points",
        fontsize=8.5,
        color="#555555",
        ha="right",
    )
    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylim(0, 102)
    ax_acc.set_xlim(1, n_epochs)
    ax_acc.legend(fontsize=9)
    ax_acc.grid(True, alpha=0.3, linestyle=":")
    ax_acc.spines[["top", "right"]].set_visible(False)


def plot_accuracy_comparison(ax: plt.Axes) -> None:
    models = [r["model"] for r in FINAL_RESULTS]
    accuracies = [r["accuracy"] for r in FINAL_RESULTS]
    colors = [r["color"] for r in FINAL_RESULTS]

    x = np.arange(len(models))
    bars = ax.bar(x, accuracies, color=colors, width=0.55, edgecolor="white", linewidth=0.8, zorder=3)

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#222222",
        )

    # Combined xticks call avoids FixedLocator UserWarning (matplotlib >= 3.7)
    ax.set_xticks(x, labels=models, fontsize=10)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy Comparison — All Models", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(True, axis="y", alpha=0.3, linestyle=":", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    ax.text(
        0.5,
        -0.13,
        "* SimCLR: linear evaluation accuracy (encoder trained without labels)",
        transform=ax.transAxes,
        fontsize=8,
        color="#777777",
        ha="center",
        va="top",
        style="italic",
    )


def main(history_path: Path, output_path: Path) -> None:
    history = load_history(history_path)

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        left=0.07,
        right=0.97,
        top=0.92,
        bottom=0.10,
        hspace=0.45,
        wspace=0.32,
    )

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[:, 1])

    plot_training_curves(ax_loss, ax_acc, history)
    plot_accuracy_comparison(ax_bar)

    fig.suptitle(
        "Computer Vision Foundations — Results Summary",
        fontsize=14,
        fontweight="bold",
        y=0.97,
        color="#222222",
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate results summary figure.")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path(__file__).parent / "pytorch_cnn" / "training_history.json",
        help="Path to ResNet training_history.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results_summary.png",
        help="Output image path",
    )
    args = parser.parse_args()
    main(args.history, args.output)
