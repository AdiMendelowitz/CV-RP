"""
SimCLR representation analysis.

Generates four plots saved to ./plots/:
  1. pretrain_loss_curve.png     — NT-Xent loss at logged checkpoint epochs
  2. tsne_embeddings.png         — t-SNE of frozen encoder features coloured by class
  3. nearest_neighbours.png      — cosine nearest-neighbour retrieval grid
  4. label_fraction_accuracy.png — linear eval accuracy vs fraction of labeled data

All plots are suitable for direct embedding in README.md.

Usage:
    python analysis.py --checkpoint checkpoints/simclr/simclr_epoch100.pt
    python analysis.py --checkpoint checkpoints/simclr/simclr_epoch100.pt --tsne-samples 1000
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from linear_eval import (
    ENCODER_DIMS,
    evaluate,
    extract_features,
    get_transforms,
    load_frozen_encoder,
    train_linear,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Sparse loss log — recorded at checkpoint intervals during pretraining.
# Full per-epoch history was not persisted; these 6 points capture the
# key phases of the learning curve.
_TRAINING_LOG: dict[int, float] = {
    1: 5.3908,
    10: 4.9881,
    25: 4.8789,
    50: 4.7932,
    75: 4.7463,
    100: 4.7313,
}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def _setup_style() -> None:
    """Apply consistent matplotlib styling across all plots."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 2,
        }
    )


_CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _balanced_subsample(labels: torch.Tensor, n_total: int, seed: int = 42) -> torch.Tensor:
    """Return indices for a class-balanced subsample of n_total items.

    Args:
        labels: Integer class labels of shape (N,).
        n_total: Approximate total number of samples to return.
        seed: RNG seed for reproducibility.

    Returns:
        1-D index tensor of length <= n_total.
    """
    per_class = n_total // 10
    rng = torch.Generator().manual_seed(seed)
    indices = []
    for cls in range(10):
        cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
        n = min(per_class, len(cls_idx))
        perm = torch.randperm(len(cls_idx), generator=rng)[:n]
        indices.append(cls_idx[perm])
    return torch.cat(indices)


def _stratified_subsample(
    features: torch.Tensor,
    labels: torch.Tensor,
    fraction: float,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a class-balanced subsample at the given fraction.

    Args:
        features: Feature matrix of shape (N, D).
        labels: Integer class labels of shape (N,).
        fraction: Fraction of each class to retain; at least 1 sample per class.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (subsampled_features, subsampled_labels).
    """
    rng = torch.Generator().manual_seed(seed)
    indices = []
    for cls in range(10):
        cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
        n = max(1, int(len(cls_idx) * fraction))
        perm = torch.randperm(len(cls_idx), generator=rng)[:n]
        indices.append(cls_idx[perm])
    idx = torch.cat(indices)
    return features[idx], labels[idx]


# ---------------------------------------------------------------------------
# Plot 1: Training loss curve
# ---------------------------------------------------------------------------


def plot_loss_curve(save_dir: Path) -> None:
    """Plot NT-Xent loss vs epoch from the sparse training log.

    Args:
        save_dir: Directory where the PNG will be saved.
    """
    epochs = list(_TRAINING_LOG.keys())
    losses = list(_TRAINING_LOG.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        epochs,
        losses,
        marker="o",
        color="steelblue",
        markersize=6,
        label="NT-Xent loss",
    )

    # Annotate final value
    ax.annotate(
        f"{losses[-1]:.4f}",
        (epochs[-1], losses[-1]),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=9,
        color="steelblue",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent Loss")
    ax.set_title("SimCLR Pretraining Loss — CIFAR-10, ResNet-18, 100 epochs")
    ax.set_xlim(-2, 107)
    ax.legend()
    fig.tight_layout()

    path = save_dir / "pretrain_loss_curve.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2: t-SNE embeddings
# ---------------------------------------------------------------------------


def plot_tsne(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_samples: int,
    save_dir: Path,
) -> None:
    """t-SNE projection of encoder features coloured by CIFAR-10 class.

    Args:
        features: Feature matrix of shape (N, D), stored on CPU.
        labels: Integer class labels of shape (N,).
        n_samples: Number of test samples to embed (balanced across classes).
        save_dir: Directory where the PNG will be saved.
    """
    from sklearn.manifold import TSNE  # optional dependency — import locally

    indices = _balanced_subsample(labels, n_samples)
    feat_np = features[indices].numpy()
    lab_np = labels[indices].numpy()

    logger.info(f"Running t-SNE on {len(feat_np)} samples (perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    embedded = tsne.fit_transform(feat_np)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
        mask = lab_np == cls_idx
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            s=8,
            alpha=0.6,
            label=cls_name,
            color=_CLASS_COLORS[cls_idx],
        )

    ax.set_title(f"t-SNE of SimCLR Encoder Features — {n_samples} test samples")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(False)
    ax.legend(markerscale=2.5, fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    path = save_dir / "tsne_embeddings.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3: Nearest-neighbour retrieval
# ---------------------------------------------------------------------------


def plot_nearest_neighbours(
    features: torch.Tensor,
    labels: torch.Tensor,
    raw_dataset: torch.utils.data.Dataset,
    n_queries: int,
    n_neighbours: int,
    save_dir: Path,
) -> None:
    """Display cosine nearest-neighbour retrieval results in embedding space.

    One query image is selected per class (first n_queries classes). The top-k
    nearest neighbours are retrieved by cosine similarity and displayed alongside.
    A red border distinguishes the query column.

    Args:
        features: Encoder feature matrix of shape (N, D), CPU.
        labels: Integer class labels of shape (N,).
        raw_dataset: Test set returning (ToTensor image, label) — no normalization.
        n_queries: Number of query images (one per class starting from class 0).
        n_neighbours: Number of neighbours to retrieve per query.
        save_dir: Directory where the PNG will be saved.
    """
    # L2-normalise for cosine similarity via matrix multiply
    norm_feats = F.normalize(features, dim=1)

    # One query per class, take the first example of each
    query_indices = [(labels == cls).nonzero(as_tuple=True)[0][0].item() for cls in range(n_queries)]

    # Similarity: (n_queries, N)
    sim = norm_feats[query_indices] @ norm_feats.T

    fig, axes = plt.subplots(
        n_queries,
        n_neighbours + 1,
        figsize=(2.2 * (n_neighbours + 1), 2.2 * n_queries),
    )

    for row, q_idx in enumerate(query_indices):
        # Mask self out before taking top-k
        row_sim = sim[row].clone()
        row_sim[q_idx] = -1.0
        top_k = row_sim.topk(n_neighbours).indices.tolist()

        for col, img_idx in enumerate([q_idx] + top_k):
            ax = axes[row, col]
            image, _ = raw_dataset[img_idx]  # (C, H, W) float32

            if col == 0:
                # Red border: pad image array with a 2-pixel red frame
                img_np = image.permute(1, 2, 0).numpy()
                b = 2
                bordered = np.ones((img_np.shape[0] + 2 * b, img_np.shape[1] + 2 * b, 3))
                bordered[:, :] = [1.0, 0.0, 0.0]
                bordered[b:-b, b:-b] = img_np
                ax.imshow(bordered)
                ax.set_title(f"Query\n{CIFAR10_CLASSES[labels[q_idx]]}", fontsize=8)
            else:
                ax.imshow(image.permute(1, 2, 0).numpy())
                match = "✓" if labels[img_idx] == labels[q_idx] else "✗"
                ax.set_title(f"{match} {CIFAR10_CLASSES[labels[img_idx]]}", fontsize=8)

            ax.axis("off")

    fig.suptitle(
        "Nearest-Neighbour Retrieval in SimCLR Embedding Space (cosine similarity)\n"
        "✓ = correct class  ✗ = different class",
        fontsize=11,
    )
    fig.tight_layout()

    path = save_dir / "nearest_neighbours.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 4: Label fraction accuracy
# ---------------------------------------------------------------------------


def plot_label_fraction(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    encoder_name: str,
    device: torch.device,
    save_dir: Path,
) -> None:
    """Plot linear eval accuracy across a range of labeled data fractions.

    Trains a fresh linear classifier at each fraction using 30 epochs — enough
    to converge for small subsets, fast enough to run serially on CPU.
    This demonstrates the label efficiency of the pretrained representations.

    Args:
        train_features: Pre-extracted training features, shape (50000, D).
        train_labels: Training labels, shape (50000,).
        test_features: Pre-extracted test features, shape (10000, D).
        test_labels: Test labels, shape (10000,).
        encoder_name: Architecture name used to look up feature dimension.
        device: Target device.
        save_dir: Directory where the PNG will be saved.
    """
    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
    accuracies: list[float] = []

    for frac in fractions:
        feat_sub, lab_sub = _stratified_subsample(train_features, train_labels, frac)
        n_labeled = len(feat_sub)

        # Namespace with only the fields train_linear reads
        eval_args = argparse.Namespace(epochs=30, batch_size=256, lr=0.1)
        classifier = nn.Linear(ENCODER_DIMS[encoder_name], 10)
        train_linear(classifier, feat_sub, lab_sub, eval_args, device)
        acc = evaluate(classifier, test_features, test_labels, device) * 100
        accuracies.append(acc)
        logger.info(f"  {frac:.0%} labels ({n_labeled:,} samples): {acc:.2f}%")

    label_counts = [int(50000 * f) for f in fractions]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(
        label_counts,
        accuracies,
        marker="o",
        color="steelblue",
        markersize=7,
        label="SimCLR encoder (frozen)",
    )

    for x, y in zip(label_counts, accuracies):
        ax.annotate(
            f"{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(5, 6),
            fontsize=9,
            color="steelblue",
        )

    # Reference lines
    ax.axhline(
        93.43,
        linestyle="--",
        color="tomato",
        alpha=0.8,
        label="Supervised ResNet-18 (93.43%)",
    )
    ax.axhline(10.0, linestyle=":", color="gray", alpha=0.6, label="Random encoder (~10%)")

    ax.set_xlabel("Number of labeled training samples (log scale)")
    ax.set_ylabel("Top-1 Test Accuracy (%)")
    ax.set_title("SimCLR Label Efficiency — CIFAR-10")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = save_dir / "label_fraction_accuracy.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def analyse(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    _setup_style()

    # --- Plot 1: loss curve (no model needed) ---
    logger.info("Plot 1/4: Training loss curve")
    plot_loss_curve(save_dir)

    # --- Load encoder once — reused by all remaining plots ---
    encoder = load_frozen_encoder(args.checkpoint, args.encoder, device)

    # --- Datasets ---
    # Normalized: for feature extraction
    eval_transform = get_transforms(train=False)
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=eval_transform)
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=eval_transform)

    # Raw (ToTensor only, no normalization): for image display in plot 3
    raw_test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers)

    # --- Extract features once — shared by plots 2, 3, 4 ---
    logger.info("Extracting train features (50,000 images)...")
    train_features, train_labels = extract_features(encoder, train_loader, device)

    logger.info("Extracting test features (10,000 images)...")
    test_features, test_labels = extract_features(encoder, test_loader, device)

    # --- Plot 2: t-SNE ---
    logger.info("Plot 2/4: t-SNE embeddings")
    plot_tsne(test_features, test_labels, args.tsne_samples, save_dir)

    # --- Plot 3: nearest neighbours ---
    logger.info("Plot 3/4: Nearest-neighbour retrieval")
    plot_nearest_neighbours(
        test_features,
        test_labels,
        raw_test_dataset,
        n_queries=5,
        n_neighbours=5,
        save_dir=save_dir,
    )

    # --- Plot 4: label fraction ---
    logger.info("Plot 4/4: Label fraction accuracy (6 runs × 30 epochs each)")
    plot_label_fraction(
        train_features,
        train_labels,
        test_features,
        test_labels,
        args.encoder,
        device,
        save_dir,
    )

    logger.info(f"Done. All plots saved to {save_dir}/")


def parse_args() -> argparse.Namespace:
    _SCRIPT_DIR = Path(__file__).parent
    parser = argparse.ArgumentParser(description="SimCLR representation analysis")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(_SCRIPT_DIR / "checkpoints" / "simclr" / "simclr_epoch100.pt"),
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--encoder", type=str, default="resnet18", choices=list(ENCODER_DIMS.keys()))
    parser.add_argument(
        "--tsne-samples",
        type=int,
        default=2000,
        help="Number of test samples for t-SNE (balanced per class)",
    )
    parser.add_argument("--save-dir", type=str, default="./plots")
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    analyse(parse_args())
