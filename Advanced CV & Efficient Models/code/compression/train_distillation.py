"""
CIFAR-10 knowledge distillation: ResNet-18 teacher -> SmallCNN student.

Modes:
    distill: Train student with combined KD + CE loss.
    baseline: Train student with CE only (no teacher).
    both: Run both sequentially and plot comparison

Assumptions:
    - Teacher checkpoint was saved as {'model_state_dict':...,...} or as a plain state dict. Both handled by
        load_teacher_from_checkpoint().
    - Teacher architecture is torchvision ResNet-18. For a custom ResNet-18 implementation, state dict keys must
        match torchvision's.
"""

import argparse
import logging
import random
from pathlib import Path

_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "computer-vision-foundations" / "code" / "pytorch_cnn"))
from resnet import resnet18 as custom_resnet18

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from distillation import build_student


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%d/%m/%Y %H:%M:%S")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def get_cifar10_loaders(data_dir: str = str(_DATA_ROOT), batch_size: int = 128,
                        num_workers: int = 2) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for CIFAR-10."""

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    val_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ----------------------------------------------------------------------------------------------------------------------
# Training loops
# ----------------------------------------------------------------------------------------------------------------------

def _train_epoch_distill(student: nn.Module, teacher: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                         device: torch.device, T: float, alpha: float) -> tuple[float, float, float, float]:
    """
    One distillation epoch.

    Returns:
        (combined loss, kd_loss, ce_loss, accuracy) - all averged over the epoch.
    """

    student.train()
    total_combined, total_kd, total_ce, correct, n = 0, 0, 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher(images)

        student_logits = student(images)

        kd = (
            F.kl_div(
                F.log_softmax(student_logits/T, dim=1),
                F.softmax(teacher_logits/T, dim=1),
                reduction='batchmean',
            )
            * (T**2)
        )

        ce = F.cross_entropy(student_logits, labels)
        loss = alpha * kd + (1-alpha) * ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_combined += bs * loss.item()
        total_kd += bs * kd.item()
        total_ce += bs * ce.item()
        correct += (student_logits.argmax(dim=1) == labels).sum().item()
        n += bs

    return total_combined/n, total_kd/n, total_ce/n, correct/n

def _train_epoch_baseline(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                          device: torch.device) -> tuple[float, float]:
    """
    One standard CE epoch.

    Returns:
        (ce_loss, accuracy)
    """

    model.train()
    total_ce, correct, n = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_ce += bs * loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += bs

    return total_ce/n, correct/n

def _val_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Validation pass. Returns (ce_loss, accuracy)."""

    model.eval()
    total_ce, correct, n = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            bs = images.size(0)
            total_ce += bs * loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            n += bs

    return total_ce/n, correct/n


# ----------------------------------------------------------------------------------------------------------------------
# Run helpers
# ----------------------------------------------------------------------------------------------------------------------

def run_distillation(teacher: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                     epochs: int, T: float, alpha: float, lr: float, checkpoint_dir: Path,
                     student_arch: str) -> dict[str, list[float]]:
    """Full distillation run. Returns history dict."""

    student = build_student(student_arch, num_classes=10).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in student.parameters())
    logger.info("Student (%s): %s parameters", student_arch, f"{n_params:,}")

    history: dict[str, list[float]] = {"train_loss": [], "kd_loss": [], "ce_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, kd_loss, ce_loss, tr_acc = _train_epoch_distill(
            student, teacher, train_loader, optimizer, device, T, alpha
        )

        val_loss, val_acc = _val_epoch(student, val_loader, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["kd_loss"].append(kd_loss)
        history["ce_loss"].append(ce_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info("Distill [%d/%d]    train=%.4f  kd=%.4f  ce=%.4f  val_acc=%.4f",
                    epoch, epochs, tr_loss, kd_loss, ce_loss, val_acc*100)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = checkpoint_dir / "best_student_distill.pth"

            torch.save(
                {"epoch": epoch, "val_acc": val_acc, "model_state_dict": student.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict()
                 },
                ckpt,
            )

    logger.info("Distillation complete. Vest val_acc=%.2f%%", best_val_acc*100)
    return history

def run_baseline(train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int,
                lr: float, checkpoint_dir: Path, student_arch: str) -> dict[str, list[float]]:
    """Standard CE baseline(same student architecture, no teacher). Return history dict."""

    student = build_student(student_arch, num_classes=10).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, _tr_acc = _train_epoch_baseline(student, train_loader, optimizer, device)
        val_loss, val_acc = _val_epoch(student, val_loader, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info("Baseline [%d/%d]   train=%.4f  val_acc=%.2f", epoch, epochs, tr_loss, val_acc*100)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = checkpoint_dir / "best_student_baseline.pth"
            torch.save(
                {"epoch": epoch, "val_acc": val_acc, "model_state_dict": student.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict()},
                ckpt,
            )

    logger.info("Baseline complete. Vest val_acc=%.2f", best_val_acc*100)
    return history



# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(distill_history: dict[str, list[float]] | None, baseline_history: dict[str, list[float]] | None,
                 save_dir: Path) -> None:
    """Save two figures: loss breakdown and val accuracy comparison."""
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(
        1,
        len((distill_history or baseline_history)["val_acc"]) + 1,  # type: ignore[index]
    )

    # --- Figure 1: distillation loss breakdown ---
    if distill_history is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, distill_history["train_loss"], label="Combined (αKD + (1-α)CE)", linewidth=2)
        ax.plot(epochs, distill_history["kd_loss"], label="KD loss (T²·KL)", linestyle="--")
        ax.plot(epochs, distill_history["ce_loss"], label="CE loss", linestyle=":")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Distillation: loss components per epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(save_dir / "distill_loss_breakdown.png", dpi=150, bbox_inches="tight")
        plt.close()

    # --- Figure 2: val accuracy comparison ---
    fig, ax = plt.subplots(figsize=(8, 4))

    if distill_history is not None:
        d_acc = [v * 100 for v in distill_history["val_acc"]]
        ax.plot(epochs, d_acc, label=f"Distillation (best {max(d_acc):.2f}%)", linewidth=2)

    if baseline_history is not None:
        b_acc = [v * 100 for v in baseline_history["val_acc"]]
        ax.plot(epochs, b_acc, label=f"Baseline CE (best {max(b_acc):.2f}%)", linestyle="--", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Student val accuracy: distillation vs baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / "val_accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Plots saved to %s", save_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    _project_root = Path(__file__).resolve().parents[3]
    _default_checkpoint = (
        _project_root
        / "computer-vision-foundations"
        / "code"
        / "pytorch_cnn"
        / "best_resnet18_cifar10 (1).pth"
    )
    p = argparse.ArgumentParser(description="Knowledge distillation on CIFAR-10")
    p.add_argument("--mode", choices=["distill", "baseline", "both"], default="both")
    p.add_argument("--checkpoint", type=str, default=str(_default_checkpoint))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.3)  # reduced from 0.7
    p.add_argument("--student-arch", type=str, default="small_cnn")
    p.add_argument("--data-dir", type=str, default=str(_DATA_ROOT))
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/distillation")
    p.add_argument("--plot-dir", type=str, default="plots/distillation")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode in ("distill", "both") and not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_cifar10_loaders(data_dir=args.data_dir, batch_size=args.batch_size,
                                                   num_workers=args.num_workers)

    distill_history, baseline_history = None, None

    if args.mode in ("distill", "both"):
        logger.info("Loading teacher from %s", args.checkpoint)
        teacher = custom_resnet18(num_classes=10)
        teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        teacher.maxpool = nn.Identity()
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        teacher.load_state_dict(checkpoint)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        n_teacher = sum(p.numel() for p in teacher.parameters())
        logger.info("Teacher (ResNet-18): %s parameters", f"{n_teacher:,}")
        logger.info("Distillation settings: T=%.1f  alpha=%.2f", args.temperature, args.alpha)

        distill_history = run_distillation(teacher=teacher, train_loader=train_loader, val_loader=val_loader,
                                           device=device, epochs=args.epochs, T=args.temperature, alpha=args.alpha,
                                           lr=args.lr, checkpoint_dir=checkpoint_dir, student_arch=args.student_arch)

    if args.mode in ("baseline", "both"):
        logger.info("Running baseline (CE only, no teacher)")
        baseline_history = run_baseline(train_loader=train_loader, val_loader=val_loader, device=device,
                                        epochs=args.epochs, lr=args.lr, checkpoint_dir=checkpoint_dir,
                                        student_arch=args.student_arch)

    if distill_history is not None and baseline_history is not None:
        best_distill = max(distill_history["val_acc"]) * 100
        best_baseline = max(baseline_history["val_acc"]) * 100
        logger.info("Gap: distillation %.2f%% vs baseline %.2f%% (Δ=%.2f%%)",
                    best_distill, best_baseline, best_distill - best_baseline)

    plot_results(distill_history, baseline_history, save_dir=Path(args.plot_dir))


if __name__ == "__main__":
    main()

