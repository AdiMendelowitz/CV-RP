# code/compression/visualize_soft_targets.py
"""
Visualise how temperature T affects the teacher's soft target distributions.

For a sample of CIFAR-10 images, plots the teacher's softmax output at T=1, 2, 4, 8 side by side, to show how higher
T exposes inter-class similarity structure that T=1 buries near zero.

"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "computer-vision-foundations" / "code" / "pytorch_cnn"))
from resnet import resnet18 as custom_resnet18

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

_project_root = Path(__file__).resolve().parents[3]

TEACHER_CHECKPOINT = (
    _project_root
    / "computer-vision-foundations"
    / "code"
    / "pytorch_cnn"
    / "best_resnet18_cifar10 (1).pth"
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
TEMPERATURES = [1, 2, 4, 8]
N_SAMPLES = 4  # number of images to visualise
SAVE_DIR = Path(__file__).resolve().parent / "plots" / "distillation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_teacher(checkpoint_path: str, device: torch.device) -> nn.Module:
    model = custom_resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_soft_targets( logits: torch.Tensor, temperatures: list[int]) -> dict[int, np.ndarray]:
    """Return softmax probabilities at each temperature. Shape: (10,) per T."""
    return {T: F.softmax(logits / T, dim=1).squeeze().numpy() for T in temperatures}


def plot_sample(ax_row: list, image_ax, image: torch.Tensor, true_label: int, soft_targets: dict[int, np.ndarray],
                temperatures: list[int]) -> None:
    """Fill one row of subplots: image + one bar chart per temperature."""

    # Unnormalise for display
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    display_img = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    image_ax.imshow(display_img)
    image_ax.set_title(f"True: {CIFAR10_CLASSES[true_label]}", fontsize=8)
    image_ax.axis("off")

    for ax, T in zip(ax_row, temperatures):
        probs = soft_targets[T]
        colors = ["tomato" if i == true_label else "steelblue" for i in range(len(CIFAR10_CLASSES))]
        ax.bar(range(10), probs, color=colors)
        ax.set_xticks(range(10))
        ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=6)
        ax.set_ylim(0, 1)
        ax.set_title(f"T={T}  (max={probs.max():.2f})", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if T != temperatures[0]:
            ax.set_yticklabels([])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cpu")

    print("Loading teacher")
    teacher = load_teacher(str(TEACHER_CHECKPOINT), device)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=tf)

    # Pick one correctly classified sample per class for the first N_SAMPLES classes.
    torch.manual_seed(SEED)
    samples: list[tuple[torch.Tensor, int]] = []
    seen_classes: set[int] = set()

    with torch.no_grad():
        for image, label in dataset:
            if label in seen_classes:
                continue
            logits = teacher(image.unsqueeze(0))
            if logits.argmax(dim=1).item() == label:
                samples.append((image, label, logits))
                seen_classes.add(label)
            if len(samples) == N_SAMPLES:
                break

    # Build figure: N_SAMPLES rows × (1 image + len(TEMPERATURES)) columns
    n_cols = 1 + len(TEMPERATURES)
    fig, axes = plt.subplots(N_SAMPLES, n_cols, figsize=(3 * n_cols, 3 * N_SAMPLES),
                             gridspec_kw={"width_ratios": [1] + [2] * len(TEMPERATURES)})
    fig.suptitle("Soft target distributions at different temperatures\n (red = true class, blue = other classes)",
                 fontsize=11, fontweight="bold")

    for row_idx, (image, label, logits) in enumerate(samples):
        soft_targets = get_soft_targets(logits, TEMPERATURES)
        plot_sample(ax_row=axes[row_idx, 1:], image_ax=axes[row_idx, 0], image=image, true_label=label,
                    soft_targets=soft_targets, temperatures=TEMPERATURES)

    plt.tight_layout()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / "soft_target_distributions.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()