"""
FGSM epsilon sweep on CIFAR-10:
Runs FGSM at eps in {2, 4, 8, 16}/255 on the first 1000 CIFAR-10 test images, logs clean accuracy, avdersarial
accuracy and mean L-inf perturbation to experiments/results/clean_vs_avdversarial.csv, and saves a 10-example
side by side figure at eps = 8/255.

Deterministic by construction: fixed test subset, no shuffling, no random start.

Run from the toolkit root:
    python -m experiments.fgsm_sweep

Data and checkpoints paths default to their known repo locations. Override with CIFAR10_DATA_ROOT and
CIFAR10_RESENT18_CKPT if needed. Downloading is disabled.
"""

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacks.fgsm import fgsm  # noqa: E402
from models.normalized_model import NormalizedModel  # noqa: E402
from models.resnet import resnet18  # noqa: E402

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

CKPT_ENV = "CIFAR10_RESNET18_CKPT"
DATA_ENV = "CIFAR10_DATA_ROOT"
PYTORCH_CNN_DIR = REPO_ROOT / "computer-vision-foundations" / "code" / "pytorch_cnn"

CKPT_ENV = "CIFAR10_RESNET18_CKPT"
DATA_ENV = "CIFAR10_DATA_ROOT"
DEFAULT_CKPT = PYTORCH_CNN_DIR / "best_resnet18_cifar10 (1).pth"
DEFAULT_DATA_ROOT = PYTORCH_CNN_DIR / "data"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

EPS_LIST = (2/255, 4/255, 8/255, 16/255)
SELECTION_EPS = 8/255
NUM_SAMPLES = 1000
BATCH_SIZE = 250
NUM_FIGURE_EXAMPLES = 10

RESULTS_DIR = TOOLKIT_ROOT / "experiments" / "results"
CSV_PATH = RESULTS_DIR / "clean_vs_adversarial.csv"
FIGURE_PATH = RESULTS_DIR / "fgsm_examples_by_eps.png"

# (clean image on CPU, true label)
type Sample = tuple[Tensor, int]
# (eps, adversarial images on CPU, predicted labels)
type Panel = tuple[float, Tensor, list[int]]


def _resolve_data_root() -> Path:
    """Return the CIFAR-10 root, failing fast rather than downloading."""
    root = Path(os.environ.get(DATA_ENV, str(DEFAULT_DATA_ROOT)))
    if not (root / "cifar-10-batches-py").is_dir():
        raise RuntimeError(f"No cifar-10-batches-py directory under {root}. "
                           f"Set {DATA_ENV} to override. Downloading is disabled.")
    return root

def _resolve_checkpoint() -> Path:
    """Return the checkpoint path, honouring the env override used by the tests."""
    path = Path(os.environ.get(CKPT_ENV, str(DEFAULT_CKPT)))
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {path}. Set {CKPT_ENV} or restore the file.")
    return path


def _load_model(device: torch.device) -> NormalizedModel:
    """Load the trained ResNet-18, wrap it for input normalisation, set eval mode."""
    net = resnet18(num_classes=10)
    state = torch.load(_resolve_checkpoint(), map_location=device, weights_only=True)
    net.load_state_dict(state)
    model = NormalizedModel(net, CIFAR10_MEAN, CIFAR10_STD).to(device)
    model.eval()
    return model


def _build_loader() -> DataLoader:
    """First NUM_SAMPLES test images in [0, 1] pixel space, in dataset order."""
    dataset = CIFAR10(root=str(_resolve_data_root()), train=False, download=False, transform=transforms.ToTensor())
    subset = Subset(dataset, range(NUM_SAMPLES))
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


@torch.no_grad()
def _clean_pass(model: NormalizedModel, loader: DataLoader, device: torch.device) -> tuple[list[Tensor], float]:
    """Return per-batch clean predictions and overall clean accuracy."""
    correct, total = 0, 0
    preds_per_batch: list[Tensor] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        preds_per_batch.append(preds)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return preds_per_batch, correct / total


@torch.no_grad()
def _fgsm_pass(model: NormalizedModel, loader: DataLoader, clean_preds: list[Tensor], eps: float,
               device: torch.device, num_examples: int) -> tuple[float, float, list[Sample]]:
    """Adversarial accuracy, mean L-inf perturbation, and up to num_examples flips.

    Flip: a sample classified correctly clean but wrongly under attack. Reuses the stored clean predictions so flip
    selection cannot disagree with the reported clean accuracy; strict zip guards batch alignment.
    """
    correct, total, linf_sum = 0, 0 , 0.0
    examples: list[Sample] = []
    for (images, labels), batch_clean_preds in zip(loader, clean_preds, strict=True):
        images, labels = images.to(device), labels.to(device)
        adv = fgsm(model, images, labels, eps)
        adv_preds = model(adv).argmax(dim=1)
        correct += (adv_preds == labels).sum().item()
        total += labels.size(0)
        linf_sum += (adv - images).abs().amax(dim=(1, 2, 3)).sum().item()
        if len(examples) < num_examples:
            flipped = (batch_clean_preds == labels) & (adv_preds != labels)
            for idx in flipped.nonzero(as_tuple=True)[0].tolist():
                examples.append((images[idx].cpu(), labels[idx].item()))
                if len(examples) == num_examples:
                    break
    return correct / total, linf_sum / total, examples


@torch.no_grad()
def _adversarial_panels(model: NormalizedModel, samples: list[Sample], device: torch.device) -> list[Panel]:
    """Adversarial images and predictions for the selected samples at each eps.
    The model is in eval mode (BatchNorm on running statistics), so predictions do not depend on batch composition and
    match the sweep pass exactly.
    """
    images = torch.stack([image for image, _ in samples]).to(device)
    labels = torch.tensor([label for _, label in samples], device=device)
    panels: list[Panel] = []
    for eps in EPS_LIST:
        adv = fgsm(model, images, labels, eps)
        preds = model(adv).argmax(dim=1)
        panels.append((eps, adv.cpu(), preds.tolist()))
    return panels


def _style_cell(ax: Axes, row_label: str | None = None) -> None:
    """Hide ticks and spines; optionally set a row label via ylabel.
    Deliberately avoids axis("off"), which would also suppress the ylabel.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if row_label is not None:
        ax.set_ylabel(row_label, fontsize=10)


def _save_figure(samples: list[Sample], panels: list[Panel], path: Path) -> None:
    """Grid: originals on top, one adversarial row per eps, predictions coloured."""
    n = len(samples)
    n_rows = 1 + len(panels)
    fig, axes = plt.subplots(n_rows, n, figsize=(1.6 * n, 1.9 * n_rows), layout="constrained")
    for col, (clean, label) in enumerate(samples):
        ax = axes[0, col]
        ax.imshow(clean.permute(1, 2, 0).numpy())
        ax.set_title(CIFAR10_CLASSES[label], fontsize=9)
        _style_cell(ax, "clean" if col == 0 else None)
    for row, (eps, adv, preds) in enumerate(panels, start=1):
        for col in range(n):
            ax = axes[row, col]
            still_correct = preds[col] == samples[col][1]
            ax.imshow(adv[col].permute(1, 2, 0).numpy())
            ax.set_title(CIFAR10_CLASSES[preds[col]], fontsize=9, color="seagreen" if still_correct else "firebrick")
            _style_cell(ax, f"{round(eps * 255)}/255" if col == 0 else None)
    fig.suptitle("FGSM by epsilon: green predictions remain correct, red are flipped")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_csv(rows: list[list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["eps_x255", "eps", "clean_acc", "fgsm_acc", "mean_linf"])
        writer.writerows(rows)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = _load_model(device)
    loader = _build_loader()

    clean_preds, clean_acc = _clean_pass(model, loader, device)
    print(f"Clean accuracy on {NUM_SAMPLES} samples: {clean_acc:.4f}")

    rows: list[list[str]] = []
    figure_samples: list[Sample] | None = None
    for eps in EPS_LIST:
        examples_wanted = NUM_FIGURE_EXAMPLES if eps == SELECTION_EPS else 0
        fgsm_acc, mean_linf, examples = _fgsm_pass(model, loader, clean_preds, eps, device, examples_wanted)
        if examples_wanted:
            figure_samples = examples
        rows.append(
            [
                str(round(eps * 255)),
                f"{eps:.6f}",
                f"{clean_acc:.4f}",
                f"{fgsm_acc:.4f}",
                f"{mean_linf:.6f}",
            ]
        )
        print(
            f"eps = {round(eps * 255):>2}/255: "
            f"fgsm_acc = {fgsm_acc:.4f}, mean L-inf = {mean_linf:.6f}"
        )

    _write_csv(rows, CSV_PATH)
    print(f"Results written to {CSV_PATH}")

    if figure_samples is None:
        raise RuntimeError(
            f"SELECTION_EPS ({SELECTION_EPS}) does not appear in EPS_LIST; "
            "no figure samples were collected."
        )
    if len(figure_samples) < NUM_FIGURE_EXAMPLES:
        raise RuntimeError(
            f"Found only {len(figure_samples)} flipped examples at "
            f"eps = {round(SELECTION_EPS * 255)}/255; expected {NUM_FIGURE_EXAMPLES}."
        )
    panels = _adversarial_panels(model, figure_samples, device)
    _save_figure(figure_samples, panels, FIGURE_PATH)
    print(f"Figure saved to {FIGURE_PATH}")


if __name__ == "__main__":
    main()