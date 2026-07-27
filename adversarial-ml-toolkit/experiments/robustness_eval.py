"""White-box robustness evaluation on CIFAR-10.

Evaluates a naturally trained ResNet-18 against no attack, FGSM, and PGD at 20 and 50 steps, all at the standard
eps = 8/255 L-infinity threat model, on the first 1,000 CIFAR-10 test images. Results are logged to
experiments/results/robustness_table.csv with columns model, defense, attack, eps, steps, accuracy.

The PGD step size is fixed at 2/255 (Madry et al.'s CIFAR-10 value), so PGD-50 is a more thorough search of the same
eps-ball then PGD-20 rather than a differently scaled attack. Evaluation is deterministic: a fixed test subset with no
shuffling, and PGD seeds the global RNG per restart, so the table reproduces exactly under any row order or batch size.

PGD uses a single random start by default. The number of restarts is set by PGD_RESTARTS: with more than one, the
strongest (highest cross-entropy loss) result per sample is kept, which is the correct inner maximiser for evaluating
defended models. The naturally trained baseline collapses under a single start, so restarts are left at one here.

Run from the toolkit root:
    python -m experiments.robustness_eval

Data and checkpoint paths default to their known repo locations; override with
CIFAR10_DATA_ROOT and CIFAR10_RESNET18_CKPT if needed. Downloading is disabled.
"""

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacks.fgsm import fgsm
from attacks.pgd import pgd
from experiments._common import build_loader, load_model
from models.normalized_model import NormalizedModel

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "resnet18"
DEFENSE = "none"  # naturally trained baseline
EPS = 8/255
ALPHA = 2/255  # Madry et al. CIFAR-10 step size
PGD_STEP_COUNTS = (20, 50)
PGD_RESTARTS = 1
SEED = 0  # base seed; each PGD restart uses SEED + restart_index

RESULTS_DIR = TOOLKIT_ROOT / "experiments" / "results"
CSV_PATH = RESULTS_DIR / "robustness_table.csv"

# Maps clean images and labels to adversarial images.
type AttackFn = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One row of the robustness table: a labelled attack and its logged fields."""

    attack: str                 # CSV attack label
    steps: int                  # CSV steps field (0 clean, 1 FGSM, k for PGD)
    eps: float                  # CSV eps field (0.0 when no perturbation is applied)
    attack_fn: AttackFn | None  # None means evaluate on clean images


def _pgd_multi_restart(model: NormalizedModel, images: Tensor, labels: Tensor, eps: float,
                       alpha: float, steps: int, restarts: int, base_seed: int) -> Tensor:
    """PGD keeping, per sample, the highest cross-entropy loss over random restarts.

    A single restart returns the plain PGD result, and with more the strongest perturbation per sample is retained,
    which is the inner maximiser used when a weak single run would overstate a defended model's robustness.

    Each restart reseeds the global RNG to base_seed + restart_index before drawing its random start, so a given
    restart's initialisation is fixed independently of batch size or of how many batches precede it. The result is
    therefore identical under any batching of the same data.
    """
    def _one_restart(index: int) -> Tensor:
        torch.manual_seed(base_seed + index)
        return pgd(model, images, labels, eps, alpha, steps=steps)

    best_adv = _one_restart(0)
    if restarts == 1:
        return best_adv
    with torch.no_grad():
        best_loss = F.cross_entropy(model(best_adv), labels, reduction="none")
    for index in range(1, restarts):
        candidate = _one_restart(index)
        with torch.no_grad():
            loss = F.cross_entropy(model(candidate), labels, reduction="none")
        take = loss > best_loss
        best_loss = torch.where(take, loss, best_loss)
        mask = take.view(-1, *([1] * (images.dim() - 1)))
        best_adv = torch.where(mask, candidate, best_adv)
    return best_adv


def _accuracy(model: NormalizedModel, loader: DataLoader, device: torch.device, spec: Evaluation,) -> float:
    """Top-1 accuracy for one evaluation spec.

    The scoring forward pass runs under no_grad; attacks manage their own gradient context internally. PGD seeds itself
    per restart, and FGSM and the clean pass are deterministic, so no seeding is needed here.
    """
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv = spec.attack_fn(images, labels) if spec.attack_fn is not None else images
        with torch.no_grad():
            preds = model(adv).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def _evaluations(model: NormalizedModel) -> list[Evaluation]:
    """Ordered evaluation specs; closures capture the loaded model."""
    specs: list[Evaluation] = [
        Evaluation("none", 0, 0.0, None),
        Evaluation("fgsm", 1, EPS, lambda x, y: fgsm(model, x, y, EPS)),
    ]
    for k in PGD_STEP_COUNTS:
        specs.append(
            Evaluation("pgd", k, EPS,
                       lambda x, y, k=k: _pgd_multi_restart(model, x, y, EPS, ALPHA, k, PGD_RESTARTS, SEED),
                       )
        )
    return specs


def _write_csv(rows: list[list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "defense", "attack", "eps", "steps", "accuracy"])
        writer.writerows(rows)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(device)
    loader = build_loader()

    rows: list[list[str]] = []
    for spec in _evaluations(model):
        acc = _accuracy(model, loader, device, spec)
        rows.append([MODEL_NAME, DEFENSE, spec.attack, f"{spec.eps:.6f}", str(spec.steps), f"{acc:.4f}",])
        tag = f"pgd-{spec.steps}" if spec.attack == "pgd" else spec.attack
        print(f"{tag:>8}: accuracy = {acc:.4f}")

    _write_csv(rows, CSV_PATH)
    print(f"Results written to {CSV_PATH}")


if __name__ == "__main__":
    main()