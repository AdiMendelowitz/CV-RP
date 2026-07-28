"""White-box robustness evaluation on CIFAR-10.

Evaluates a naturally trained ResNet-18 against no attack, FGSM, PGD (20 and 50 steps), and the Carlini-Wagner L2 attack
on the first 1,000 CIFAR-10 test images. FGSM and PGD use the standard eps = 8/255 L-infinity threat model; C&W is an L2
attack with no L-infinity budget. Results are logged to experiments/results/robustness_table.csv with columns model,
defense, attack, eps, steps, accuracy, success_rate, mean_linf, mean_l2.

Both perturbation norms are reported for every attack, measured from the returned adversarial images rather than
assumed, so the L-infinity attacks and the L2 attack can be compared honestly rather than forced onto a shared budget.
The norms are averaged over successful samples only, where a success is a sample classified correctly before the attack
and incorrectly after it, so they measure the distortion a successful attack costs rather than being diluted by samples
the attack left unchanged. success_rate reports the fraction of attackable samples that were broken. The eps column is
blank for attacks with no L-infinity budget; steps holds the iteration count where it applies.

The PGD step size is fixed at 2/255 (Madry et al.'s CIFAR-10 value), so PGD-50 is a more thorough search of the same
eps-ball then PGD-20 rather than a differently scaled attack. Evaluation is deterministic at a fixed batch size and
ordering: a fixed test subset with no shuffling, and PGD reseeds the global RNG per restart. C&W is deterministic
given the model and input.

PGD uses a single random start by default. The number of restarts is set by PGD_RESTARTS: with more than one, the
strongest (highest cross-entropy loss) result per sample is kept, which is the correct inner maximiser for evaluating
defended models. The naturally trained baseline collapses under a single start, so restarts are left at one here.

Run from the toolkit root:
    python -m experiments.robustness_eval

Data and checkpoint paths default to their known repo locations; override with
CIFAR10_DATA_ROOT and CIFAR10_RESNET18_CKPT if needed. Downloading is disabled.
"""

import csv
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacks.cw import cw
from attacks.fgsm import fgsm
from attacks.pgd import pgd
from experiments._common import build_loader, load_model
from models.normalized_model import NormalizedModel

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "resnet18"
DEFENSE = "none"  # naturally trained baseline
EPS = 8 / 255
ALPHA = 2 / 255  # Madry et al. CIFAR-10 step size
PGD_STEP_COUNTS = (20, 50)
PGD_RESTARTS = 1
SEED = 0  # base seed; each PGD restart uses SEED + restart_index

CW_C = 1.0
CW_STEPS = 100

RESULTS_DIR = TOOLKIT_ROOT / "experiments" / "results"
CSV_PATH = RESULTS_DIR / "robustness_table.csv"

# Maps clean images and labels to adversarial images.
type AttackFn = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One evaluation spec: a labelled attack and the fields describing it."""

    attack: str                 # CSV attack label
    steps: int                  # iteration count (0 clean, 1 FGSM, k for PGD/C&W)
    eps: float                  # configured L-inf budget; nan when not an L-inf attack
    attack_fn: AttackFn | None  # None means evaluate on clean images


@dataclass(frozen=True, slots=True)
class Result:
    """Measured outcome of one evaluation over the test subset."""

    accuracy: float  # top-1 accuracy under the attack
    success_rate: float  # fraction of clean-correct samples the attack broke (0.0 for the clean pass)
    mean_linf: float  # mean L-inf over broken samples (0.0 if none)
    mean_l2: float  # mean L2 over broken samples (0.0 if none)


@torch.no_grad()
def _clean_predictions(model: NormalizedModel, loader: DataLoader, device: torch.device) -> list[Tensor]:
    """Per-batch clean predictions, computed once and shared by every evaluation.

    Reusing one set of clean predictions keeps the clean row, the success rate, and the norm mask
    consistent with each other, and matches the pattern in fgsm_sweep.py.
    """
    return [model(images.to(device)).argmax(dim=1) for images, _ in loader]


def _pgd_multi_restart(model: NormalizedModel, images: Tensor, labels: Tensor, eps: float, alpha: float,
                       steps: int, restarts: int, base_seed: int) -> Tensor:
    """PGD keeping, per sample, the highest cross-entropy loss over random restarts.

    A single restart returns the plain PGD result. With more, the strongest perturbation per sample is retained, which
    is the inner maximiser used when a weak single run would overstate a defended model's robustness.

    Each restart reseeds the global RNG to base_seed + restart_index before drawing its random start, so a given restart
    is reproducible across runs at a fixed batch size and ordering. The reseed happens once per call, and a call handles
    one batch, so the random start is not invariant to a change of batch size; making it so would require a
    dedicated generator advanced per sample.
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


def _evaluate(model: NormalizedModel, loader: DataLoader, device: torch.device, spec: Evaluation,
              clean_preds: list[Tensor]) -> Result:
    """Accuracy, success rate, and mean L-inf and L2 over successfully attacked samples.

    Success: a sample the model classified correctly before the attack and incorrectly after it. The attack takes no
    credit for the model's pre-existing errors and success_rate is not a restatement of 1 - accuracy. The rate is
    reported over the attackable samples, those correct on clean input. Norms are measured from the returned adversarial
    images and averaged over successes only, so they report what a successful attack costs rather than being diluted by
    samples the attack left unchanged (C&W returns the clean image both for samples it never flips and for samples that
    were already misclassified, where its margin is satisfied at the first iteration).

    The scoring forward pass runs under no_grad; attacks manage their own gradient context. The clean pass has no
    attack, so its success rate and norms are zero.
    """
    correct, total, attackable_total, flipped_total = 0, 0, 0, 0
    linf_sum, l2_sum = 0.0, 0.0
    for (images, labels), batch_clean_preds in zip(loader, clean_preds, strict=True):
        images, labels = images.to(device), labels.to(device)
        clean_correct = batch_clean_preds == labels
        if spec.attack_fn is None:
            correct += clean_correct.sum().item()
            total += labels.size(0)
            continue

        adv = spec.attack_fn(images, labels)
        with torch.no_grad():
            preds = model(adv).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        attackable_total += clean_correct.sum().item()

        flipped = clean_correct & (preds != labels)
        flipped_total += flipped.sum().item()
        if flipped.any():
            delta = (adv - images).flatten(1)[flipped]
            linf_sum += delta.abs().amax(dim=1).sum().item()
            l2_sum += delta.norm(dim=1).sum().item()

    accuracy = correct / total
    success_rate = flipped_total / attackable_total if attackable_total else 0.0
    mean_linf = linf_sum / flipped_total if flipped_total else 0.0
    mean_l2 = l2_sum / flipped_total if flipped_total else 0.0
    return Result(accuracy, success_rate, mean_linf, mean_l2)


def _evaluations(model: NormalizedModel) -> list[Evaluation]:
    """Ordered evaluation specs; closures capture the loaded model."""
    specs: list[Evaluation] = [
        Evaluation("none", 0, math.nan, None),
        Evaluation("fgsm", 1, EPS, lambda x, y: fgsm(model, x, y, EPS)),
    ]
    for k in PGD_STEP_COUNTS:
        specs.append(
            Evaluation("pgd", k, EPS,
                       lambda x, y, k=k: _pgd_multi_restart(model, x, y, EPS, ALPHA, k, PGD_RESTARTS, SEED),
                       )
        )
    specs.append(Evaluation("cw_l2", CW_STEPS, math.nan, lambda x, y: cw(model, x, y, c=CW_C, steps=CW_STEPS)))
    return specs


def _write_csv(rows: list[list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "defense", "attack", "eps", "steps", "accuracy", "success_rate", "mean_linf", "mean_l2"]
        )
        writer.writerows(rows)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(device)
    loader = build_loader()
    clean_preds = _clean_predictions(model, loader, device)

    rows: list[list[str]] = []
    for spec in _evaluations(model):
        res = _evaluate(model, loader, device, spec, clean_preds)
        eps_field = "" if math.isnan(spec.eps) else f"{spec.eps:.6f}"
        rows.append(
            [
                MODEL_NAME,
                DEFENSE,
                spec.attack,
                eps_field,
                str(spec.steps),
                f"{res.accuracy:.4f}",
                f"{res.success_rate:.4f}",
                f"{res.mean_linf:.6f}",
                f"{res.mean_l2:.6f}",
            ]
        )
        tag = f"pgd-{spec.steps}" if spec.attack == "pgd" else spec.attack
        print(
            f"{tag:>8}: accuracy = {res.accuracy:.4f}, "
            f"success = {res.success_rate:.4f}, "
            f"mean L-inf = {res.mean_linf:.6f}, mean L2 = {res.mean_l2:.6f}"
        )

    _write_csv(rows, CSV_PATH)
    print(f"Results written to {CSV_PATH}")


if __name__ == "__main__":
    main()