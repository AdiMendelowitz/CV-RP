"""PGD adversarial training for CIFAR-10 ResNet-18 (Madry et al., 2018).

Run from the toolkit root:

    python -m defenses.adversarial_training --epochs 30

Adversarial examples are generated in [0, 1] pixel space by attacks.pgd against the normalisation-wrapping model, so the
training-time threat model matches the one used by experiments/robustness_eval.py.

Checkpoint selection uses robust accuracy on a held-out split of the training set. Robust test accuracy peaks shortly
after the first learning-rate decay and then degrades (Rice, Wong and Kolter, 2020, "Overfitting in Adversarially Robust
Deep Learning"), so the final epoch is a poor choice. The test set is never touched here.

The best checkpoint is written as a bare state_dict of the unwrapped network, matching the format of the existing clean
checkpoint, so ``CIFAR10_RESNET18_CKPT`` can point at it for evaluation.
"""

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacks.pgd import pgd
from experiments._common import CIFAR10_MEAN, CIFAR10_STD, resolve_data_root
from models.normalized_model import NormalizedModel
from models.resnet import resnet18

DEFENSES_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = DEFENSES_DIR / "results"
DEFAULT_CKPT_DIR = DEFENSES_DIR / "checkpoints"
HISTORY_FILENAME = "adv_training_history.csv"
CONFIG_FILENAME = "adv_training_config.json"
CHECKPOINT_FILENAME = "resnet18_cifar10_pgd_at.pth"
NUM_CLASSES = 10

# Held out independently of the training seed so that runs at different seeds are selected and compared on identical
# validation images.
SPLIT_SEED = 12345


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Hyperparameters for a PGD adversarial-training run."""

    epochs: int = 30
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_milestones: tuple[float, ...] = (0.5, 0.75)  # fractions of epochs
    lr_gamma: float = 0.1
    eps: float = 8 / 255
    alpha: float = 2 / 255
    attack_steps: int = 7
    val_attack_steps: int = 10
    val_size: int = 5000
    val_eval_samples: int = 1024
    val_batch_size: int = 256
    seed: int = 0
    num_workers: int = 0
    device: str | None = None
    out_dir: Path = DEFAULT_OUT_DIR
    ckpt_dir: Path = DEFAULT_CKPT_DIR

    def __post_init__(self) -> None:
        positive_ints = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "attack_steps": self.attack_steps,
            "val_attack_steps": self.val_attack_steps,
            "val_size": self.val_size,
            "val_eval_samples": self.val_eval_samples,
            "val_batch_size": self.val_batch_size,
        }
        for name, value in positive_ints.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if not 0.0 < self.lr_gamma < 1.0:
            raise ValueError(f"lr_gamma must be in (0, 1), got {self.lr_gamma}")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if not 0.0 < self.alpha <= self.eps:
            # A step wider than the budget saturates the projection on iteration one, which
            # degrades PGD to a randomly initialised FGSM without any visible signal.
            raise ValueError(f"alpha must be in (0, eps], got alpha={self.alpha}, eps={self.eps}")
        if self.val_eval_samples > self.val_size:
            raise ValueError(
                f"val_eval_samples must be <= val_size, got {self.val_eval_samples} > {self.val_size}"
            )
        if not self.lr_milestones:
            raise ValueError("lr_milestones must contain at least one fraction")
        if any(not 0.0 < f < 1.0 for f in self.lr_milestones):
            raise ValueError(f"lr_milestones must all lie in (0, 1), got {self.lr_milestones}")
        if list(self.lr_milestones) != sorted(set(self.lr_milestones)):
            raise ValueError(f"lr_milestones must be strictly increasing, got {self.lr_milestones}")
        epochs = self.milestone_epochs()
        if len(set(epochs)) != len(epochs):
            raise ValueError(
                f"lr_milestones {self.lr_milestones} collapse onto the same epoch at epochs={self.epochs}"
            )

    def milestone_epochs(self) -> list[int]:
        """Convert milestone fractions into the epoch indices consumed by MultiStepLR."""
        return [max(1, round(fraction * self.epochs)) for fraction in self.lr_milestones]


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """One row of the training history."""

    epoch: int
    lr: float
    train_loss: float
    train_adv_acc: float
    val_clean_acc: float
    val_robust_acc: float
    seconds: float


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Outcome of a completed run, for callers that drive ``train`` directly from a notebook."""

    history: tuple[EpochMetrics, ...]
    best_epoch: int
    best_val_robust_acc: float
    checkpoint_path: Path
    history_path: Path


def set_seed(seed: int) -> None:
    """Seed the torch RNGs and pin cuDNN to deterministic kernels.

    Weight initialisation, augmentation and the PGD random start all draw from the torch global RNG, so seeding it is
    sufficient. CPU runs reproduce bitwise. On CUDA the backward pass of nn.AdaptiveAvgPool2d accumulates with atomic
    adds and has no deterministic kernel, so torch.use_deterministic_algorithms(True) would raise rather than help and
    GPU runs reproduce only up to floating-point accumulation order.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str | None) -> torch.device:
    """Return the requested device, or CUDA when available."""
    if name is not None:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loaders(config: TrainConfig, device: torch.device) -> tuple[DataLoader, DataLoader]:
    """Split the CIFAR-10 training set into an augmented train loader and a clean validation loader.

    Both loaders emit tensors in [0, 1]; normalisation is applied inside the model wrapper so that attacks operate in
    pixel space. The split is drawn from SPLIT_SEED, not training seed, keeping the held-out images fixed across seeds.
    """
    root = resolve_data_root()
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.ToTensor()

    train_source = CIFAR10(root=str(root), train=True, download=False, transform=train_transform)
    val_source = CIFAR10(root=str(root), train=True, download=False, transform=eval_transform)
    if config.val_size >= len(train_source):
        raise ValueError(f"val_size must be < {len(train_source)}, got {config.val_size}")

    split_generator = torch.Generator().manual_seed(SPLIT_SEED)
    permutation = torch.randperm(len(train_source), generator=split_generator).tolist()
    train_indices = permutation[: -config.val_size]
    val_indices = permutation[-config.val_size :]

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        Subset(train_source, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = DataLoader(
        Subset(val_source, val_indices),
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def build_model(device: torch.device) -> tuple[nn.Module, nn.Module]:
    """Return the normalisation-wrapped model and the bare network whose weights are saved.

    resnet18 defaults to 1000 classes, so num_classes is passed explicitly. The wrapper holds mean and std as buffers,
    which keeps them out of both net.state_dict() and model.parameters(): the saved checkpoint loads strictly into a
    bare resnet18, and weight decay never reaches the normalisation constants.
    """
    net = resnet18(num_classes=NUM_CLASSES)
    model = NormalizedModel(net, CIFAR10_MEAN, CIFAR10_STD).to(device)
    return model, net


def write_run_config(config: TrainConfig, path: Path, device: torch.device) -> None:
    """Record the run's hyperparameters beside its history so every result stays traceable."""
    payload: dict[str, object] = {
        key: (str(value) if isinstance(value, Path) else value) for key, value in asdict(config).items()
    }
    payload["split_seed"] = SPLIT_SEED
    payload["resolved_device"] = str(device)
    payload["torch_version"] = torch.__version__
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(net: nn.Module, path: Path) -> None:
    """Write the network state dict atomically so an interrupted save cannot destroy the previous best."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save({key: value.cpu() for key, value in net.state_dict().items()}, tmp_path)
    tmp_path.replace(path)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, *, eps: float, alpha: float, steps: int,
             device: torch.device, max_samples: int) -> tuple[float, float]:
    """Returns (clean accuracy, robust accuracy) over exactly the first max_samples validation images.

    The final batch is truncated so the metric does not shift with val_batch_size. The PGD attack omits its random
    start, leaving epoch-to-epoch comparison free of attack noise. This robust accuracy is a checkpoint-selection signal
    at steps iterations; the reported figure comes from experiments/robustness_eval.py. pgd re-enables gradients
    internally, so calling it under no_grad is safe.
    """
    was_training = model.training
    model.eval()
    clean_correct, robust_correct, seen = 0, 0, 0
    try:
        for images, labels in loader:
            if seen >= max_samples:
                break
            remaining = max_samples - seen
            if remaining < labels.size(0):
                images, labels = images[:remaining], labels[:remaining]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            clean_correct += int((model(images).argmax(dim=1) == labels).sum().item())
            adversarial = pgd(model, images, labels, eps=eps, alpha=alpha, steps=steps, random_start=False)
            robust_correct += int((model(adversarial).argmax(dim=1) == labels).sum().item())
            seen += labels.size(0)
    finally:
        model.train(was_training)
    if seen == 0:
        raise ValueError("validation loader yielded no samples")
    return clean_correct / seen, robust_correct / seen


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, config: TrainConfig,
                    device: torch.device) -> tuple[float, float]:
    """Run one adversarial-training epoch and return (mean loss, adversarial accuracy)."""
    loss_sum, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Generate in eval mode so the attack's forward passes leave BatchNorm running statistics untouched and the
        # training-time threat model matches the evaluation harness. The consequence is that those statistics are
        # estimated from adversarial batches alone, which depresses clean accuracy relative to train-mode generation
        # (Xie and Yuille, ICLR 2020).
        model.eval()
        adversarial = pgd(model, images, labels, eps=config.eps, alpha=config.alpha,
                          steps=config.attack_steps, random_start=True)
        model.train()

        logits = model(adversarial)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch = labels.size(0)
        loss_sum += loss.item() * batch
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        seen += batch
    if seen == 0:
        raise ValueError("training loader yielded no samples")
    return loss_sum / seen, correct / seen


def train(config: TrainConfig) -> TrainResult:
    """Adversarially train a ResNet-18 and keep the checkpoint with the best held-out robust accuracy.

    History is flushed and the best checkpoint is replaced after every epoch, so interrupting the run leaves both
    artifacts consistent with each other.
    """
    set_seed(config.seed)
    device = resolve_device(config.device)
    train_loader, val_loader = build_loaders(config, device)
    model, net = build_model(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.milestone_epochs(), gamma=config.lr_gamma
    )

    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = config.out_dir / HISTORY_FILENAME
    checkpoint_path = config.ckpt_dir / CHECKPOINT_FILENAME
    write_run_config(config, config.out_dir / CONFIG_FILENAME, device)

    history: list[EpochMetrics] = []
    best_robust_acc = -1.0
    best_epoch = -1

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(EpochMetrics)])
        writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            started = time.perf_counter()
            current_lr = optimizer.param_groups[0]["lr"]
            train_loss, train_adv_acc = train_one_epoch(model, train_loader, optimizer, config, device)
            val_clean_acc, val_robust_acc = evaluate(
                model,
                val_loader,
                eps=config.eps,
                alpha=config.alpha,
                steps=config.val_attack_steps,
                device=device,
                max_samples=config.val_eval_samples,
            )
            scheduler.step()

            metrics = EpochMetrics(
                epoch=epoch,
                lr=current_lr,
                train_loss=train_loss,
                train_adv_acc=train_adv_acc,
                val_clean_acc=val_clean_acc,
                val_robust_acc=val_robust_acc,
                seconds=time.perf_counter() - started,
            )
            history.append(metrics)
            writer.writerow(asdict(metrics))
            handle.flush()

            if val_robust_acc > best_robust_acc:
                best_robust_acc = val_robust_acc
                best_epoch = epoch
                save_checkpoint(net, checkpoint_path)

            marker = " *" if best_epoch == epoch else ""
            print(
                f"epoch {epoch:3d}/{config.epochs}  lr {current_lr:.4f}  "
                f"loss {train_loss:.4f}  train_adv {train_adv_acc:.4f}  "
                f"val_clean {val_clean_acc:.4f}  val_robust {val_robust_acc:.4f}  "
                f"{metrics.seconds:.1f}s{marker}",
                flush=True,
            )

    print(f"best epoch {best_epoch} with val_robust {best_robust_acc:.4f} saved to {checkpoint_path}")
    return TrainResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_val_robust_acc=best_robust_acc,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    """Parse command-line overrides on top of the TrainConfig defaults."""
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="PGD adversarial training for CIFAR-10 ResNet-18.")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--momentum", type=float, default=defaults.momentum)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument(
        "--lr-milestones",
        type=float,
        nargs="+",
        default=list(defaults.lr_milestones),
        help="decay points as increasing fractions of --epochs, e.g. 0.5 0.75",
    )
    parser.add_argument("--lr-gamma", type=float, default=defaults.lr_gamma)
    parser.add_argument("--eps", type=float, default=defaults.eps,
                        help="L-infinity budget in [0, 1] pixel space")
    parser.add_argument("--alpha", type=float, default=defaults.alpha,
                        help="PGD step size in [0, 1], must be <= eps")
    parser.add_argument("--attack-steps", type=int, default=defaults.attack_steps)
    parser.add_argument("--val-attack-steps", type=int, default=defaults.val_attack_steps)
    parser.add_argument("--val-size", type=int, default=defaults.val_size)
    parser.add_argument("--val-eval-samples", type=int, default=defaults.val_eval_samples)
    parser.add_argument("--val-batch-size", type=int, default=defaults.val_batch_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--device", type=str, default=defaults.device, help="e.g. cuda, cpu")
    parser.add_argument("--out-dir", type=Path, default=defaults.out_dir)
    parser.add_argument("--ckpt-dir", type=Path, default=defaults.ckpt_dir)
    overrides = vars(parser.parse_args(argv))
    overrides["lr_milestones"] = tuple(overrides["lr_milestones"])
    return TrainConfig(**overrides)


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()