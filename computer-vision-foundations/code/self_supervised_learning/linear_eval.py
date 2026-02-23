"""
Linear evaluation protocol for SimCLR pretrained encoder.

Freezes the encoder, trains a single linear layer on labeled CIFAR-10 and reports top-1
test accuracy.

Reference: Chen et al. (2020), Appendix B.9
Paper: https://arxiv.org/abs/2002.05709

Usage:
    python linear_eval.py --checkpoint ./checkpoints/simclr/simclr_epoch100.pt
    python linear_eval.py --checkpoint ./checkpoints/simclr/simclr_epoch100.pt --epochs 90 --lr 0.1
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from simclr import SimCLR, CIFAR10_MEAN, CIFAR10_STD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Encoder output dimensions for supported architectures
ENCODER_DIMS: dict[str, int] = {"resnet18": 512, "resnet50": 2048}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(train: bool) -> transforms.Compose:
    """
    Standard CIFAR-10 transforms for linear evaluation.

    Training: RandomResizedCrop + RandomHorizontalFlip — minimal augmentation only. Stronger augmentation
    would help the linear head but artificially inflate the score beyond what the representations alone achieve.

    Test: ToTensor + Normalize only — no cropping that could alter semantics.

    Args:
        train: If True return training transforms, else test transforms.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def load_frozen_encoder(checkpoint_path: str, encoder_name: str, device: torch.device,) -> nn.Module:
    """
    Loads the SimCLR encoder from a checkpoint and freezes all its parameters.

    The encoder is kept in eval() mode throughout, which ensures BatchNorm uses the running
    statistics accumulated during pretraining rather than recomputing them from the current (labeled) batch.

    Args:
        checkpoint_path: Path to SimCLR checkpoint (.pt file).
        encoder_name: Architecture name ('resnet18'/'resnet50').
        device: Target device.

    Returns:
        Frozen encoder in eval mode on the target device.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = SimCLR(base_encoder=encoder_name)
    model.load_state_dict(ckpt["model_state_dict"])

    encoder = model.encoder

    for param in encoder.parameters():
        param.requires_grad = False

    encoder.eval()
    encoder.to(device)

    logger.info(f"Loaded encoder from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    return encoder


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(encoder: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Runs the frozen encoder over an entire dataset once and caches the output.

    Pre-extracting features is significantly faster than re-running the encoder every epoch
    during linear classifier training, since the encoder never changes.

    Args:
        encoder: Frozen encoder in eval mode.
        loader: DataLoader yielding (image, label) batches.
        device: Device the encoder lives on.

    Returns:
        features: Shape (N, hidden_dim), stored on CPU.
        labels: Shape (N,), stored on CPU.
    """
    all_features, all_labels = [], []

    for images, labels in loader:
        features = encoder(images.to(device)).flatten(start_dim=1)
        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


# ---------------------------------------------------------------------------
# Linear classifier training
# ---------------------------------------------------------------------------

def train_linear(classifier: nn.Linear, train_features: torch.Tensor, train_labels: torch.Tensor,
                 args: argparse.Namespace, device: torch.device) -> None:
    """
    Trains the linear classifier on pre-extracted features.

    Uses SGD with Nesterov momentum and no weight decay, matching the paper's linear evaluation protocol
    (Appendix B.9). Weight decay on the linear layer would penalize the scale of the learned weights
    independently of the representations, distorting the evaluation.

    Args:
        classifier: Linear head to train in-place.
        train_features: Pre-extracted features, shape (N, hidden_dim).
        train_labels: Ground truth labels, shape (N,).
        args: Training hyperparameters from parse_args.
        device: Target device.
    """
    classifier.train()
    classifier.to(device)

    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0

        for features, labels in loader:
            logits = classifier(features.to(device))
            loss = loss_fn(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            logger.info(
                f"Linear Epoch [{epoch:3d}/{args.epochs}]  "
                f"loss={total_loss / len(loader):.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(classifier: nn.Linear, test_features: torch.Tensor, test_labels: torch.Tensor,
             device: torch.device) -> float:
    """
    Computes top-1 accuracy of the linear classifier on pre-extracted features.

    Args:
        classifier: Trained linear head.
        test_features: Pre-extracted features, shape (N, hidden_dim).
        test_labels: Ground truth labels, shape (N,).
        device: Target device.

    Returns:
        Top-1 accuracy as a float in [0, 1].
    """
    classifier.eval()

    logits = classifier(test_features.to(device))
    predictions = logits.argmax(dim=1).cpu()

    return (predictions == test_labels).float().mean().item()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def linear_eval(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- ENCODER ---
    encoder = load_frozen_encoder(args.checkpoint, args.encoder, device)

    # --- DATA ---
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                     transform=get_transforms(train=True))
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                    transform=get_transforms(train=False))

    # shuffle=False: order is irrelevant for feature extraction
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # --- FEATURE EXTRACTION ---
    logger.info("Extracting train features...")
    train_features, train_labels = extract_features(encoder, train_loader, device)
    logger.info(f"Train features: {train_features.shape}")

    logger.info("Extracting test features...")
    test_features, test_labels = extract_features(encoder, test_loader, device)
    logger.info(f"Test features: {test_features.shape}")

    # --- LINEAR CLASSIFIER ---
    hidden_dim = ENCODER_DIMS[args.encoder]
    classifier = nn.Linear(hidden_dim, args.num_classes)

    # --- TRAIN ---
    logger.info("Training linear classifier...")
    train_linear(classifier, train_features, train_labels, args, device)

    # --- EVALUATE ---
    accuracy = evaluate(classifier, test_features, test_labels, device)
    logger.info(f"Top-1 Test Accuracy: {accuracy * 100:.2f}%")

    # --- SAVE ---
    save_path = Path(args.checkpoint).parent / "linear_eval_results.pt"
    torch.save(
        {"accuracy": accuracy, "classifier_state_dict": classifier.state_dict(), "args": vars(args)},
        save_path
    )
    logger.info(f"Results saved to {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear evaluation of SimCLR encoder")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SimCLR checkpoint (.pt)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--encoder", type=str, default="resnet18", choices=list(ENCODER_DIMS.keys()))
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    linear_eval(parse_args())