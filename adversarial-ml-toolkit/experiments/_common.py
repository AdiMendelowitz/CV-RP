"""Shared loading helpers for the adversarial-ml-toolkit evaluation scripts.

Owns the canonical data and checkpoint paths so the individual experiment scripts
cannot drift from each other. Paths default to their repository locations and are
overridable with CIFAR10_DATA_ROOT and CIFAR10_RESNET18_CKPT. Downloading is
disabled: the machine's torchvision fetch hits an expired certificate, and the
local CIFAR-10 copy is authoritative.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models.normalized_model import NormalizedModel
from models.resnet import resnet18

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTORCH_CNN_DIR = REPO_ROOT / "computer-vision-foundations" / "code" / "pytorch_cnn"

CKPT_ENV = "CIFAR10_RESNET18_CKPT"
DATA_ENV = "CIFAR10_DATA_ROOT"
DEFAULT_CKPT = PYTORCH_CNN_DIR / "best_resnet18_cifar10 (1).pth"
DEFAULT_DATA_ROOT = PYTORCH_CNN_DIR / "data"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DEFAULT_NUM_SAMPLES = 1000
DEFAULT_BATCH_SIZE = 250


def resolve_data_root() -> Path:
    """Return the CIFAR-10 root, failing fast rather than downloading."""
    root = Path(os.environ.get(DATA_ENV, str(DEFAULT_DATA_ROOT)))
    if not (root / "cifar-10-batches-py").is_dir():
        raise RuntimeError(f"No cifar-10-batches-py directory under {root}. "
                           f"Set {DATA_ENV} to override. Downloading is disabled.")
    return root

def resolve_checkpoint() -> Path:
    """Return the checkpoint path, honouring the env override used by the tests."""
    path = Path(os.environ.get(CKPT_ENV, str(DEFAULT_CKPT)))
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {path}. Set {CKPT_ENV} or restore the file.")
    return path

def load_model(device: torch.device) -> NormalizedModel:
    """Load the trained ResNet-18, wrap it for input normalisation, set eval mode."""
    net = resnet18(num_classes=10)
    state = torch.load(resolve_checkpoint(), map_location=device, weights_only=True)
    net.load_state_dict(state)
    model = NormalizedModel(net, CIFAR10_MEAN, CIFAR10_STD).to(device)
    model.eval()
    return model


def build_loader(num_samples: int = DEFAULT_NUM_SAMPLES, batch_size: int = DEFAULT_BATCH_SIZE) -> DataLoader:
    """First num_samples test images in [0, 1] pixel space, in dataset order.

    Defaults reproduce the canonical evaluation subset exactly.

    Raises:
        ValueError: If num_samples or batch_size is < 1, or num_samples exceeds the test set size.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    dataset = CIFAR10(root=str(resolve_data_root()), train=False, download=False, transform=transforms.ToTensor())
    if num_samples > len(dataset):
        raise ValueError(f"num_samples must be <= {len(dataset)}, got {num_samples}")
    subset = Subset(dataset, range(num_samples))
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
