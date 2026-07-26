"""Tests for the FGSM attack.

Fast tests run against a small BatchNorm-free model on synthetic inputs in
[0, 1], so they need no checkpoint and no dataset. The clean-accuracy and
effectiveness tests need the trained ResNet-18 CIFAR-10 checkpoint and a real
batch; they skip unless both are available.

Checkpoint resolution: the ``CIFAR10_RESNET18_CKPT`` environment variable
overrides the default path, which points at the Week 1 checkpoint relative to
the toolkit root. Data root: ``CIFAR10_DATA_ROOT`` (default ./data).

The real-model tests assume the model normalises inputs internally, matching the
toolkit's pixel-space threat model. A model that expects externally normalised
inputs will read back low clean accuracy and fail the sanity test.
"""

import os
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from attacks.fgsm import fgsm

CKPT_ENV = "CIFAR10_RESNET18_CKPT"
DATA_ENV = "CIFAR10_DATA_ROOT"
CKPT_ENV = "CIFAR10_RESNET18_CKPT"
DATA_ENV = "CIFAR10_DATA_ROOT"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
DEFAULT_CKPT = (
    Path(__file__).resolve().parents[2]
    / "computer-vision-foundations"
    / "code"
    / "pytorch_cnn"
    / "best_resnet18_cifar10 (1).pth"
)


class _TinyNet(nn.Module):
    """Minimal conv classifier with input-to-logit gradient flow and no BatchNorm.

    Adaptive pooling accepts any spatial size, and the absence of BatchNorm means
    eval and train modes behave identically, keeping the unit tests free of the
    eval-mode precondition.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.head(x)


@pytest.fixture
def tiny_model() -> nn.Module:
    torch.manual_seed(0)
    model = _TinyNet()
    model.eval()
    return model


@pytest.fixture
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    images = torch.rand(8, 3, 32, 32)  # torch.rand draws in [0, 1)
    labels = torch.randint(0, 10, (8,))
    return images, labels


@pytest.mark.parametrize("targeted", [False, True])
@pytest.mark.parametrize("eps", [0.0, 8 / 255, 16 / 255])
def test_fgsm_preserves_shape_and_dtype(
    tiny_model: nn.Module,
    sample_batch: tuple[torch.Tensor, torch.Tensor],
    eps: float,
    targeted: bool,
) -> None:
    images, labels = sample_batch
    adv = fgsm(tiny_model, images, labels, eps=eps, targeted=targeted)
    assert adv.shape == images.shape
    assert adv.dtype == images.dtype


@pytest.mark.parametrize("targeted", [False, True])
@pytest.mark.parametrize("eps", [0.0, 8 / 255, 16 / 255])
def test_fgsm_respects_linf_budget_and_pixel_range(
    tiny_model: nn.Module,
    sample_batch: tuple[torch.Tensor, torch.Tensor],
    eps: float,
    targeted: bool,
) -> None:
    images, labels = sample_batch
    adv = fgsm(tiny_model, images, labels, eps=eps, targeted=targeted)
    assert (adv - images).abs().max().item() <= eps + 1e-6
    assert adv.min().item() >= 0.0
    assert adv.max().item() <= 1.0


def test_fgsm_perturbs_the_input(
    tiny_model: nn.Module,
    sample_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    # Guards against a broken attack that returns the input unchanged: an identity
    # function would pass every other fast test.
    images, labels = sample_batch
    adv = fgsm(tiny_model, images, labels, eps=8 / 255)
    assert not torch.equal(adv, images)


def test_fgsm_does_not_mutate_input(
    tiny_model: nn.Module,
    sample_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    images, labels = sample_batch
    original = images.clone()
    fgsm(tiny_model, images, labels, eps=8 / 255)
    assert torch.equal(images, original)


def test_fgsm_leaves_parameter_grads_none(
    tiny_model: nn.Module,
    sample_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    images, labels = sample_batch
    assert all(p.grad is None for p in tiny_model.parameters())
    fgsm(tiny_model, images, labels, eps=8 / 255)
    assert all(p.grad is None for p in tiny_model.parameters())


def test_fgsm_runs_inside_no_grad(
    tiny_model: nn.Module,
    sample_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    # Guards the enable_grad wrapper: the attack must work inside an eval sweep.
    images, labels = sample_batch
    with torch.no_grad():
        adv = fgsm(tiny_model, images, labels, eps=8 / 255)
    assert adv.shape == images.shape
    assert not adv.requires_grad


def _load_trained_resnet() -> nn.Module:
    ckpt = Path(os.environ.get(CKPT_ENV, DEFAULT_CKPT))
    if not ckpt.is_file():
        pytest.skip(f"checkpoint not found at {ckpt}; set {CKPT_ENV} to override")
    try:
        from models.resnet import resnet18
    except ImportError:
        pytest.skip("models.resnet.resnet18 not importable from the test root")

    model = resnet18(num_classes=10)
    obj = torch.load(ckpt, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    elif isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
    else:
        state = obj
    model.load_state_dict(state)

    from models.normalized_model import NormalizedModel
    model = NormalizedModel(model, mean=CIFAR10_MEAN, std=CIFAR10_STD)
    model.eval()  # eval mode runs on CPU by design; keep the batch on CPU too
    return model


def _cifar_test_batch(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    pytest.importorskip("torchvision")
    from torchvision import datasets, transforms

    root = os.environ.get(DATA_ENV, "./data")
    try:
        dataset = datasets.CIFAR10(
            root=root, train=False, download=False, transform=transforms.ToTensor()
        )
    except RuntimeError:
        pytest.skip(f"CIFAR-10 test set not found under {root!r}; set {DATA_ENV}")
    images, labels = next(iter(DataLoader(dataset, batch_size=n, shuffle=False)))
    return images, labels


@pytest.mark.slow
def test_trained_resnet_clean_accuracy() -> None:
    # Wiring sanity check, isolated from the attack: confirms the checkpoint loads
    # and the model receives inputs in the space it expects.
    model = _load_trained_resnet()
    images, labels = _cifar_test_batch(256)
    assert images.min().item() >= 0.0 and images.max().item() <= 1.0

    with torch.no_grad():
        clean_acc = (model(images).argmax(dim=1) == labels).float().mean().item()
    assert clean_acc > 0.90, f"clean accuracy {clean_acc:.3f}; check checkpoint or normalisation"


@pytest.mark.slow
def test_fgsm_effectiveness_on_trained_resnet() -> None:
    model = _load_trained_resnet()
    images, labels = _cifar_test_batch(256)

    adv = fgsm(model, images, labels, eps=16 / 255)
    with torch.no_grad():
        adv_acc = (model(adv).argmax(dim=1) == labels).float().mean().item()
    assert adv_acc < 0.50, f"adversarial accuracy {adv_acc:.3f}; check gradient flow"