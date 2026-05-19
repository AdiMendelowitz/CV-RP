"""
Post training quantization (dynamic & static) for a pre-trained ResNet-18.

Measures model size (MB), median single-sample inference latency (ms) and top-1 acuracy on CIFAR-10 before and after
each quantization scheme.

Both schemes run on CPU. PyTorch's quantized kernels are CPU-only for static quantization, so models and inputs are
explicitly kept on CPU regardless of what device the checkpoint was trained on.


Note on dynamic quantization and Conv2d:
    PyTorch does not support dynamic quantization for nn.Conv2d,  passing it to quantize_dynamic is silently ignored.
    Only nn.Linear layers are quantized in the dynamic scheme.
    Static quantization covers all Conv2d layers via the observer/calibration pipeline.
    Reference: https://pytorch.org/docs/stable/quantization.html
"""

import argparse
import copy
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_ROOT = _REPO_ROOT / "data"
_STATIC_QUANT_CKPT = Path(__file__).resolve().parent / "checkpoints" / "quantization" / "resnet18_static_int8.pth"

sys.path.insert(0, str(_REPO_ROOT / "computer-vision-foundations" / "code" / "pytorch_cnn"))
from resnet import resnet18 as custom_resnet18  # noqa: E402

# ----------------------------------------------------------------------------------------------------------------------
# Data Loaders
# ----------------------------------------------------------------------------------------------------------------------


def get_cifar10_loaders(data_dir: str = str(_DATA_ROOT), batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """
    Calibration loader uses the training split with no shuffling so that consecutive batches are representative of the
    full distribution. The test loader is used for accuracy evaluation.

    Returns:
        (calibration_loader, test_loader) for CIFAR-10.
    """

    # in get_cifar10_loaders
    normalise = T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
    transforms = T.Compose([T.ToTensor(), normalise])

    calibration_set = torchvision.datasets.CIFAR10(_DATA_ROOT, train=True, download=True, transform=transforms)
    test_set = torchvision.datasets.CIFAR10(_DATA_ROOT, train=False, download=True, transform=transforms)

    calibration_loader = DataLoader(
        calibration_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    return calibration_loader, test_loader


# ----------------------------------------------------------------------------------------------------------------------
# Measurement Utilities
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    label: str
    size_mb: float
    latency_ms: float
    top1_acc: float


def model_size_md(mode: nn.Module) -> float:
    """Serialize the model state dict to a temp file and return its size in MB."""

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_path = f.name
    try:
        torch.save(mode.state_dict(), tmp_path)
        return os.path.getsize(tmp_path) / (1024**2)
    finally:
        os.remove(tmp_path)


def median_latency_ms(model: nn.Module, input_tensor: torch.Tensor, n_warmup: int = 20, n_runs: int = 100) -> float:
    """
    Warm-up rins are discarded so that JIT compilation and cache effects don't inflate the measured latency.

    Returns:
        Median single-sample inference latency in milliseconds.
    """

    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_tensor)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(input_tensor)
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def evaluate_top1(model: nn.Module, loader: DataLoader) -> float:
    """Return top-1 accuracy on the given loader. All data stays on CPU"""

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
    return 100 * correct / total


def run_benchmark(
    model: nn.Module, test_loader: DataLoader, label: str, latency_input: torch.Tensor
) -> BenchmarkResult:
    return BenchmarkResult(
        label=label,
        size_mb=model_size_md(model),
        latency_ms=median_latency_ms(model, latency_input),
        top1_acc=evaluate_top1(model, test_loader),
    )


# ----------------------------------------------------------------------------------------------------------------------
# Quantization
# ----------------------------------------------------------------------------------------------------------------------


class QuantizableWrapper(nn.Module):
    """
    Wraps a model with QuantStub / DeQuantStub for static PTQ.

    PyTorch's static quantization pipeline requires explicit quant/dequant boundry markers so that the observer
    insertion and conversion passes know where floating point tensors enter and exit the quantized domain.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.dequant(self.model(self.quant(input)))


def fuse_resnet18_modules(model: nn.Module) -> None:
    """
    Fuse Conv-BN-ReLU sequences in a ResNet-18 in-place

    Fusion replaces adjacent Conv+BN_+ReLU chains with a single fused operator, eliminating intermediate requantization
    steps and reducing quantization error. Assume the standard ResNet-18 module naming convention.
    """
    torch.quantization.fuse_modules(model, ["conv1", "bn1", "relu"], inplace=True)
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        for block in getattr(model, layer_name):
            torch.quantization.fuse_modules(block, ["conv1", "bn1"], inplace=True)
            torch.quantization.fuse_modules(block, ["conv2", "bn2"], inplace=True)
            if getattr(block, "downsample", None) is not None:
                torch.quantization.fuse_modules(block.downsample, ["0", "1"], inplace=True)


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """
    Weights are pre-quantized to INT8, activations are quantized on the fly at inference time. Only nn.Linear
    layers are quantized because PyTorch doesn't support dynamic quantization for nn.Conv2d.

    Returns:
        Dynamic quantized copy of the model.
    """
    quantized = copy.deepcopy(model).cpu()
    return torch.quantization.quantize_dynamic(quantized, qconfig_spec={nn.Linear}, dtype=torch.qint8)


def apply_static_quantization(
    model: nn.Module, calibration_loader: DataLoader, n_calibration_batches: int = 100
) -> nn.Module:
    """
    Fuses Conc-BN-ReLU modules, runs calibration data through observer-instrumented forward passes to collect activation
    range statistics, then converts to a fully INT8 model.
    Uses per-channel weight quantization and per-tensor activation quantization (PyTorch x86 default, consistent with
    Jacob et al. 2018).

    Returns:
        A statically quantized copy of the model.
    """
    base = copy.deepcopy(model).cpu().eval()
    fuse_resnet18_modules(base)

    wrapped = QuantizableWrapper(base)
    wrapped.qconfig = torch.quantization.get_default_qconfig("x86")
    torch.quantization.prepare(wrapped, inplace=True)

    wrapped.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(calibration_loader):
            if i >= n_calibration_batches:
                break
            wrapped(x.cpu())

    torch.quantization.convert(wrapped, inplace=True)
    return wrapped


# ----------------------------------------------------------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------------------------------------------------------

N_CALIBRATION_BATCHES = 100


def parse_args() -> argparse.Namespace:

    _default_checkpoint = (
        _REPO_ROOT / "computer-vision-foundations" / "code" / "pytorch_cnn" / "best_resnet18_cifar10 (1).pth"
    )
    parser = argparse.ArgumentParser(description="PTQ benchmarks for ResNet-18 on CIFAR-10")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(_default_checkpoint),
        help="Path to ResNet-18 state dict checkpoint (.pth)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for evaluation and calibration (default: 128)"
    )
    return parser.parse_args()


def print_results(results: list[BenchmarkResult]) -> None:
    col_w = 32
    header = f"{'Model':<{col_w}} {'Size (MB)':>10} {'Latency (ms)':>14} {'Top-1 (%)':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(f"{r.label:<{col_w}} {r.size_mb:>10.2f} {r.latency_ms:>14.2f} {r.top1_acc:>10.2f}")
    print()


def main() -> None:
    args = parse_args()

    print(f"Data directory: {_DATA_ROOT}")
    print("Loading CIFAR-10...")
    calibration_loader, test_loader = get_cifar10_loaders(args.batch_size)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = custom_resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    latency_input = torch.randn(1, 3, 32, 32)
    results: list[BenchmarkResult] = []

    print("Benchmarking FP32 baseline...")
    results.append(run_benchmark(model, test_loader, "FP32 (baseline)", latency_input))

    print("Applying dynamic quantization (Linear only)...")
    dynamic_model = apply_dynamic_quantization(model)
    results.append(run_benchmark(dynamic_model, test_loader, "INT8 dynamic (Linear only)", latency_input))

    print(f"Applying static quantization (calibrating on {N_CALIBRATION_BATCHES} batches)...")
    static_model = apply_static_quantization(model, calibration_loader, N_CALIBRATION_BATCHES)
    _STATIC_QUANT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(static_model, _STATIC_QUANT_CKPT)
    print(f"Static INT8 model saved to {_STATIC_QUANT_CKPT}")
    results.append(run_benchmark(static_model, test_loader, "INT8 static (PTQ)", latency_input))

    print_results(results)


if __name__ == "__main__":
    main()
