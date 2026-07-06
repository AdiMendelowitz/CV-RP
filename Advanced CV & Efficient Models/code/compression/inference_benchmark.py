# code/compression/inference_benchmark.py
"""
Inference time benchmark: ResNet-18 teacher vs SmallCNN student.

Measures CPU latency, throughput, and model size for both models.
Loads the best checkpoints saved by train_distillation.py.

"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from distillation import build_student
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "computer-vision-foundations" / "code" / "pytorch_cnn"))
from resnet import resnet18 as custom_resnet18

_project_root = Path(__file__).resolve().parents[3]
_DATA_ROOT = _project_root / "data"

TEACHER_CHECKPOINT = (
    _project_root / "computer-vision-foundations" / "code" / "pytorch_cnn" / "best_resnet18_cifar10.pth"
)

DISTILL_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "distillation" / "best_student_distill.pth"
BASELINE_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "distillation" / "best_student_baseline.pth"

# Teacher was trained with these (confirmed from train_cifar.py)
_TEACHER_MEAN = (0.4914, 0.4822, 0.4465)
_TEACHER_STD = (0.2470, 0.2435, 0.2616)


BATCH_SIZE = 128
NUM_WORKERS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_val_loader(mean: tuple, std: tuple) -> DataLoader:
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    ds = datasets.CIFAR10(root=str(_DATA_ROOT), train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Estimate in-memory parameter size in MB (float32)."""
    return count_parameters(model) * 4 / (1024**2)


def measure_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100


def measure_latency_single_ms(model: nn.Module, n_warmup: int = 20, n_runs: int = 100) -> float:
    """Median single-sample CPU latency in ms. Matches quantization.py methodology."""
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        times: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def print_row(label: str, params: int, size_mb: float, acc: float, latency_ms: float) -> None:
    print(f"  {label:<30} {params:>12,}  {size_mb:>8.2f} MB  {acc:>8.2f}%  {latency_ms:>10.2f} ms")


def _load_cifar_resnet18(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load the custom CIFAR-adapted ResNet-18 (3×3 stem, shortcut naming) from checkpoint."""
    model = custom_resnet18(num_classes=10)
    # Replace 7×7 stem with 3×3 to match the architecture the checkpoint was trained with.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cpu")  # CPU benchmark — matches deployment target
    loader = get_val_loader(_TEACHER_MEAN, _TEACHER_STD)

    print("\nLoading models")
    teacher = _load_cifar_resnet18(str(TEACHER_CHECKPOINT), device)

    student_distill = build_student("small_cnn", num_classes=10)
    ckpt_d = torch.load(DISTILL_CHECKPOINT, map_location=device, weights_only=True)
    student_distill.load_state_dict(ckpt_d["model_state_dict"])
    student_distill.eval()
    student_baseline = build_student("small_cnn", num_classes=10)
    ckpt_b = torch.load(BASELINE_CHECKPOINT, map_location=device, weights_only=True)
    student_baseline.load_state_dict(ckpt_b["model_state_dict"])
    student_baseline.eval()

    models_to_bench = [
        ("ResNet-18 (teacher)", teacher, loader),
        ("SmallCNN — distilled", student_distill, loader),
        ("SmallCNN — baseline CE", student_baseline, loader),
    ]

    print("Measuring accuracy...")
    accuracies = {label: measure_accuracy(model, loader, device) for label, model, _ in models_to_bench}

    print("Measuring single-sample latency (20 warmup + 100 runs, median)...")
    latencies = {label: measure_latency_single_ms(model) for label, model, _ in models_to_bench}

    # --- Print table ---
    print(f"\n{'=' * 80}")
    print("  Inference Benchmark — CPU  (latency: single sample, median of 100 runs)")
    print(f"{'=' * 80}")
    print(f"  {'Model':<30} {'Parameters':>12}  {'Size (MB)':>10}  {'Top-1 (%)':>10}  {'Latency (ms)':>13}")
    print(f"  {'-' * 30} {'-' * 12}  {'-' * 10}  {'-' * 10}  {'-' * 13}")

    for label, model, _ in models_to_bench:
        params = count_parameters(model)
        size = model_size_mb(model)
        acc = accuracies[label]
        lat = latencies[label]
        print_row(label, params, size, acc, lat)

    print(f"{'=' * 80}")

    # --- Compression summary ---
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student_distill)
    teacher_lat = latencies["ResNet-18 (teacher)"]
    distill_lat = latencies["SmallCNN — distilled"]

    print(f"\n  Compression ratio (params):  {teacher_params / student_params:.1f}x")
    print(f"  Speedup — distilled vs teacher:  {teacher_lat / distill_lat:.1f}x")
    print(
        f"  Accuracy gap — distilled vs teacher:  "
        f"{accuracies['ResNet-18 (teacher)'] - accuracies['SmallCNN — distilled']:.2f}pp\n"
    )
    print(
        f"  Distillation gain over baseline CE:  "
        f"{accuracies['SmallCNN — distilled'] - accuracies['SmallCNN — baseline CE']:.2f}pp\n"
    )


if __name__ == "__main__":
    main()
