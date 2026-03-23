"""
Linear probe benchmark: frozen timm backbones evaluated on CIFAR-10.

Protocol:
    - Pretrained ImageNet weights loaded via timm (num_classes=0 strips the head)
    - CIFAR-10 resized to 224x224 with ImageNet normalization
    - Features L2-normalized before logistic regression (C=0.316)
    - Inference time: median over 200 forward passes, batch=1, CPU
"""

import time
from pathlib import Path
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Config --------------------------------------------------------------------------

MODELS: dict[str, str] = {
    "ResNet-18": "resnet18",
    "ViT-Tiny": "vit_tiny_patch16_224",
    "EfficientNet-B0": "efficientnet_b0",
    "ConvNeXt-Tiny": "convnext_tiny",
}

IMG_SIZE = 224
EXTRACT_BATCH = 256
TIMING_RUNS = 200
LR_C = 0.316
DATA_DIR = str(Path(__file__).resolve().parents[2] / "data")
OUTPUT_MD = "experiments/architecture_benchmark.md"
DEVICE = torch.device("cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Data -------------------------------------------------------------------------

def get_cifar10_loaders() -> tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = datasets.CIFAR10(DATA_DIR, train=True, transform=tfm, download=True)
    test_ds = datasets.CIFAR10(DATA_DIR, train=False, transform=tfm, download=True)
    loader_kwargs = dict(batch_size=EXTRACT_BATCH, num_workers=2, pin_memory=False)
    return DataLoader(train_ds, **loader_kwargs), DataLoader(test_ds, **loader_kwargs)

# --- Feature Extraction ----------------------------------------------------------

@torch.no_grad()
def extract_features(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    all_features, all_labels = [], []
    for images, targets in loader:
        all_features.append(model(images).numpy())
        all_labels.append(targets.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)


# --- Inference timing -----------------------------------------------------------

def measure_inference_ms(model: torch.nn.Module) -> float:
    """Median forward-pass time in ms, batch=1, CPU."""
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        for _ in range(10):     # warmup
            model(dummy)
        times = []
        for _ in range(TIMING_RUNS):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0)*1000)
    return float(np.median(times))

# --- Benchmark ----------------------------------------------------------------

def count_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def benchmark_model(display_name: str, timm_name: str, train_loader: DataLoader,
                    test_loader: DataLoader) -> dict:

    print(f"\n── {display_name} ──────────────────────────")

    # num_classes=0: strips classification head, returns pooled feature vectore
    model = timm.create_model(timm_name, pretrained=True, num_classes=0)
    model.eval()

    params_m = count_params_m(model)
    print(f"  Params:     {params_m:.1f}M")

    inf_ms = measure_inference_ms(model)
    print(f"  Inference time: {inf_ms:.1f} ms/image (CPU, batch=1)")

    print("Extracting train features")
    X_train, y_train = extract_features(model, train_loader)
    print("Extracting test features")
    X_test, y_test = extract_features(model, test_loader)

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    print("Fitting logistic regression")
    clf = LogisticRegression(C=LR_C, max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test) * 100
    print(f"  Top-1 acc: {acc:.2f}%")

    return {"model": display_name, "acc": acc, "params_m": params_m, "inf_ms": inf_ms}


# --- Markdown ----------------------------------------------------------------

def write_markdown(results: list[dict]) -> None:
    Path(OUTPUT_MD).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Architecture Benchmark — Linear Probe on CIFAR-10",
        "",
        "**Protocol:** Frozen pretrained backbone (ImageNet weights via timm) + "
        f"logistic regression (C={LR_C}, L2-normalised features). "
        f"Input resized to {IMG_SIZE}×{IMG_SIZE}. "
        "Inference time: median over 200 runs, batch=1, CPU.",
        "",
        "| Model | Top-1 Acc (%) | Params (M) | Inference (ms/img) |",
        "|-------|:-------------:|:----------:|:------------------:|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['acc']:.2f} | {r['params_m']:.1f} | {r['inf_ms']:.1f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Accuracy reflects **representation quality** of the pretrained backbone, "
        "not fine-tuned performance on CIFAR-10.",
        "- ViT-Tiny is expected to underperform CNNs here: "
        "attention lacks the locality inductive bias that benefits small datasets.",
        "- Inference time covers the backbone forward pass only (no logistic regression head).",
        "",
    ]
    Path(OUTPUT_MD).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to {OUTPUT_MD}")


if __name__ == "__main__":
    print("Loading CIFAR-10")
    train_loader, test_loader = get_cifar10_loaders()

    results = [
        benchmark_model(name, timm_name, train_loader, test_loader)
        for name, timm_name in MODELS.items()
    ]

    print("\n\n── Summary ──────────────────────────────────────────")
    print(f"{'Model':<20} {'Acc':>9} {'Params':>10} {'ms/img':>10}")
    print("─" * 53)
    for r in results:
        print(
            f"{r['model']:<20} {r['acc']:>8.2f}% "
            f"{r['params_m']:>9.1f}M {r['inf_ms']:>9.1f}ms"
        )

    write_markdown(results)