"""
Receptive Field Analysis — EfficientNet-B0 vs ConvNeXt-Tiny
=============================================================
Two complementary approaches:

1. Theoretical RF: analytically computed from the layer config using the
   standard accumulation formula:
       RF_new = RF_prev + (kernel - 1) * cumulative_stride

2. Empirical RF: gradient-based measurement. A single output neuron at the
   spatial centre of a feature map is activated; backpropagation reveals
   which input pixels contributed — their bounding box is the effective RF.

Uses torchvision reference implementations so the script is self-contained.
Replace with your own models by swapping the hook names and layer configs.

Usage:
    pip install torch torchvision matplotlib
    python receptive_field_analysis.py
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models

INPUT_SIZE = 224  # standard ImageNet spatial resolution

# ---------------------------------------------------------------------------
# 1. Theoretical receptive field
# ---------------------------------------------------------------------------


@dataclass
class Layer:
    """Minimal description of one spatial layer."""

    name: str
    kernel: int
    stride: int


def theoretical_rf(layers: list[Layer]) -> list[tuple[str, int, int]]:
    """
    Compute theoretical receptive field after each layer.

    Returns:
        List of (layer_name, rf_size, cumulative_stride) tuples.
    """
    rf: int = 1
    cumulative_stride: int = 1
    results: list[tuple[str, int, int]] = []

    for layer in layers:
        if layer.kernel > 1:
            rf += (layer.kernel - 1) * cumulative_stride
        cumulative_stride *= layer.stride
        results.append((layer.name, rf, cumulative_stride))

    return results


# EfficientNet-B0 spatial layers (depthwise convs only — pointwise k=1 convs
# and SE global pooling do not contribute to the receptive field).
EFFICIENTNET_B0_LAYERS: list[Layer] = [
    # Stem
    Layer("stem_conv", kernel=3, stride=2),
    # Stage 1: expand_ratio=1, k=3, s=1 — no expansion conv
    Layer("s1_b1_dw", kernel=3, stride=1),
    # Stage 2: expand_ratio=6, k=3, s=2 (block 1) then s=1 (block 2)
    Layer("s2_b1_dw", kernel=3, stride=2),
    Layer("s2_b2_dw", kernel=3, stride=1),
    # Stage 3: expand_ratio=6, k=5, s=2 (block 1) then s=1 (block 2)
    Layer("s3_b1_dw", kernel=5, stride=2),
    Layer("s3_b2_dw", kernel=5, stride=1),
    # Stage 4: expand_ratio=6, k=3, s=2 (block 1) then s=1 ×2
    Layer("s4_b1_dw", kernel=3, stride=2),
    Layer("s4_b2_dw", kernel=3, stride=1),
    Layer("s4_b3_dw", kernel=3, stride=1),
    # Stage 5: expand_ratio=6, k=5, s=1 ×3
    Layer("s5_b1_dw", kernel=5, stride=1),
    Layer("s5_b2_dw", kernel=5, stride=1),
    Layer("s5_b3_dw", kernel=5, stride=1),
    # Stage 6: expand_ratio=6, k=5, s=2 (block 1) then s=1 ×3
    Layer("s6_b1_dw", kernel=5, stride=2),
    Layer("s6_b2_dw", kernel=5, stride=1),
    Layer("s6_b3_dw", kernel=5, stride=1),
    Layer("s6_b4_dw", kernel=5, stride=1),
    # Stage 7: expand_ratio=6, k=3, s=1
    Layer("s7_b1_dw", kernel=3, stride=1),
    # Head conv is 1x1 — no RF change; omitted
]

# ConvNeXt-Tiny spatial layers.
# Each block: DWConv 7x7, stride=1. Downsampler: Conv 2x2, stride=2.
CONVNEXT_TINY_LAYERS: list[Layer] = [
    # Patchify stem: Conv 4x4, stride=4
    Layer("stem_conv", kernel=4, stride=4),
    # Stage 1: 3 blocks, DWConv 7x7 s=1
    Layer("s1_b1_dw", kernel=7, stride=1),
    Layer("s1_b2_dw", kernel=7, stride=1),
    Layer("s1_b3_dw", kernel=7, stride=1),
    # Downsampler 1->2: Conv 2x2, stride=2
    Layer("down1", kernel=2, stride=2),
    # Stage 2: 3 blocks
    Layer("s2_b1_dw", kernel=7, stride=1),
    Layer("s2_b2_dw", kernel=7, stride=1),
    Layer("s2_b3_dw", kernel=7, stride=1),
    # Downsampler 2->3
    Layer("down2", kernel=2, stride=2),
    # Stage 3: 9 blocks
    Layer("s3_b1_dw", kernel=7, stride=1),
    Layer("s3_b2_dw", kernel=7, stride=1),
    Layer("s3_b3_dw", kernel=7, stride=1),
    Layer("s3_b4_dw", kernel=7, stride=1),
    Layer("s3_b5_dw", kernel=7, stride=1),
    Layer("s3_b6_dw", kernel=7, stride=1),
    Layer("s3_b7_dw", kernel=7, stride=1),
    Layer("s3_b8_dw", kernel=7, stride=1),
    Layer("s3_b9_dw", kernel=7, stride=1),
    # Downsampler 3->4
    Layer("down3", kernel=2, stride=2),
    # Stage 4: 3 blocks
    Layer("s4_b1_dw", kernel=7, stride=1),
    Layer("s4_b2_dw", kernel=7, stride=1),
    Layer("s4_b3_dw", kernel=7, stride=1),
]

# Layer names at which to print the theoretical RF summary (stage end-points).
EFFICIENTNET_SUMMARY_LAYERS: set[str] = {
    "stem_conv",
    "s2_b1_dw",
    "s3_b1_dw",
    "s4_b1_dw",
    "s5_b1_dw",
    "s6_b1_dw",
    "s7_b1_dw",
}
CONVNEXT_SUMMARY_LAYERS: set[str] = {
    "stem_conv",
    "s1_b3_dw",
    "s2_b3_dw",
    "s3_b9_dw",
    "s4_b3_dw",
}


def print_theoretical(
    name: str,
    results: list[tuple[str, int, int]],
    summary_layers: set[str],
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Theoretical Receptive Field -- {name}")
    print(f"{'=' * 60}")
    print(f"  {'Layer':<22} {'RF (px)':>8}  {'Output stride':>14}")
    print(f"  {'-' * 22} {'-' * 8}  {'-' * 14}")
    for layer_name, rf, stride in results:
        if layer_name in summary_layers:
            print(f"  {layer_name:<22} {rf:>8}  {stride:>14}")
    print()


# ---------------------------------------------------------------------------
# 2. Empirical receptive field (gradient method)
# ---------------------------------------------------------------------------

# torchvision EfficientNet-B0: features.0=stem, features.1-7=stages,
# features.8=head conv
EFFICIENTNET_HOOK_NAMES: list[str] = [
    "features.0",
    "features.1",
    "features.2",
    "features.3",
    "features.4",
    "features.5",
    "features.6",
    "features.7",
    "features.8",
]

# torchvision ConvNeXt-Tiny: features.0=stem, odd indices=stages,
# even indices 2/4/6=downsamplers
CONVNEXT_HOOK_NAMES: list[str] = [
    "features.0",
    "features.1",
    "features.2",
    "features.3",
    "features.4",
    "features.5",
    "features.6",
    "features.7",
]

EFFICIENTNET_STAGE_LABELS: dict[str, str] = {
    "features.0": "stem",
    "features.1": "stage 1 out",
    "features.2": "stage 2 out",
    "features.3": "stage 3 out",
    "features.4": "stage 4 out",
    "features.5": "stage 5 out",
    "features.6": "stage 6 out",
    "features.7": "stage 7 out",
    "features.8": "head conv",
}
CONVNEXT_STAGE_LABELS: dict[str, str] = {
    "features.0": "stem",
    "features.1": "stage 1 out",
    "features.2": "downsample 1->2",
    "features.3": "stage 2 out",
    "features.4": "downsample 2->3",
    "features.5": "stage 3 out",
    "features.6": "downsample 3->4",
    "features.7": "stage 4 out",
}


def empirical_rf(
    model: nn.Module,
    stage_hooks: list[str],
    input_size: tuple[int, int, int, int] = (1, 3, INPUT_SIZE, INPUT_SIZE),
) -> dict[str, int]:
    """
    Estimate the effective receptive field at each hooked module.

    One independent forward+backward pass is performed per hook point:
      1. Fresh input tensor created (avoids retained-graph gradient contamination).
      2. Centre spatial neuron of the hooked feature map is selected.
      3. Backpropagate from that scalar to the input.
      4. Measure the bounding box of non-zero gradient pixels.

    A small-variance Gaussian input (std=0.1) is used so activations stay in
    the linear regime of SiLU/sigmoid, ensuring stable gradients at all depths.

    Args:
        model:       Model to analyse (set to eval mode internally).
        stage_hooks: Named modules to hook, in forward-pass order.
        input_size:  (1, C, H, W) -- batch size must be 1.

    Returns:
        Dict mapping module name to effective RF diameter (pixels),
        in the same order as stage_hooks.
    """
    model.eval()
    named_modules = dict(model.named_modules())
    rf_estimates: dict[str, int] = {}

    for hook_name in stage_hooks:
        if hook_name not in named_modules:
            continue

        # Fresh input and fresh graph per hook: eliminates all cross-pass
        # gradient contamination that arises when using retain_graph=True
        # with a shared input tensor.
        torch.manual_seed(0)
        x = torch.randn(input_size).mul(0.1).requires_grad_(True)

        # Capture a single activation with a one-shot hook.
        captured: list[torch.Tensor] = []
        hook = named_modules[hook_name].register_forward_hook(
            lambda _m, _i, out: captured.append(out)
        )
        model(x)
        hook.remove()

        if not captured or captured[0].dim() != 4:
            continue

        feat = captured[0]
        _, _, h, w = feat.shape
        ch, cw = h // 2, w // 2

        feat[0, :, ch, cw].sum().backward()

        if x.grad is None:
            continue

        grad_map = x.grad[0].abs().sum(dim=0)
        # Relative threshold: any pixel with gradient >= 0.1% of the peak is
        # counted as part of the RF. This is robust to the scale variation
        # caused by vanishing gradients across many layers with random weights,
        # unlike an absolute threshold which silently zeros out deep stages.
        peak = grad_map.max()
        if peak == 0:
            rf_estimates[hook_name] = 0
            continue
        grad_mask = grad_map > peak * 1e-3
        rows = grad_mask.any(dim=1).nonzero(as_tuple=True)[0]
        cols = grad_mask.any(dim=0).nonzero(as_tuple=True)[0]

        if len(rows) == 0 or len(cols) == 0:
            rf_estimates[hook_name] = 0
        else:
            rf_h = int((rows[-1] - rows[0]).item()) + 1
            rf_w = int((cols[-1] - cols[0]).item()) + 1
            rf_estimates[hook_name] = max(rf_h, rf_w)

    return rf_estimates


def print_empirical(
    name: str,
    rf_dict: dict[str, int],
    labels: dict[str, str],
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Empirical Receptive Field -- {name}")
    print(f"  (gradient method, centre neuron, {INPUT_SIZE}x{INPUT_SIZE} input)")
    print(f"{'=' * 60}")
    print(f"  {'Stage':<22} {'RF (px)':>8}")
    print(f"  {'-' * 22} {'-' * 8}")
    for module_name, rf in rf_dict.items():
        label = labels.get(module_name, module_name)
        print(f"  {label:<22} {rf:>8}")
    print()


# ---------------------------------------------------------------------------
# 3. Visualisation
# ---------------------------------------------------------------------------


def _filter_summary(
    results: list[tuple[str, int, int]],
    summary_layers: set[str],
) -> tuple[list[str], list[int]]:
    """Return names and RF values only at the specified summary checkpoints."""
    names = [r[0] for r in results if r[0] in summary_layers]
    rfs = [r[1] for r in results if r[0] in summary_layers]
    return names, rfs


def visualise_rf_growth(
    theoretical_eff: list[tuple[str, int, int]],
    theoretical_cnx: list[tuple[str, int, int]],
    empirical_eff: dict[str, int],
    empirical_cnx: dict[str, int],
    save_path: str = "receptive_field_comparison.png",
) -> None:
    """Plot theoretical RF growth curves and empirical estimates side-by-side."""
    eff_th_names, eff_th_rfs = _filter_summary(
        theoretical_eff, EFFICIENTNET_SUMMARY_LAYERS
    )
    cnx_th_names, cnx_th_rfs = _filter_summary(
        theoretical_cnx, CONVNEXT_SUMMARY_LAYERS
    )

    panels = [
        (
            "EfficientNet-B0",
            eff_th_names,
            eff_th_rfs,
            [EFFICIENTNET_STAGE_LABELS[k] for k in empirical_eff],
            list(empirical_eff.values()),
        ),
        (
            "ConvNeXt-Tiny",
            cnx_th_names,
            cnx_th_rfs,
            [CONVNEXT_STAGE_LABELS[k] for k in empirical_cnx],
            list(empirical_cnx.values()),
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Receptive Field Growth -- EfficientNet-B0 vs ConvNeXt-Tiny",
        fontsize=13,
        fontweight="bold",
    )

    for ax, (title, th_names, th_rfs, emp_labels, emp_values) in zip(axes, panels):
        ax.plot(
            range(len(th_rfs)),
            th_rfs,
            marker="o",
            color="steelblue",
            label="Theoretical",
            linewidth=2,
            markersize=6,
        )
        if emp_values:
            ax.plot(
                range(len(emp_values)),
                emp_values,
                marker="s",
                color="tomato",
                linestyle="--",
                label="Empirical",
                linewidth=2,
                markersize=6,
            )

        # Use the longer label list so every tick has a label.
        tick_labels = emp_labels if len(emp_labels) >= len(th_names) else th_names
        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Receptive Field (pixels)")
        ax.set_xlabel("Stage")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if th_rfs:
            ax.annotate(
                f"Final RF: {th_rfs[-1]}px",
                xy=(len(th_rfs) - 1, th_rfs[-1]),
                xytext=(-40, 10),
                textcoords="offset points",
                fontsize=8,
                color="steelblue",
                arrowprops={"arrowstyle": "->", "color": "steelblue", "lw": 1},
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {save_path}")


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\nReceptive Field Analysis: EfficientNet-B0 vs ConvNeXt-Tiny")
    print("=" * 60)

    # --- Theoretical ---
    theo_eff = theoretical_rf(EFFICIENTNET_B0_LAYERS)
    theo_cnx = theoretical_rf(CONVNEXT_TINY_LAYERS)

    print_theoretical("EfficientNet-B0", theo_eff, EFFICIENTNET_SUMMARY_LAYERS)
    print_theoretical("ConvNeXt-Tiny", theo_cnx, CONVNEXT_SUMMARY_LAYERS)

    final_eff = theo_eff[-1][1]
    final_cnx = theo_cnx[-1][1]
    print("  Summary -- Theoretical final RF:")
    print(f"    EfficientNet-B0 : {final_eff} px  (out of {INPUT_SIZE} px input)")
    print(f"    ConvNeXt-Tiny   : {final_cnx} px  (out of {INPUT_SIZE} px input)")
    print("\n  Coverage ratios:")
    print(f"    EfficientNet-B0 : {final_eff / INPUT_SIZE * 100:.1f}%")
    print(f"    ConvNeXt-Tiny   : {final_cnx / INPUT_SIZE * 100:.1f}%")

    # --- Empirical ---
    # Pretrained weights are required for ConvNeXt. Its layer scale parameter
    # (gamma) is initialised to 1e-6, making the depthwise-conv branch gradient
    # ~1e-6x the skip-connection gradient. Any practical threshold then sees only
    # the identity path, making every block look like a no-op and collapsing the
    # measured RF to the downsampler stride alone. Trained weights have learned
    # gamma values that reflect the branch's actual contribution, so the spatial
    # gradient spread is correct. EfficientNet behaves correctly with either
    # setting (no layer scale), but we use pretrained weights consistently.
    # First run will download ~20 MB (EfficientNet) and ~28 MB (ConvNeXt).
    print("\nRunning empirical RF analysis (gradient method)...")
    print("Loading pretrained torchvision models (downloads on first run)...")

    eff_model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )
    cnx_model = models.convnext_tiny(
        weights=models.ConvNeXt_Tiny_Weights.DEFAULT
    )

    emp_eff = empirical_rf(eff_model, stage_hooks=EFFICIENTNET_HOOK_NAMES)
    emp_cnx = empirical_rf(cnx_model, stage_hooks=CONVNEXT_HOOK_NAMES)

    print_empirical("EfficientNet-B0", emp_eff, EFFICIENTNET_STAGE_LABELS)
    print_empirical("ConvNeXt-Tiny", emp_cnx, CONVNEXT_STAGE_LABELS)

    # --- Comparison table ---
    emp_final_eff = list(emp_eff.values())[-1] if emp_eff else 0
    emp_final_cnx = list(emp_cnx.values())[-1] if emp_cnx else 0

    print(f"\n{'=' * 60}")
    print(f"  Final RF Comparison ({INPUT_SIZE}x{INPUT_SIZE} input)")
    print(f"{'=' * 60}")
    print(f"  {'Model':<22} {'Theoretical':>12}  {'Empirical':>10}  {'Coverage':>10}")
    print(f"  {'-' * 22} {'-' * 12}  {'-' * 10}  {'-' * 10}")
    print(
        f"  {'EfficientNet-B0':<22} {final_eff:>10}px"
        f"  {emp_final_eff:>8}px  {final_eff / INPUT_SIZE * 100:>9.1f}%"
    )
    print(
        f"  {'ConvNeXt-Tiny':<22} {final_cnx:>10}px"
        f"  {emp_final_cnx:>8}px  {final_cnx / INPUT_SIZE * 100:>9.1f}%"
    )
    print()

    # --- Visualise ---
    visualise_rf_growth(
        theo_eff,
        theo_cnx,
        emp_eff,
        emp_cnx,
        save_path="receptive_field_comparison.png",
    )

    # --- Key observations ---
    print("Key observations:\n")
    print(
        "  1. ConvNeXt's 7x7 depthwise convs accumulate RF aggressively.\n"
        "     Each block adds 6 x cumulative_stride pixels to the RF.\n"
        "     9 blocks in stage 3 alone add significant coverage.\n"
    )
    print(
        "  2. EfficientNet uses 3x3 and 5x5 kernels but accumulates RF\n"
        "     more slowly -- SE adds no spatial extent, and pointwise\n"
        "     (1x1) convolutions never grow the RF.\n"
    )
    print(
        "  3. Theoretical RF is often an overestimate. The gradient method\n"
        "     measures the *effective* RF -- pixels that actually influence\n"
        "     the output. In practice, effective RF is roughly Gaussian-\n"
        "     shaped and smaller than the theoretical bound (Luo et al., 2017).\n"
    )
    print(
        "  4. Both architectures exceed 224 px RF by the final stage,\n"
        "     meaning each output neuron theoretically sees the entire input.\n"
        "     This makes global average pooling semantically well-grounded."
    )


if __name__ == "__main__":
    main()