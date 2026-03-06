"""
EfficientNet implementation from scratch

Reference: "EfficientNet: Rethinking Model Scaling for CNNs" - Tan & Le, 2019
https://arxiv.org/abs/1905.11946

Architecture: MBConv blocks with Squeeze-and-Excitation, scaled by a compound
coefficient that uniformly increases width, depth, and resolution.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------------------------------------------------


def _make_divisible(v: float, divisor: int = 8) -> int:
    """
    Round v to the nearest multiple of divisor (HW friendly channel count).
    Args:
        v: Raw channel count, typically the result of a scaling step
        divisor: Target alignment multiple; 8 suits most GPU tensor cores, but 4 may be used for mobile CPUs
    Returns:
        Smallest multiple of divisor such that v <= divisor < v*0.9
    """
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    """
    Stochastic depth (drop connect): drop entire sample paths during training.
    Scales surviving paths to 1/(1-drop_prob) and preserves expected value.

    Args:
        x: Input feature map of shape (B, C, H, W).
        drop_prob: Probability of zeroing a sample's residual branch.
        training: Is the model in training mode.

    Returns:
        Feature map of shape (B, C, H, W) with some samples' residual branches dropped.
    """

    if not training or drop_prob == 0.0:
        return x
    keep_prob = 1 - drop_prob
    # Shape (B, 1, 1, 1) to broadcast over C, H, W
    mask = torch.bernoulli(torch.full((x.shape[0], 1, 1, 1), keep_prob, device=x.device, dtype=x.dtype))
    return x * mask / keep_prob


# ---------------------------------------------------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------------------------------------------------


class SqueezeExcitation(nn.Module):
    """
    Channel-wise recalibration: global avg-pool -> FC bottleneck -> sigmoid gate.
    SE ratio of 0.25 is the paper's default.

    Args:
        in_channels: Number of input channels.
        reduced_channels: Bottleneck width; typically int(in_channels * se_ratio).
    """

    def __init__(self, in_channels: int, reduced_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.act = nn.SiLU()
        self.expand = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply channel-wise SE recalibration.

        Args:
            x: Feature map of shape (B, C, H, W).

        Returns:
            Recalibrated feature map of shape (B, C, H, W). Each channel is scaled by a learnable sigmoid
            gate derived by global context.
        """
        scale = self.pool(x)
        scale = self.act(self.reduce(scale))
        scale = torch.sigmoid(self.expand(scale))
        return x * scale


class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution with squeeze and excitation.

    Structure (if expand_ratio > 1): [1x1 expand] -> [k*k depthwise] -> [SE] -> [1x1 project]
    Skip connection is used only when stride=1 and in_channels == out_channels.
    Stochastic depth is applied only to the residual branch.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Depthwise convolution kernel size (3 or 5 in B0).
        stride: Depthwise convolution stride.
        expand_ratio: Channels expansion factor applied before depthwise convolution; if 1, expand phase is skipped.
        se_ratio: SE bottleneck ratio relative to in_channels, default 0.25.
        drop_path_rate: Stochastic depth probability for the residual branch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.use_residual = stride == 1 and in_channels == out_channels
        self.drop_path_rate = drop_path_rate

        mid_channels = _make_divisible(in_channels * expand_ratio)
        # SE reduced channels are relative to in_channels, not expanded
        reduced_channels = max(1, int(in_channels * se_ratio))
        padding = (kernel_size - 1) // 2

        layers: list[nn.Module] = []

        # Pointwise expansion (if expand_ratio > 1, e.g. first MBconv stage)
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels, momentum=0.01, eps=1e-3),
                nn.SiLU(),
            ]

        # Depthwise conv + SE + pointwise projection
        layers += [
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size,
                stride,
                padding,
                groups=mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels, momentum=0.01, eps=1e-3),
            nn.SiLU(),
            SqueezeExcitation(mid_channels, reduced_channels),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply MBconv block with optional skip connection.

        Args:
            x: Input feature map of shape (B, in_channels, H, W).

        Returns:
            Output feature map of shape (B, out_channels, H_out, W_out).
        """

        # H_out = H/stride. When skip connection is active, the output = sum of projected features + identity.
        out = self.block(x)
        if self.use_residual:
            out = _drop_path(out, self.drop_path_rate, self.training)
            out += x
        return out


# ---------------------------------------------------------------------------------------------------------------------
# EfficientNet Architecture
# ---------------------------------------------------------------------------------------------------------------------


class EfficientNet(nn.Module):
    """
    EfficientNet: uniformly scalable CNN via compound scaling coefficient.

    B0 baseline is defined by _BASE_BLOCKS. B1-B7 scale width/depth proportionally.
    Drop-path rates increase linearly across all blocks from 0 to _MAX_DROP_PATH_RATE.

    Args:
        width_multiplier: Channel width multiplier. >1.0 widen the network, <1.0 narrow it.
        depth_multiplier: Depth multiplier. >1.0 deepen the network, <1.0 shrink it.
        dropout: Dropout rate applied before the final classifier, default 0.2 for B0.
        num_classes: Number of output logits.
    """

    # B0 baseline configuration: (expand_ratio, out_channels, repeats, stride, kernel_size)
    _BASE_BLOCKS: list[tuple[int, int, int, int, int]] = [
        (1, 16, 1, 1, 3),
        (6, 24, 2, 2, 3),
        (6, 40, 2, 2, 5),
        (6, 80, 3, 2, 3),
        (6, 112, 3, 1, 5),
        (6, 192, 4, 2, 5),
        (6, 320, 1, 1, 3),
    ]
    _MAX_DROP_PATH_RATE: float = 0.2

    def __init__(
        self,
        width_multiplier: float = 1.0,
        depth_multiplier: float = 1.0,
        dropout: float = 0.2,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()

        def scale_ch(c: int) -> int:
            return _make_divisible(c * width_multiplier)

        def scale_d(d: int) -> int:
            return max(1, math.ceil(d * depth_multiplier))

        total_blocks = sum(scale_d(r) for _, _, r, _, _ in self._BASE_BLOCKS)

        stem_ch = scale_ch(32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch, momentum=0.01, eps=1e-3),
            nn.SiLU(),
        )

        blocks: list[nn.Module] = []
        block_idx = 0
        in_ch = stem_ch
        for expand_ratio, out_ch, repeats, stride, kernel_size in self._BASE_BLOCKS:
            out_ch = scale_ch(out_ch)
            for i in range(scale_d(repeats)):
                # Linear schedule: rate increases 0 -> _MAX_DROP_PATH_RATE
                drop_path_rate = self._MAX_DROP_PATH_RATE * block_idx / max(total_blocks - 1, 1)
                blocks.append(
                    MBConv(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel_size,
                        stride=stride if i == 0 else 1,
                        expand_ratio=expand_ratio,
                        drop_path_rate=drop_path_rate,
                    )
                )
                in_ch = out_ch
                block_idx += 1

        self.blocks = nn.Sequential(*blocks)

        head_ch = scale_ch(1280)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_ch, momentum=0.01, eps=1e-3),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(head_ch, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Run a full EfficientNet forward pass.
        Args:
            x: Input image batch of shape (B, 3, H, W).
        Returns:
            Class logits of shape (B, num_classes).
        """

        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


# ---------------------------------------------------------------------------------------------------------------------
# Factory functions for EfficientNet variants B0-B7 with different scaling coefficients.
# ----------------------------------------------------------------------------------------------------------------------
def efficientnet_b0(num_classes: int = 1000) -> EfficientNet:
    """
    EfficientNet-B0: baseline model with compound coefficient 1.0. (~5.3M parameters)

    Args:
        num_classes: Number of output logits for the classifier head.

    Returns:
        EfficientNet configured with width_multiplier=1.0 and depth_multiplier=1.0.
    """
    return EfficientNet(width_multiplier=1.0, depth_multiplier=1.0, dropout=0.2, num_classes=num_classes)


def efficientnet_b1(num_classes: int = 1000) -> EfficientNet:
    """EfficientNet-B1: depth-scaled variant (~7.8M parameters)"""
    return EfficientNet(width_multiplier=1.0, depth_multiplier=1.1, dropout=0.2, num_classes=num_classes)


def efficientnet_b4(num_classes: int = 1000) -> EfficientNet:
    """EfficientNet-B4: width and depth scaled variant (~19M parameters)"""
    return EfficientNet(width_multiplier=1.4, depth_multiplier=1.8, dropout=0.4, num_classes=num_classes)


# ----------------------------------------------------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    from torchinfo import summary

    model = efficientnet_b0()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"EfficientNet-B0 total parameters: {total_params/1e6:.2f}M, expected ~5.3M")

    summary(
        model,
        input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        depth=3,
    )

    # Sanity check: forward pass with dummy input
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (
        2,
        1000,
    ), f"Expected output shape (2, 1000), got {logits.shape}"
    print("\nForward pass OK, output shape:", logits.shape)
