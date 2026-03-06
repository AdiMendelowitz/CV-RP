"""
ConvNeXt implementation from scratch.

Reference: "A ConvNet for the 2020s" - Liu et al., 2022 https://arxiv.org/abs/2201.03545

Architecture: a pure ConvNet that adopts design choices from Vision Transformers — large depthwise kernels (7×7),
inverted bottleneck MLP, LayerNorm, GELU, fewer activation/norm layers, while remaining a standard convolutional network.
"""

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------------------------------------------------

def _drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    """
    Stochastic depth: drop entier paths during training. Scales surviving paths by 1/(1 - drop_prob) to preserve
    expected values. Equivalent to DropConnect applied to residual branches.

    Args:
        x: Input feature map of shape (B, C, H, W).
        drop_prob: Probability of zeroing a sample's residual branch.
        training: Is the model currently in training mode.

    Returns:
        Feature maps of shape (B, C, H, W).
    """
    if not training or drop_prob == 0.0:
        return x
    keep_prob = 1 - drop_prob
    mask = torch.bernoulli(
        torch.full((x.shape[0], 1, 1, 1), keep_prob, device=x.device, dtype=x.dtype)
    )
    return x*mask / keep_prob

class LayerNorm2d(nn.LayerNorm):
    """
    LayerNorm for NCHW tensors.
    nn.LayerNorm expects the normalized dimension last (NHWC). This class permutes NCHW <-> NHCW around the
    standard call so the rest of the network can remain in the default NCHW layout.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization to an NCHW feature map.

        Args:
            x: Feature map (B, C, H, W).
        Returns:
            Normalized feature map (B, C, H, W).
        """

        return super().forward(x.permute(0,2,3,1)).permute(0,3,1,2)

# ---------------------------------------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------------------------------------

class ConvNeXtBlock(nn.Module):
    """
    The atomic unit of the ConvNeXt architecture.

    Design:
    - 7x7 depthwise convolution (large receptive field, low FLOPs)
    - LayerNorm instead of BatchNorm
    - Inverted bottleneck MLP (4x expansion via 1x1  convs)
    - GELU activation
    - Layer scale: per-channel learnable multipliers initialized to 1e-6 to stabilize training of deep networks
    - Stochastic depth on the residual branch

    Args:
        dim: Number of input/output channels.
        layer_scale: Initial value for the per-channel learnable scale y.
        drop_path_rate: Probability of dropping the residual branch per step.
    """

    def __init__(self, dim: int, layer_scale: float = 1e-6, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        self.drop_path_rate = drop_path_rate

        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)

        # Inverted bottleneck MLP: expand 4x then project back
        self.pw_conv1 = nn.Conv2d(dim, 4*dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Conv2d(4*dim, dim, kernel_size=1)

        # y: (1, dim, 1, 1) broadcasts correctly in NCHW layout
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), layer_scale))

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply ConvNeXt block with residual connection.
        Spatial resolution and channel count are preserved, only the feature values change.

        Args:
            x: Feature map of shape (B, C, H, W).

        Returns:
             Feature maps of shape (B, C, H, W).
        """
        residual = x

        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.act(self.pw_conv1(x))
        x = self.pw_conv2(x) * self.gamma

        # Apply layer scale and stochastic depth to the residual branch, then add it back to the input
        x = _drop_path(x, self.drop_path_rate, self.training) + residual
        return x

# ---------------------------------------------------------------------------------------------------------------------
# ConvNeXt
# ---------------------------------------------------------------------------------------------------------------------

class ConvNeXt(nn.Module):
    """
    A pure ConvNet competitive with Swin Transformers at all scales.

    There are 4 stages, each runs a fixed number of ConvNeXt blocks at a given channel dimension.
    2x2 stride-2 convolution downsamples spatially and doubles the channel count between stages.
    Stem uses a 4x4 stride-4 "patchify" convolution, similar to patch embedding in ViTs.

    Args:
        depths: Number of ConvNeXt blocks in each stage.
        dims: Channel dimentions for each stage.
        num_classes: Number of output logits.
        drop_path_rate: Max stochastic depth rate (linearly scheduled).
        layer_scale: Initial value for per-channel learnable scale y.
    """

    def __init__(self, depths: list[int], dims: list[int], num_classes: int = 1000,
                 drop_path_rate: float = 0.0, layer_scale: float = 1e-6) -> None:
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("depth and dims must be lists of length 4, got {} and {}".format(depths, dims))

        # Patchify stem: non-overlapping 4x4 patches -> first stage channels. 224 -> 56 feature map (stride 4).
        self.stem = nn.Sequential(nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                                  LayerNorm2d(dims[0], eps=1e-6))

        # Linear drop-path schedule across all blocks
        total_blocks = sum(depths)
        dp_rates = [drop_path_rate*i / max(total_blocks-1, 1) for i in range(total_blocks)]

        # 4 stages + 3 inter-stage downsamples
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        block_idx = 0
        for stage_idx in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock(dims[stage_idx], layer_scale, dp_rates[block_idx + j])
                for j in range(depths[stage_idx])
            ])
            self.stages.append(stage)
            block_idx += depths[stage_idx]

            if stage_idx <3:
                self.downsamples.append(nn.Sequential(
                    LayerNorm2d(dims[stage_idx], eps=1e-6),
                    nn.Conv2d(dims[stage_idx], dims[stage_idx+1], kernel_size=2, stride=2)
                ))

        # Classifier head: global avg-pool > LayerNorm > Linear
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(dims[-1], eps=1e-6),
            nn.Linear(dims[-1], num_classes)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Run a full ConvNeXt forward pass.

        Args:
            x: Input image batch of shape (B, 3, H, W).

        Returns:
            Class logits of shape (B, num_classes).
        """
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        return self.head(x)

# ---------------------------------------------------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------------------------------------------------

def convnext_tiny(num_classes: int = 1000) -> ConvNeXt:
    """
    ConvNeXt-Tiny: depth=[3, 3, 9, 3], dims=[96, 192, 384, 768], ~28M parameters.

    Args:
        num_classes: Number of output logits for the classifier head.

    Returns:
        ConvNext-tiny configured with drop_path_rate=0.1.
    """
    return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes, drop_path_rate=0.1)

def convnext_small(num_classes: int = 1000) -> ConvNeXt:
    """
    ConvNeXt-Small: depth=[3, 3, 27, 3], dims=[96, 192, 384, 768], ~50M parameters.

    Returns:
        ConvNext-small configured with drop_path_rate=0.4.
    """
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], num_classes=num_classes, drop_path_rate=0.4)

def convnext_base(num_classes: int = 1000) -> ConvNeXt:
    """
    ConvNeXt-Base: depth=[3, 3, 27, 3], dims=[128, 256, 512, 1024], ~89M parameters.

    Returns:
        ConvNext-base configured with drop_path_rate=0.5.
    """
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=num_classes, drop_path_rate=0.5)


# ---------------------------------------------------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    from torchinfo import summary

    model = convnext_tiny()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"ConvNeXt-Tiny total parameters: {total_params:,} (expected ~28M)")

    summary(model, input_size=(1, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "kernel_size"], depth=3)

    # Sanity check: forward pass prdouces correct output shape
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 1000), f"Unexpected output shape {logits.shape}, expected (2, 1000)"
    print("Forward pass OK, output shape:", logits.shape)











