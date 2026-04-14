"""
U-Net: Convolutional Networks for Biomedical Image Segmentation.

Ronneberger, Fischer, Brox. MICCAI 2015. arXiv:1505.04597.

Implementation notes:
  - Uses same-padding (padding=1) on all 3x3 convolutions so the output spatial size matches the input.
    The original paper used valid (unpadded) convolutions, which shrinks 572x572 input to 388x388 output.
    The padded variant is standard in practice and avoids the skip-connection crop logic.
  - Conv -> BatchNorm -> ReLU ordering follows modern convention; the original paper predates widespread
  BatchNorm use in segmentation.
  - Up supports both bilinear upsampling (fewer parameters, smoother gradients) and transposed convolution
  (learnable upsampling, original paper style).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2

    The basic building block that's used in the encoder, decoder and bottleneck.
    In the original paper each stages applies 2 x unpadded 3x3 convolutions. Here padding=1 preserves spatial dimensions.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class Down(nn.Module):
    """
    One encoder step: 2x2 max pooling followed by a DoubleConv.

    Following the contracting path from the paper: spatial dimensions / 2, channel depth X 2.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels after the DoubleConv.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class Up(nn.Module):
    """
    One decoder step: upsample, concatenate skip connection, apply DoubleConv.

    The skip connection brings high-resolution spatial details from the matching encoder. If the upsampled tensor and
    the skip tensor differ in spatial size by 1 pixel (when input dims aren't divisible by 32), the upsampled tensor
    is padded to match.

    Args:
        in_channles: Number of input channels. This equals the sum of upsampled channels + skip channels =>
                     2 x out_channels from the previous Up stage.
        out_channels: Number of output channels after the DoubleConv.
        bilinear: True => use bilinear interpolation for upsampling.
                  False => use learnable ConvTranspose2d (original paper).
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            # Halve channels with a 1x1 conv before concatenation so DoubleConv recieves in_channels total (half from
            # upsample, half from skip), matching the transposed-conv branch's channel count.
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, in_channels//2, kernel_size=1),
            )
        else:
            # ConvTranspose2d doubles spatial dims and halves channels
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Pad x to match skip's spatial size if they differ by 1 pixel
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh > 0 or dw > 0:
            x = F.pad(x, [dw//2, dw - dw//2, dh//2, dh - dh//2])

        x = torch.cat([x, skip], dim=1)

        return self.conv(x)

class UNet(nn.Module):
    """
    Full U-Net encoder-decoder with 4 down-steps and 4 up-steps.

    Architecture follows figure 1 of Ronneberger et al. (2015), adapted for same-padding convolutions.
    Channel progression through the encoder: 64 -> 128 -> 256 -> 512 -> 1024.
    Decoder reverses this with skip connections concatenated at each matching resolution level.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output segmentation classes.
        bilinear: Upsampling strategy passed to all Up modules.
    """

    def __init__(self, in_channels: int, num_classes: int, bilinear: bool = True) -> None:
        super().__init__()

        # Encoder
        self.input_conv = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output projection
        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Encoder: save skip connection at each resolution level.
        s1 = self.input_conv(x)     # (B, 64, H, W)
        s2 = self.down1(s1)         # (B, 128, H/2, W/2)
        s3 = self.down2(s2)         # (B, 256, H/4, W/4)
        s4 = self.down3(s3)         # (B, 512, H/8, W/8)
        x = self.down4(s4)          # (B, 1024, H/16, W/16)

        # Decoder: upsample and fuse skip connections.
        x = self.up1(x, s4)         # (B, 512, H/8, W/8)
        x = self.up2(x, s3)         # (B, 256, H/4, W/4)
        x = self.up3(x, s2)         # (B, 128, H/2, W/2)
        x = self.up4(x, s1)         # (B, 64, H, W)

        return self.output_conv(x)      # (B, num_classes, H, W)


if __name__ == "__main__":
    model = UNet(in_channels=1, num_classes=2)
    model.eval()

    for h, w in [(64, 64), (128, 128), (65, 65)]:
        x = torch.zeros(1, 1, h, w)
        out = model(x)
        assert out.shape == (1, 2, h, w), f"Shape mismatch for input ({h},{w}): {out.shape}"
        print(f"Input {(1, 1, h, w)} -> Output {tuple(out.shape)} OK")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")


