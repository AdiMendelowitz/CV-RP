"""
Shape-assertion tests for UNet and its submodules.

Run from the repo root:
    pytest "Advanced CV & Efficient Models/code/segmentation/test_unet.py" -v
"""

import torch
import pytest

from unet import DoubleConv, Down, Up, UNet


class TestDoubleConv:
    def test_output_shape_matches_input_spatial(self) -> None:
        module = DoubleConv(in_channels=3, out_channels=64)
        x = torch.zeros(2, 3, 64, 64)
        out = module(x)
        assert out.shape == (2, 64, 64, 64)

    def test_channel_transform(self) -> None:
        module = DoubleConv(in_channels=64, out_channels=128)
        x = torch.zeros(1, 64, 32, 32)
        out = module(x)
        assert out.shape[1] == 128

    def test_same_in_out_channels(self) -> None:
        module = DoubleConv(in_channels=64, out_channels=64)
        x = torch.zeros(1, 64, 16, 16)
        out = module(x)
        assert out.shape == (1, 64, 16, 16)


class TestDown:
    def test_halves_spatial_dimensions(self) -> None:
        module = Down(in_channels=64, out_channels=128)
        x = torch.zeros(2, 64, 64, 64)
        out = module(x)
        assert out.shape == (2, 128, 32, 32)

    def test_minimal_spatial_size(self) -> None:
        module = Down(in_channels=32, out_channels=64)
        x = torch.zeros(1, 32, 4, 4)
        out = module(x)
        assert out.shape == (1, 64, 2, 2)


class TestUp:
    def test_output_matches_skip_spatial(self) -> None:
        module = Up(in_channels=128, out_channels=64, bilinear=True)
        x = torch.zeros(2, 128, 32, 32)
        skip = torch.zeros(2, 64, 64, 64)
        out = module(x, skip)
        assert out.shape == (2, 64, 64, 64)

    def test_transpose_conv_variant(self) -> None:
        module = Up(in_channels=128, out_channels=64, bilinear=False)
        x = torch.zeros(2, 128, 32, 32)
        skip = torch.zeros(2, 64, 64, 64)
        out = module(x, skip)
        assert out.shape == (2, 64, 64, 64)

    def test_odd_spatial_size_padded(self) -> None:
        # When input H/W is not divisible by 16, upsampled tensor may be 1px smaller than skip.
        # Up.forward pads the upsampled tensor to match skip's spatial size.
        module = Up(in_channels=128, out_channels=64, bilinear=True)
        x = torch.zeros(1, 128, 16, 16)
        skip = torch.zeros(1, 64, 33, 33)
        out = module(x, skip)
        assert out.shape == (1, 64, 33, 33)


class TestUNet:
    @pytest.mark.parametrize(
        "in_channels,num_classes",
        [
            (1, 1),
            (3, 2),
            (1, 10),
        ],
    )
    def test_output_shape(self, in_channels: int, num_classes: int) -> None:
        model = UNet(in_channels=in_channels, num_classes=num_classes)
        model.eval()
        x = torch.zeros(2, in_channels, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, num_classes, 256, 256)

    def test_batch_size_one(self) -> None:
        model = UNet(in_channels=1, num_classes=1)
        model.eval()
        x = torch.zeros(1, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 128, 128)

    def test_non_square_input(self) -> None:
        model = UNet(in_channels=3, num_classes=2)
        model.eval()
        x = torch.zeros(1, 3, 256, 192)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 256, 192)

    def test_odd_spatial_input(self) -> None:
        # Verifies the padding logic in Up handles non-power-of-2 spatial dims.
        model = UNet(in_channels=1, num_classes=2)
        model.eval()
        x = torch.zeros(1, 1, 65, 65)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 65, 65)

    def test_output_dtype_matches_input(self) -> None:
        model = UNet(in_channels=1, num_classes=1)
        model.eval()
        x = torch.zeros(1, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == x.dtype

    def test_gradient_flows(self) -> None:
        model = UNet(in_channels=1, num_classes=1)
        x = torch.randn(1, 1, 128, 128)
        out = model(x)
        out.mean().backward()
        assert any(p.grad is not None for p in model.parameters())
