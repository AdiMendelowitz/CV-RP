"""
Tests for roi_align.py and mask_head.py.

All tests use synthetic data; no external downloads are required.
"""

import torch
from roi_align import bilinear_interpolation, roi_align
from mask_head import MaskHead

# ---------------------------------------------------------------------------
# roi_align
# ---------------------------------------------------------------------------


class TestRoIAlign:

    def test_output_shape(self):
        """Output shape must be (R, C, output_size, output_size)."""
        C, H, W = 32, 64, 64
        R, output_size = 5, 7
        feature_map = torch.rand(C, H, W)
        boxes = torch.tensor(
            [
                [5.0, 5.0, 20.0, 20.0],
                [10.0, 10.0, 40.0, 40.0],
                [0.0, 0.0, 63.0, 63.0],
                [15.0, 8.0, 30.0, 50.0],
                [2.0, 2.0, 10.0, 10.0],
            ]
        )
        out = roi_align(feature_map, boxes, output_size=output_size)
        assert out.shape == (
            R,
            C,
            output_size,
            output_size,
        ), f"Expected shape ({R}, {C}, {output_size}, {output_size}), got {out.shape}"

    def test_float_coords_differ_from_snapped_integer_coords(self):
        """Bilinear interpolation at fractional coordinates must differ from nearest-integer sampling."""
        torch.manual_seed(0)
        C, H, W = 8, 32, 32
        feature_map = torch.rand(C, H, W)

        # A box with fractional coordinates.
        box_float = torch.tensor([[4.3, 4.7, 15.9, 16.2]])
        # The same box snapped to integer pixel boundaries.
        box_int = torch.tensor([[4.0, 5.0, 16.0, 16.0]])

        out_float = roi_align(feature_map, box_float, output_size=7)
        out_int = roi_align(feature_map, box_int, output_size=7)

        assert not torch.allclose(out_float, out_int, atol=1e-4), (
            "Float-coordinate RoI and snapped integer RoI produced identical outputs; "
            "bilinear interpolation is not having an effect."
        )

    def test_identical_boxes_produce_identical_output(self):
        """Two identical boxes must produce identical pooled features."""
        C, H, W = 16, 32, 32
        feature_map = torch.rand(C, H, W)
        box = torch.tensor([[5.0, 5.0, 25.0, 25.0]])
        boxes = box.expand(3, -1)  # three identical boxes

        out = roi_align(feature_map, boxes, output_size=7)
        assert torch.allclose(out[0], out[1], atol=1e-6) and torch.allclose(
            out[1], out[2], atol=1e-6
        ), "Identical boxes produced different pooled features."

    def test_output_values_finite(self):
        """All output values must be finite for valid inputs."""
        C, H, W = 8, 32, 32
        feature_map = torch.rand(C, H, W)
        boxes = torch.tensor([[1.0, 1.0, 15.0, 15.0], [8.0, 8.0, 30.0, 30.0]])
        out = roi_align(feature_map, boxes, output_size=7)
        assert torch.isfinite(out).all(), "RoI Align output contains NaN or Inf."

    def test_single_pixel_feature_map_returns_constant(self):
        """On a constant feature map, every RoI must return that constant value."""
        C, H, W = 4, 16, 16
        fill_value = 3.7
        feature_map = torch.full((C, H, W), fill_value)
        boxes = torch.tensor([[2.0, 2.0, 12.0, 12.0]])
        out = roi_align(feature_map, boxes, output_size=5)
        assert torch.allclose(
            out, torch.full_like(out, fill_value), atol=1e-5
        ), "Constant feature map did not produce constant RoI output."


# ---------------------------------------------------------------------------
# bilinear_interpolate
# ---------------------------------------------------------------------------


class TestBilinearInterpolate:

    def test_output_shape(self):
        """Output shape must be (C, N) for N sample points."""
        C, H, W, N = 16, 32, 32, 20
        feature_map = torch.rand(C, H, W)
        y = torch.rand(N) * (H - 1)
        x = torch.rand(N) * (W - 1)
        out = bilinear_interpolation(feature_map, y, x)
        assert out.shape == (C, N), f"Expected shape ({C}, {N}), got {out.shape}"

    def test_integer_coords_match_direct_index(self):
        """At exact integer coordinates, interpolation must equal direct pixel lookup."""
        C, H, W = 8, 16, 16
        feature_map = torch.rand(C, H, W)
        ys = torch.tensor([2.0, 5.0, 10.0])
        xs = torch.tensor([3.0, 7.0, 1.0])
        out = bilinear_interpolation(feature_map, ys, xs)
        for i, (r, c) in enumerate(zip([2, 5, 10], [3, 7, 1])):
            assert torch.allclose(
                out[:, i], feature_map[:, r, c], atol=1e-6
            ), f"Interpolated value at integer coord ({r},{c}) does not match direct lookup."


# ---------------------------------------------------------------------------
# MaskHead
# ---------------------------------------------------------------------------


class TestMaskHead:

    def test_output_shape(self):
        """Output shape must be (B, num_classes, 28, 28) for (B, in_channels, 14, 14) input."""
        B, in_channels, num_classes = 4, 256, 80
        head = MaskHead(in_channels=in_channels, num_classes=num_classes)
        x = torch.rand(B, in_channels, 14, 14)
        out = head(x)
        assert out.shape == (B, num_classes, 28, 28), f"Expected shape ({B}, {num_classes}, 28, 28), got {out.shape}"

    def test_gradient_flows_to_all_parameters(self):
        """A backward pass must produce non-None, non-zero gradients for every parameter."""
        B, in_channels, num_classes = 2, 64, 10
        head = MaskHead(in_channels=in_channels, num_classes=num_classes)
        x = torch.rand(B, in_channels, 14, 14)
        loss = head(x).sum()
        loss.backward()

        for name, param in head.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert param.grad.abs().sum().item() > 0.0, f"Zero gradient for parameter: {name}"

    def test_hidden_channels_respected(self):
        """Custom hidden_channels must not raise and must produce the correct output shape."""
        B, in_channels, num_classes, hidden = 2, 128, 5, 64
        head = MaskHead(in_channels=in_channels, num_classes=num_classes, hidden_channels=hidden)
        x = torch.rand(B, in_channels, 14, 14)
        out = head(x)
        assert out.shape == (B, num_classes, 28, 28), f"Expected shape ({B}, {num_classes}, 28, 28), got {out.shape}"

    def test_output_is_raw_logits(self):
        """Output must be unbounded raw logits, not probabilities in [0, 1]."""
        B, in_channels, num_classes = 2, 64, 5
        head = MaskHead(in_channels=in_channels, num_classes=num_classes)
        x = torch.rand(B, in_channels, 14, 14)
        out = head(x)
        has_values_outside_unit = (out.abs() > 1.0).any()
        assert (
            has_values_outside_unit
        ), "All output values are in [-1, 1]; expected raw logits with values outside this range."

    def test_output_values_finite(self):
        """Output must not contain NaN or Inf for valid inputs."""
        B, in_channels, num_classes = 2, 64, 5
        head = MaskHead(in_channels=in_channels, num_classes=num_classes)
        x = torch.rand(B, in_channels, 14, 14)
        out = head(x)
        assert torch.isfinite(out).all(), "MaskHead output contains NaN or Inf."
