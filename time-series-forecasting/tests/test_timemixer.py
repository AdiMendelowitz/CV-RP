"""Unit tests for TimeMixer.

Tests cover end-to-end output shape, gradient flow, and per-scale intermediate
shapes before FMM -- verifying that each resolution produces a valid (B, C, d_model)
representation.

Run with:
    pytest time-series-forecasting/tests/test_timemixer.py -v
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

from models.timemixer import FMMBlock, PDMBlock, SeriesDecomposition, TimeMixer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEQ_LEN = 512
PRED_LEN = 96
BATCH = 4
C = 7
D_MODEL = 16
NUM_SCALES = 3
DECOMP_KERNEL = 25


@pytest.fixture()
def default_model() -> TimeMixer:
    return TimeMixer(
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        num_scales=NUM_SCALES,
        d_model=D_MODEL,
        decomp_kernel=DECOMP_KERNEL,
        dropout=0.1,
    )


# ---------------------------------------------------------------------------
# SeriesDecomposition
# ---------------------------------------------------------------------------


class TestSeriesDecomposition:
    def test_output_shapes_match_input(self) -> None:
        """seasonal and trend must have the same shape as the input."""
        module = SeriesDecomposition(kernel_size=25)
        x = torch.randn(BATCH, C, SEQ_LEN)
        seasonal, trend = module(x)
        assert seasonal.shape == x.shape, f"seasonal shape {seasonal.shape} != input {x.shape}"
        assert trend.shape == x.shape, f"trend shape {trend.shape} != input {x.shape}"

    def test_seasonal_plus_trend_equals_input(self) -> None:
        """seasonal + trend must reconstruct the input exactly."""
        module = SeriesDecomposition(kernel_size=25)
        x = torch.randn(BATCH, C, SEQ_LEN)
        seasonal, trend = module(x)
        torch.testing.assert_close(seasonal + trend, x)


# ---------------------------------------------------------------------------
# PDMBlock
# ---------------------------------------------------------------------------


class TestPDMBlock:
    def test_output_shape(self) -> None:
        """(B, C, L) seasonal + trend -> (B, C, d_model)."""
        block = PDMBlock(seq_len=SEQ_LEN, d_model=D_MODEL)
        seasonal = torch.randn(BATCH, C, SEQ_LEN)
        trend = torch.randn(BATCH, C, SEQ_LEN)
        out = block(seasonal, trend)
        assert out.shape == (BATCH, C, D_MODEL), f"Expected {(BATCH, C, D_MODEL)}, got {out.shape}"


# ---------------------------------------------------------------------------
# FMMBlock
# ---------------------------------------------------------------------------


class TestFMMBlock:
    def test_output_shape(self) -> None:
        """List of (B, C, d_model) tensors -> (B, C, pred_len)."""
        block = FMMBlock(num_scales=NUM_SCALES, d_model=D_MODEL, pred_len=PRED_LEN)
        reps = [torch.randn(BATCH, C, D_MODEL) for _ in range(NUM_SCALES)]
        out = block(reps)
        assert out.shape == (BATCH, C, PRED_LEN), f"Expected {(BATCH, C, PRED_LEN)}, got {out.shape}"

    def test_scale_weights_sum_to_one(self) -> None:
        """Softmax ensemble weights must sum to 1.0."""
        import torch.nn.functional as F

        block = FMMBlock(num_scales=NUM_SCALES, d_model=D_MODEL, pred_len=PRED_LEN)
        weights = F.softmax(block.scale_weights, dim=0)
        assert abs(weights.sum().item() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# TimeMixer end-to-end
# ---------------------------------------------------------------------------


class TestTimeMixer:
    def test_output_shape(self, default_model: TimeMixer) -> None:
        """(4, 512, 7) -> (4, 96, 7) for pred_len=96."""
        x = torch.randn(BATCH, SEQ_LEN, C)
        out = default_model(x)
        assert out.shape == (BATCH, PRED_LEN, C), f"Expected {(BATCH, PRED_LEN, C)}, got {out.shape}"

    def test_gradient_flow(self, default_model: TimeMixer) -> None:
        """All parameters with requires_grad=True must receive a gradient."""
        x = torch.randn(BATCH, SEQ_LEN, C)
        loss = default_model(x).sum()
        loss.backward()
        for name, param in default_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter '{name}' has no gradient."

    def test_per_scale_intermediate_shapes(self, default_model: TimeMixer) -> None:
        """Each scale must produce a valid (B, C, d_model) PDM representation."""
        x = torch.randn(BATCH, SEQ_LEN, C)
        x_s = x.transpose(1, 2)  # (B, C, seq_len)

        for s, (downsampler, pdm) in enumerate(zip(default_model.downsamplers, default_model.pdm_blocks)):
            # Pass the progressively downsampled tensor to the next downsampler
            x_s = downsampler(x_s)

            seasonal, trend = default_model.decomposition(x_s)
            rep = pdm(seasonal, trend)
            expected = (BATCH, C, D_MODEL)
            assert rep.shape == expected, f"Scale {s}: expected PDM output {expected}, got {rep.shape}"

    def test_different_pred_lens(self) -> None:
        """Model must produce correct shape for each forecast horizon."""
        for pred_len in [96, 192, 336, 720]:
            model = TimeMixer(seq_len=SEQ_LEN, pred_len=pred_len)
            x = torch.randn(BATCH, SEQ_LEN, C)
            out = model(x)
            assert out.shape == (
                BATCH,
                pred_len,
                C,
            ), f"pred_len={pred_len}: expected {(BATCH, pred_len, C)}, got {out.shape}"
