"""Unit tests for iTransformer.

Tests cover output shape, gradient flow, and the cross-variate interaction property --
the defining behavioral difference from PatchTST's channel independence.

Run with:
    pytest time-series-forecasting/tests/test_itransformer.py -v
"""

import torch
import pytest

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.itransformer import ForecastHead, VariateEmbedding, iTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_model() -> iTransformer:
    """iTransformer with paper-approximate config for ETTh1 (seq_len=96)."""
    return iTransformer(
        seq_len=96,
        pred_len=96,
        d_model=512,
        num_heads=8,
        num_layers=3,
        dropout=0.1,
    )


# ---------------------------------------------------------------------------
# VariateEmbedding
# ---------------------------------------------------------------------------

class TestVariateEmbedding:
    def test_output_shape(self) -> None:
        """(B, seq_len, C) -> (B, C, d_model)."""
        module = VariateEmbedding(seq_len=96, d_model=512)
        x = torch.randn(4, 96, 7)
        out = module(x)
        assert out.shape == (4, 7, 512), f"Expected (4, 7, 512), got {out.shape}"


# ---------------------------------------------------------------------------
# ForecastHead
# ---------------------------------------------------------------------------

class TestForecastHead:
    def test_output_shape(self) -> None:
        """(B, C, d_model) -> (B, pred_len, C)."""
        module = ForecastHead(d_model=512, pred_len=96)
        x = torch.randn(4, 7, 512)
        out = module(x)
        assert out.shape == (4, 96, 7), f"Expected (4, 96, 7), got {out.shape}"


# ---------------------------------------------------------------------------
# iTransformer end-to-end
# ---------------------------------------------------------------------------

class TestITransformer:
    def test_output_shape(self, default_model: iTransformer) -> None:
        """End-to-end shape: (4, 96, 7) -> (4, 96, 7) for pred_len=96."""
        x = torch.randn(4, 96, 7)
        out = default_model(x)
        assert out.shape == (4, 96, 7), f"Expected (4, 96, 7), got {out.shape}"

    def test_gradient_flow(self, default_model: iTransformer) -> None:
        """All parameters with requires_grad=True must receive a gradient."""
        x = torch.randn(4, 96, 7)
        loss = default_model(x).sum()
        loss.backward()

        for name, param in default_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter '{name}' has no gradient."

    def test_cross_variate_interaction(self, default_model: iTransformer) -> None:
        """Perturbing channel 0 must affect all output channels.

        This is the behavioral inverse of PatchTST's channel independence test.
        Because attention runs over variate tokens jointly, any perturbation to one
        variate's history propagates through the attention matrix to all other variates.
        """
        default_model.eval()
        torch.manual_seed(0)

        x = torch.randn(1, 96, 7)
        with torch.no_grad():
            baseline = default_model(x)

        x_perturbed = x.clone()
        x_perturbed[:, :, 0] += 10.0
        with torch.no_grad():
            perturbed = default_model(x_perturbed)

        delta = (perturbed - baseline).abs()
        # All channels must change, not just channel 0.
        for c in range(7):
            assert delta[:, :, c].max().item() > 1e-6, (
                f"Channel {c} did not change after perturbing channel 0. "
                "iTransformer should exhibit cross-variate interaction."
            )

    def test_invalid_heads_raises(self) -> None:
        """d_model not divisible by num_heads must raise ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            iTransformer(seq_len=96, pred_len=96, d_model=512, num_heads=7)