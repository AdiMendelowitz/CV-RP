"""Unit tests for models/patchtst.py.

Covers shape correctness, gradient flow, and channel independence.
All tests use synthetic tensors -- no external data downloads required.
"""

import pytest
import torch

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.patchtst import PatchEmbedding, PatchTST


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEQ_LEN = 512
PRED_LEN = 96
NUM_VARIATES = 7
BATCH = 4
PATCH_SIZE = 16
STRIDE = 8
D_MODEL = 128
NUM_HEADS = 16
NUM_LAYERS = 3
DROPOUT = 0.0  # disabled for deterministic tests


@pytest.fixture()
def patch_embedding() -> PatchEmbedding:
    return PatchEmbedding(patch_size=PATCH_SIZE, stride=STRIDE, d_model=D_MODEL, dropout=DROPOUT)


@pytest.fixture()
def model() -> PatchTST:
    return PatchTST(
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        num_variates=NUM_VARIATES,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_patch_embedding_output_shape(patch_embedding: PatchEmbedding) -> None:
    """PatchEmbedding output must be (B, 63, d_model) for seq_len=512, P=16, S=8."""
    x = torch.randn(BATCH, SEQ_LEN, 1)
    out = patch_embedding(x)
    assert out.shape == (BATCH, 63, D_MODEL), f"Expected (4, 63, 128), got {tuple(out.shape)}"


def test_patchtst_output_shape(model: PatchTST) -> None:
    """PatchTST end-to-end must produce (B, pred_len, C) from (B, seq_len, C)."""
    x = torch.randn(BATCH, SEQ_LEN, NUM_VARIATES)
    out = model(x)
    assert out.shape == (BATCH, PRED_LEN, NUM_VARIATES), f"Expected {(BATCH, PRED_LEN, NUM_VARIATES)}, got {tuple(out.shape)}"


# ---------------------------------------------------------------------------
# Gradient flow test
# ---------------------------------------------------------------------------


def test_gradient_flow(model: PatchTST) -> None:
    """All parameters with requires_grad=True must receive gradients after backward."""
    x = torch.randn(BATCH, SEQ_LEN, NUM_VARIATES)
    out = model(x)
    loss = out.mean()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter: {name}"


# ---------------------------------------------------------------------------
# Channel independence test
# ---------------------------------------------------------------------------


def test_channel_independence(model: PatchTST) -> None:
    """Perturbing channel 0 must change only the channel 0 slice of the output."""
    model.eval()
    with torch.no_grad():
        x = torch.randn(BATCH, SEQ_LEN, NUM_VARIATES)
        baseline = model(x).clone()

        x_perturbed = x.clone()
        x_perturbed[:, :, 0] += 100.0  # large perturbation to channel 0
        perturbed = model(x_perturbed)

    # Channel 0 output must differ.
    assert not torch.allclose(baseline[:, :, 0], perturbed[:, :, 0]), "Channel 0 output did not change after perturbation."

    # All other channels must remain identical.
    for c in range(1, NUM_VARIATES):
        assert torch.allclose(baseline[:, :, c], perturbed[:, :, c], atol=1e-5), (
            f"Channel {c} output changed after perturbing only channel 0 -- channel independence violated."
        )

# ---------------------------------------------------------------------------
# CD mode tests (channel_mixing=True)
# ---------------------------------------------------------------------------

class TestPatchTSTChannelMixing:
    def test_cd_output_shape(self) -> None:
        """CD mode must produce the same output shape as CI mode."""
        model = PatchTST(
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            num_variates=NUM_VARIATES,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout=0.0,
            channel_mixing=True,
        )
        x = torch.randn(BATCH, SEQ_LEN, NUM_VARIATES)
        out = model(x)
        assert out.shape == (BATCH, PRED_LEN, NUM_VARIATES), (
            f"Expected {(BATCH, PRED_LEN, NUM_VARIATES)}, got {out.shape}"
        )

    def test_cd_cross_channel_interaction(self) -> None:
        """In CD mode, perturbing channel 0 must change all output channels.

        This is the behavioral inverse of the CI channel independence test.
        In CD mode, attention runs over patches from all variates simultaneously,
        so a change to any variate's input propagates through the attention matrix
        to every other variate's output representation.
        """
        model = PatchTST(
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            num_variates=NUM_VARIATES,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout=0.0,
            channel_mixing=True,
        )
        model.eval()
        torch.manual_seed(0)

        x = torch.randn(1, SEQ_LEN, NUM_VARIATES)
        with torch.no_grad():
            baseline = model(x)

        x_perturbed = x.clone()
        x_perturbed[:, :, 0] += 10.0
        with torch.no_grad():
            perturbed = model(x_perturbed)

        delta = (perturbed - baseline).abs()
        for c in range(NUM_VARIATES):
            assert delta[:, :, c].max().item() > 1e-6, (
                f"Channel {c} output did not change after perturbing channel 0 in CD mode. "
                "Cross-variate interaction is expected in channel_mixing=True."
            )

    def test_cd_gradient_flow(self) -> None:
        """All parameters must receive gradients in CD mode."""
        model = PatchTST(
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            num_variates=NUM_VARIATES,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout=0.0,
            channel_mixing=True,
        )
        x = torch.randn(BATCH, SEQ_LEN, NUM_VARIATES)
        loss = model(x).sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter '{name}' has no gradient in CD mode."