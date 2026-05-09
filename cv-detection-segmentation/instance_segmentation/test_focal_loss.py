"""
Unit tests for focal_loss.

Coverage:
- Exact equivalence to nn.CrossEntropyLoss(weight=alpha) when gamma=0.
- Monotone decrease of per-sample focal weight with increasing gamma.
- Shape and device invariants.
- Input validation.
"""

import pytest
import torch
import torch.nn as nn

from focal_loss import focal_loss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small deterministic batch: N=8, C=7 (matching HAM10000 task)."""
    torch.manual_seed(0)
    N, C = 8, 7
    logits = torch.randn(N, C)
    targets = torch.randint(0, C, (N,))
    alpha = torch.tensor([1.5, 1.2, 0.8, 3.0, 1.1, 0.5, 4.0])  # inverse-freq style
    return logits, targets, alpha


# ---------------------------------------------------------------------------
# Correctness: gamma=0 must recover weighted CE exactly
# ---------------------------------------------------------------------------

class TestGammaZeroEquivalence:
    def test_scalar_equals_weighted_ce(self, batch: tuple) -> None:
        logits, targets, alpha = batch
        fl = focal_loss(logits, targets, alpha, gamma=0.0)
        ce = nn.CrossEntropyLoss(weight=alpha)(logits, targets)
        assert torch.allclose(fl, ce, atol=1e-6), (f"focal_loss(gamma=0) = {fl.item():.8f}, "
                                                   f"CrossEntropyLoss    = {ce.item():.8f}")

    def test_equivalence_single_class_dominant(self) -> None:
        """Holds even when alpha weights are highly imbalanced."""
        torch.manual_seed(1)
        logits = torch.randn(16, 7)
        targets = torch.zeros(16, dtype=torch.long)   # all same class
        alpha = torch.tensor([0.1, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        fl = focal_loss(logits, targets, alpha, gamma=0.0)
        ce = nn.CrossEntropyLoss(weight=alpha)(logits, targets)
        assert torch.allclose(fl, ce, atol=1e-6)

    def test_equivalence_uniform_alpha(self) -> None:
        """With uniform alpha, focal(gamma=0) = standard CE (up to alpha scale)."""
        torch.manual_seed(2)
        logits = torch.randn(12, 4)
        targets = torch.randint(0, 4, (12,))
        alpha = torch.ones(4)
        fl = focal_loss(logits, targets, alpha, gamma=0.0)
        ce = nn.CrossEntropyLoss(weight=alpha)(logits, targets)
        assert torch.allclose(fl, ce, atol=1e-6)


# ---------------------------------------------------------------------------
# Correctness: focal weight decreases monotonically with gamma
# ---------------------------------------------------------------------------

class TestFocalWeightMonotonicity:
    def test_loss_decreases_as_gamma_increases_for_easy_examples(self) -> None:
        """
        For easy examples (high p_t), increasing gamma should reduce loss.

        Construct logits so a single class dominates, making p_t close to 1 for most samples.
        The focal weight (1-p_t)^gamma falls as gamma rises, so the mean loss must be strictly decreasing in gamma.
        """
        torch.manual_seed(3)
        N, C = 32, 7
        # Large logit for class 0 so p_t is high for all samples.
        logits  = torch.full((N, C), -5.0)
        logits[:, 0] = 10.0
        targets = torch.zeros(N, dtype=torch.long)
        alpha   = torch.ones(C)

        gammas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
        losses = [focal_loss(logits, targets, alpha, gamma=g).item() for g in gammas]

        for i in range(len(losses) - 1):
            assert losses[i] > losses[i + 1], (
                f"Expected loss to decrease: gamma={gammas[i]} -> {gammas[i+1]}, "
                f"losses={losses[i]:.6f} -> {losses[i+1]:.6f}"
            )

    def test_focal_weight_formula_directly(self) -> None:
        """
        For a fixed p_t in (0, 1), (1-p_t)^gamma is strictly decreasing in gamma.
        This is the mathematical property the loss relies on.
        """
        p_t = torch.tensor([0.9, 0.7, 0.5, 0.3])  # various confidence levels
        gammas = [0.0, 0.5, 1.0, 2.0, 5.0]
        for pt in p_t:
            weights = [(1.0 - pt.item()) ** g for g in gammas]
            for i in range(len(weights) - 1):
                assert weights[i] >= weights[i + 1], (
                    f"p_t={pt.item()}: weight not decreasing at gamma={gammas[i+1]}"
                )


# ---------------------------------------------------------------------------
# Output properties
# ---------------------------------------------------------------------------

class TestOutputProperties:
    def test_output_is_scalar(self, batch: tuple) -> None:
        logits, targets, alpha = batch
        out = focal_loss(logits, targets, alpha, gamma=2.0)
        assert out.shape == torch.Size([])

    def test_output_is_non_negative(self, batch: tuple) -> None:
        logits, targets, alpha = batch
        out = focal_loss(logits, targets, alpha, gamma=2.0)
        assert out.item() >= 0.0

    def test_gradient_flows(self, batch: tuple) -> None:
        logits, targets, alpha = batch
        logits = logits.requires_grad_(True)
        focal_loss(logits, targets, alpha, gamma=2.0).backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_perfect_predictions_low_loss(self) -> None:
        """With very confident correct predictions, focal loss should be near zero."""
        C = 7
        logits  = torch.eye(C) * 50.0         # one-hot style, 7 samples
        targets = torch.arange(C)
        alpha   = torch.ones(C)
        loss = focal_loss(logits, targets, alpha, gamma=2.0)
        assert loss.item() < 1e-3


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_wrong_logits_ndim(self, batch: tuple) -> None:
        logits, targets, alpha = batch
        with pytest.raises(ValueError, match="2-D"):
            focal_loss(logits.unsqueeze(0), targets, alpha)

    def test_wrong_targets_shape(self, batch: tuple) -> None:
        logits, targets, alpha = batch
        with pytest.raises(ValueError, match="targets"):
            focal_loss(logits, targets.unsqueeze(1), alpha)

    def test_wrong_alpha_length(self, batch: tuple) -> None:
        logits, targets, alpha = batch
        with pytest.raises(ValueError, match="alpha length"):
            focal_loss(logits, targets, alpha[:-1])

    def test_negative_gamma(self, batch: tuple) -> None:
        logits, targets, alpha = batch
        with pytest.raises(ValueError, match="gamma"):
            focal_loss(logits, targets, alpha, gamma=-1.0)