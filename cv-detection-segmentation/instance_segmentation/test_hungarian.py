"""
Tests for hungarian_loss.py

covers compute_giou, build_cost_matrix, hungaria_match and set_prediction_loss.
All tests use synthetic data, no external download are required.
"""

import torch
import pytest
from hungarian_loss import build_cost_matrix, compute_giou, hungarian_match, set_prediction_loss

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_boxes(*args: tuple) -> torch.Tensor:
    """Returns multiple boxes as an (N, 4) tensor. Each arg is (cx, cy, w, h)."""
    return torch.tensor(args, dtype=torch.float32)


# ---------------------------------------------------------------------------
# compute_giou
# ---------------------------------------------------------------------------


class TestComputeGIoU:

    def test_perfect_overlap_returns_one(self):
        """A box compared against itself must give GIoU = 1.0."""
        boxes = _make_boxes(
            (0.5, 0.5, 0.4, 0.4),
            (0.2, 0.3, 0.2, 0.1),
        )
        result = compute_giou(boxes, boxes)  # (2, 2)
        diagonal = result.diagonal()
        assert torch.allclose(diagonal, torch.ones(2), atol=1e-5), f"Expected diagonal GIoU = 1.0, got {diagonal}"

    def test_non_overlapping_boxes_leq_zero(self):
        """Boxes with no intersection must give GIoU <= 0."""
        # Box A: top-left quadrant. Box B: bottom-right quadrant. No overlap.
        box_a = _make_boxes((0.1, 0.1, 0.1, 0.1))
        box_b = _make_boxes((0.9, 0.9, 0.1, 0.1))
        result = compute_giou(box_a, box_b)  # (1, 1)
        assert result.item() <= 0.0, f"Expected GIoU <= 0 for non-overlapping boxes, got {result.item():.4f}"

    def test_non_overlapping_boxes_geq_minus_one(self):
        """GIoU must stay within the valid range [-1, 1]."""
        box_a = _make_boxes((0.1, 0.1, 0.1, 0.1))
        box_b = _make_boxes((0.9, 0.9, 0.1, 0.1))
        result = compute_giou(box_a, box_b)
        assert result.item() >= -1.0, f"GIoU below -1.0: {result.item():.4f}"

    def test_symmetry(self):
        """GIoU(A, B) must equal GIoU(B, A)^T."""
        boxes_a = _make_boxes(
            (0.3, 0.3, 0.2, 0.2),
            (0.7, 0.7, 0.3, 0.3),
        )
        boxes_b = _make_boxes(
            (0.4, 0.4, 0.1, 0.1),
            (0.6, 0.2, 0.2, 0.2),
            (0.8, 0.8, 0.1, 0.1),
        )
        ab = compute_giou(boxes_a, boxes_b)  # (2, 3)
        ba = compute_giou(boxes_b, boxes_a)  # (3, 2)
        assert torch.allclose(ab, ba.T, atol=1e-5), "GIoU matrix is not symmetric under transposition."

    def test_output_shape(self):
        """Output shape must be (N, M) for inputs of shape (N, 4) and (M, 4)."""
        boxes_a = torch.rand(5, 4).abs()
        boxes_b = torch.rand(3, 4).abs()
        # Ensure valid cx,cy,w,h: keep w,h small to avoid going out of [0,1]
        boxes_a[:, 2:] = boxes_a[:, 2:] * 0.3 + 0.05
        boxes_b[:, 2:] = boxes_b[:, 2:] * 0.3 + 0.05
        result = compute_giou(boxes_a, boxes_b)
        assert result.shape == (5, 3), f"Expected shape (5, 3), got {result.shape}"


# ---------------------------------------------------------------------------
# build_cost_matrix
# ---------------------------------------------------------------------------


class TestBuildCostMatrix:

    def test_output_shape(self):
        """Output shape must be (num_predictions, num_targets)."""
        N, M, C = 10, 4, 3  # 10 queries, 4 targets, 3 classes (+1 no-object = 4 logits)
        pred_logits = torch.randn(N, C + 1)
        pred_boxes = torch.rand(N, 4)
        pred_boxes[:, 2:] = pred_boxes[:, 2:] * 0.3 + 0.05
        target_labels = torch.randint(0, C, (M,))
        target_boxes = torch.rand(M, 4)
        target_boxes[:, 2:] = target_boxes[:, 2:] * 0.3 + 0.05

        cost = build_cost_matrix(pred_logits, pred_boxes, target_labels, target_boxes)
        assert cost.shape == (N, M), f"Expected shape ({N}, {M}), got {cost.shape}"

    def test_all_values_finite(self):
        """Cost matrix must not contain NaN or Inf for valid inputs."""
        N, M, C = 8, 3, 5
        pred_logits = torch.randn(N, C + 1)
        pred_boxes = torch.rand(N, 4)
        pred_boxes[:, 2:] = pred_boxes[:, 2:] * 0.3 + 0.05
        target_labels = torch.randint(0, C, (M,))
        target_boxes = torch.rand(M, 4)
        target_boxes[:, 2:] = target_boxes[:, 2:] * 0.3 + 0.05

        cost = build_cost_matrix(pred_logits, pred_boxes, target_labels, target_boxes)
        assert torch.isfinite(cost).all(), "Cost matrix contains NaN or Inf values."

    def test_custom_weights_scale_cost(self):
        """Doubling cost_bbox must double the L1 contribution."""
        N, M, C = 4, 2, 2
        pred_logits = torch.zeros(N, C + 1)
        pred_boxes = torch.full((N, 4), 0.5)
        pred_boxes[:, 2:] = 0.2
        target_labels = torch.zeros(M, dtype=torch.long)
        target_boxes = torch.full((M, 4), 0.3)
        target_boxes[:, 2:] = 0.2

        cost_base = build_cost_matrix(
            pred_logits,
            pred_boxes,
            target_labels,
            target_boxes,
            cost_class=0.0,
            cost_bbox=1.0,
            cost_giou=0.0,
        )
        cost_double = build_cost_matrix(
            pred_logits,
            pred_boxes,
            target_labels,
            target_boxes,
            cost_class=0.0,
            cost_bbox=2.0,
            cost_giou=0.0,
        )
        assert torch.allclose(
            cost_double, 2.0 * cost_base, atol=1e-5
        ), "Doubling cost_bbox did not double the cost matrix."


# ---------------------------------------------------------------------------
# hungarian_match
# ---------------------------------------------------------------------------


class TestHungarianMatch:

    def test_diagonal_cost_returns_identity_permutation(self):
        """A diagonal cost matrix (zero on diagonal, ones elsewhere) must map i -> i."""
        M = 4
        cost_matrix = torch.ones(M, M) - torch.eye(M)  # 0 on diagonal, 1 elsewhere
        pred_idx, tgt_idx = hungarian_match(cost_matrix)

        assert torch.equal(
            pred_idx, torch.arange(M)
        ), f"Expected pred_indices = {list(range(M))}, got {pred_idx.tolist()}"
        assert torch.equal(tgt_idx, torch.arange(M)), f"Expected tgt_indices = {list(range(M))}, got {tgt_idx.tolist()}"

    def test_assignment_is_one_to_one(self):
        """No prediction index must appear more than once in the result."""
        N, M = 10, 4
        cost_matrix = torch.rand(N, M)
        pred_idx, tgt_idx = hungarian_match(cost_matrix)

        assert len(pred_idx) == M, f"Expected {M} assignments, got {len(pred_idx)}"
        assert len(pred_idx.unique()) == M, f"Duplicate prediction indices in assignment: {pred_idx.tolist()}"
        assert len(tgt_idx.unique()) == M, f"Duplicate target indices in assignment: {tgt_idx.tolist()}"

    def test_returns_long_tensors(self):
        """Returned indices must be LongTensors."""
        cost_matrix = torch.rand(5, 3)
        pred_idx, tgt_idx = hungarian_match(cost_matrix)
        assert pred_idx.dtype == torch.long, f"pred_idx dtype: {pred_idx.dtype}"
        assert tgt_idx.dtype == torch.long, f"tgt_idx dtype: {tgt_idx.dtype}"

    def test_minimum_cost_assignment(self):
        """The matched assignment must have lower total cost than a known suboptimal one."""
        # Cost matrix where the optimal assignment is anti-diagonal.
        cost_matrix = torch.tensor(
            [
                [10.0, 1.0],
                [1.0, 10.0],
            ]
        )
        pred_idx, tgt_idx = hungarian_match(cost_matrix)
        optimal_cost = cost_matrix[pred_idx, tgt_idx].sum().item()
        # Anti-diagonal: pred 0 -> tgt 1, pred 1 -> tgt 0, total cost = 2.0
        # Diagonal:      pred 0 -> tgt 0, pred 1 -> tgt 1, total cost = 20.0
        assert optimal_cost == pytest.approx(2.0), f"Expected optimal cost 2.0, got {optimal_cost}"


# ---------------------------------------------------------------------------
# set_prediction_loss
# ---------------------------------------------------------------------------


class TestSetPredictionLoss:

    def _make_perfect_inputs(self):
        """
        Construct a batch where predictions exactly match targets.

        N=3 queries, M=2 targets, num_classes=2.
        - Query 0 predicts class 0 with certainty and box = target box 0.
        - Query 1 predicts class 1 with certainty and box = target box 1.
        - Query 2 predicts no-object with certainty.
        """
        logit_scale = 100.0
        pred_logits = torch.tensor(
            [
                [
                    [logit_scale, -logit_scale, -logit_scale],  # strongly class 0
                    [-logit_scale, logit_scale, -logit_scale],  # strongly class 1
                    [-logit_scale, -logit_scale, logit_scale],  # strongly no-object
                ]
            ]
        )  # (1, 3, 3)

        tgt_boxes = _make_boxes(
            (0.25, 0.25, 0.2, 0.2),
            (0.75, 0.75, 0.2, 0.2),
        )
        pred_boxes = torch.cat([tgt_boxes, _make_boxes((0.5, 0.5, 0.1, 0.1))], dim=0)
        pred_boxes = pred_boxes.unsqueeze(0)  # (1, 3, 4)

        targets = [
            {
                "labels": torch.tensor([0, 1], dtype=torch.long),
                "boxes": tgt_boxes,
            }
        ]
        return pred_logits, pred_boxes, targets

    def test_near_zero_loss_on_perfect_predictions(self):
        """All three loss components must be near zero for perfect predictions."""
        pred_logits, pred_boxes, targets = self._make_perfect_inputs()
        losses = set_prediction_loss(pred_logits, pred_boxes, targets)

        for name, value in losses.items():
            assert value.item() == pytest.approx(
                0.0, abs=1e-3
            ), f"{name} = {value.item():.6f}, expected near zero for perfect predictions."

    def test_positive_loss_on_random_predictions(self):
        """All three loss components must be strictly positive for random predictions."""
        torch.manual_seed(0)
        B, N, C = 2, 6, 3
        pred_logits = torch.randn(B, N, C + 1)
        pred_boxes = torch.rand(B, N, 4)
        pred_boxes[..., 2:] = pred_boxes[..., 2:] * 0.3 + 0.05

        targets = [
            {
                "labels": torch.tensor([0, 2], dtype=torch.long),
                "boxes": _make_boxes((0.2, 0.2, 0.15, 0.15), (0.7, 0.6, 0.2, 0.2)),
            },
            {
                "labels": torch.tensor([1], dtype=torch.long),
                "boxes": _make_boxes((0.5, 0.5, 0.3, 0.3)),
            },
        ]
        losses = set_prediction_loss(pred_logits, pred_boxes, targets)

        for name, value in losses.items():
            assert value.item() > 0.0, f"{name} = {value.item():.6f}, expected strictly positive for random inputs."

    def test_output_keys(self):
        """Return dict must contain exactly loss_ce, loss_bbox, and loss_giou."""
        pred_logits, pred_boxes, targets = self._make_perfect_inputs()
        losses = set_prediction_loss(pred_logits, pred_boxes, targets)
        assert set(losses.keys()) == {
            "loss_ce",
            "loss_bbox",
            "loss_giou",
        }, f"Unexpected loss keys: {set(losses.keys())}"

    def test_all_losses_are_scalar_tensors(self):
        """Each returned loss must be a zero-dimensional tensor."""
        pred_logits, pred_boxes, targets = self._make_perfect_inputs()
        losses = set_prediction_loss(pred_logits, pred_boxes, targets)
        for name, value in losses.items():
            assert value.ndim == 0, f"{name} has shape {value.shape}, expected scalar."

    def test_empty_targets_does_not_raise(self):
        """An image with no ground-truth objects must not raise and must return finite losses."""
        B, N, C = 1, 4, 2
        pred_logits = torch.randn(B, N, C + 1)
        pred_boxes = torch.rand(B, N, 4)
        pred_boxes[..., 2:] = 0.2
        targets = [{"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)}]

        losses = set_prediction_loss(pred_logits, pred_boxes, targets)
        for name, value in losses.items():
            assert torch.isfinite(value), f"{name} is not finite for empty targets."
