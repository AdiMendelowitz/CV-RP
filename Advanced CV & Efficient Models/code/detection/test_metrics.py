"""
Unit tests for code/detection/metrics.py.

All expected values are derived analytically from known inputs so that
correctness can be verified without reference to any external library.
"""

import numpy as np
import pytest

from metrics import compute_ap, compute_iou, compute_map, non_max_suppression

# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------


class TestComputeIou:
    def test_identical_boxes(self) -> None:
        box = np.array([0.0, 0.0, 10.0, 10.0])
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        box1 = np.array([0.0, 0.0, 5.0, 5.0])
        box2 = np.array([6.0, 6.0, 10.0, 10.0])
        assert compute_iou(box1, box2) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        # box1: [0,0,4,4] area=16, box2: [2,2,6,6] area=16
        # intersection: [2,2,4,4] area=4, union=28
        box1 = np.array([0.0, 0.0, 4.0, 4.0])
        box2 = np.array([2.0, 2.0, 6.0, 6.0])
        assert compute_iou(box1, box2) == pytest.approx(4.0 / 28.0)

    def test_containment(self) -> None:
        # inner area=4, outer area=100, union=100, intersection=4
        outer = np.array([0.0, 0.0, 10.0, 10.0])
        inner = np.array([4.0, 4.0, 6.0, 6.0])
        assert compute_iou(outer, inner) == pytest.approx(4.0 / 100.0)

    def test_zero_area_box(self) -> None:
        # Degenerate box with zero width -- inter_area=0, IoU=0.
        box1 = np.array([2.0, 2.0, 2.0, 5.0])
        box2 = np.array([0.0, 0.0, 4.0, 4.0])
        assert compute_iou(box1, box2) == pytest.approx(0.0)

    def test_touching_edges(self) -> None:
        # Boxes share an edge: intersection width=0, IoU=0.
        box1 = np.array([0.0, 0.0, 5.0, 5.0])
        box2 = np.array([5.0, 0.0, 10.0, 5.0])
        assert compute_iou(box1, box2) == pytest.approx(0.0)

    def test_symmetry(self) -> None:
        box1 = np.array([1.0, 2.0, 5.0, 6.0])
        box2 = np.array([3.0, 4.0, 8.0, 9.0])
        assert compute_iou(box1, box2) == pytest.approx(compute_iou(box2, box1))


# ---------------------------------------------------------------------------
# non_max_suppression
# ---------------------------------------------------------------------------


class TestNonMaxSuppression:
    def test_empty_input(self) -> None:
        kept = non_max_suppression(np.zeros((0, 4)), np.zeros(0), iou_threshold=0.5)
        assert len(kept) == 0

    def test_single_box(self) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        scores = np.array([0.9])
        kept = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert list(kept) == [0]

    def test_suppresses_overlapping_lower_score(self) -> None:
        # Two heavily overlapping boxes -- lower score must be suppressed.
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 11.0, 11.0],
            ]
        )
        scores = np.array([0.9, 0.75])
        kept = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert list(kept) == [0]

    def test_keeps_non_overlapping_boxes(self) -> None:
        # Three boxes, none overlapping -- all must be kept.
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 0.0, 30.0, 10.0],
                [40.0, 0.0, 50.0, 10.0],
            ]
        )
        scores = np.array([0.9, 0.8, 0.7])
        kept = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert set(kept) == {0, 1, 2}

    def test_threshold_boundary(self) -> None:
        # IoU exactly at threshold must NOT suppress (rule: IoU > threshold suppresses).
        box1 = np.array([0.0, 0.0, 4.0, 4.0])
        box2 = np.array([2.0, 2.0, 6.0, 6.0])
        iou = compute_iou(box1, box2)  # 4/28 ~ 0.143
        boxes = np.stack([box1, box2])
        scores = np.array([0.9, 0.8])
        kept = non_max_suppression(boxes, scores, iou_threshold=iou)
        assert set(kept) == {0, 1}

    def test_score_ordering_respected(self) -> None:
        # Higher-score box must be kept regardless of array position.
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.5, 0.5, 10.5, 10.5],
            ]
        )
        scores = np.array([0.6, 0.9])  # second box has higher score
        kept = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert list(kept) == [1]

    def test_chain_suppression(self) -> None:
        # Box A overlaps B; B overlaps C; A does not overlap C.
        # After keeping A and suppressing B, C must survive because it only
        # overlapped B (which is already gone from candidates).
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],  # A -- highest score, kept
                [5.0, 0.0, 15.0, 10.0],  # B -- overlaps A (iou>0.5), suppressed
                [12.0, 0.0, 22.0, 10.0],  # C -- overlaps B but not A, kept
            ]
        )
        scores = np.array([0.9, 0.8, 0.7])
        kept = non_max_suppression(boxes, scores, iou_threshold=0.3)
        # Verify A is kept, B is suppressed, C survives.
        assert 0 in kept
        assert 1 not in kept
        assert 2 in kept


# ---------------------------------------------------------------------------
# compute_ap
# ---------------------------------------------------------------------------


class TestComputeAp:
    def test_perfect_precision(self) -> None:
        # Precision=1 at all recall levels gives AP=1.
        recall = np.linspace(0.0, 1.0, 11)
        precision = np.ones(11)
        assert compute_ap(recall, precision) == pytest.approx(1.0)

    def test_zero_precision(self) -> None:
        recall = np.linspace(0.0, 1.0, 11)
        precision = np.zeros(11)
        assert compute_ap(recall, precision) == pytest.approx(0.0)

    def test_known_staircase(self) -> None:
        # Precision=1 for recall<=0.5, precision=0 for recall>0.5.
        # 11-point: thresholds 0..0.5 (6 points) get p=1, rest get p=0.
        # AP = 6/11
        recall = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        precision = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert compute_ap(recall, precision) == pytest.approx(6.0 / 11.0)

    def test_single_point(self) -> None:
        # Single point at recall=1.0, precision=0.5.
        # All 11 thresholds have recall >= threshold, so AP = 0.5.
        recall = np.array([1.0])
        precision = np.array([0.5])
        assert compute_ap(recall, precision) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_map
# ---------------------------------------------------------------------------


class TestComputeMap:
    @staticmethod
    def _make_pred(boxes: list[list[float]], scores: list[float], labels: list[int]) -> dict:
        return {
            "boxes": np.array(boxes, dtype=float),
            "scores": np.array(scores, dtype=float),
            "labels": np.array(labels, dtype=int),
        }

    @staticmethod
    def _make_target(boxes: list[list[float]], labels: list[int]) -> dict:
        return {
            "boxes": np.array(boxes, dtype=float),
            "labels": np.array(labels, dtype=int),
        }

    def test_perfect_single_class(self) -> None:
        # One image, one GT box, one correct high-IoU prediction.
        targets = {0: self._make_target([[0.0, 0.0, 10.0, 10.0]], [0])}
        predictions = {0: self._make_pred([[0.5, 0.5, 10.5, 10.5]], [0.9], [0])}
        assert compute_map(predictions, targets, iou_threshold=0.5) == pytest.approx(1.0)

    def test_no_predictions(self) -> None:
        targets = {0: self._make_target([[0.0, 0.0, 10.0, 10.0]], [0])}
        assert compute_map({}, targets, iou_threshold=0.5) == pytest.approx(0.0)

    def test_no_targets(self) -> None:
        predictions = {0: self._make_pred([[0.0, 0.0, 10.0, 10.0]], [0.9], [0])}
        assert compute_map(predictions, {}, iou_threshold=0.5) == pytest.approx(0.0)

    def test_low_iou_prediction_is_fp(self) -> None:
        # IoU ~ 0.02, well below threshold=0.5 -> FP, AP=0.
        targets = {0: self._make_target([[0.0, 0.0, 10.0, 10.0]], [0])}
        predictions = {0: self._make_pred([[8.0, 8.0, 18.0, 18.0]], [0.9], [0])}
        assert compute_map(predictions, targets, iou_threshold=0.5) == pytest.approx(0.0)

    def test_two_classes_averaged(self) -> None:
        # Class 0: perfect prediction (AP=1). Class 1: no prediction (AP=0).
        # mAP = 0.5.
        targets = {
            0: self._make_target(
                [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
                [0, 1],
            ),
        }
        predictions = {0: self._make_pred([[0.5, 0.5, 10.5, 10.5]], [0.9], [0])}
        assert compute_map(predictions, targets, iou_threshold=0.5) == pytest.approx(0.5)

    def test_duplicate_detection_counts_once(self) -> None:
        # Two predictions both matching GT box 0; GT box 1 never detected.
        # n_gt=2, tp=[1,0], fp=[0,1], recall=[0.5,0.5], precision=[1.0,0.5]
        # 11-point AP: thresholds 0.0-0.5 (6 pts) -> max_p=1.0; 0.6-1.0 (5 pts) -> 0.0
        # AP = 6/11
        targets = {
            0: self._make_target(
                [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
                [0, 0],
            ),
        }
        predictions = {
            0: self._make_pred(
                [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]],
                [0.9, 0.8],
                [0, 0],
            ),
        }
        assert compute_map(predictions, targets, iou_threshold=0.5) == pytest.approx(6.0 / 11.0)

    def test_multi_image_global_ranking(self) -> None:
        # Two images: Image 0 has a low-confidence correct detection, Image 1 has a high-confidence incorrect detection.
        # Global sort puts the FP first, which should lower the PR curve compared to ranking TPs first.
        #
        # Global order by score: FP(0.95) -> TP(0.80)
        # After FP: tp_cum=0, fp_cum=1, recall=0.0, precision=0.0
        # After TP: tp_cum=1, fp_cum=1, recall=1.0, precision=0.5
        # 11-point AP: thresholds 0.0-0.5 (6 pts) get max_p=0.5; 0.6-1.0 (5 pts) get 0.0
        # AP = 6*0.5/11 = 3/11
        targets = {
            0: self._make_target([[0.0, 0.0, 10.0, 10.0]], [0]),
            1: self._make_target([[50.0, 50.0, 60.0, 60.0]], [0]),
        }
        predictions = {
            0: self._make_pred([[0.0, 0.0, 10.0, 10.0]], [0.80], [0]),  # TP
            1: self._make_pred([[0.0, 0.0, 10.0, 10.0]], [0.95], [0]),  # FP (wrong location)
        }
        assert compute_map(predictions, targets, iou_threshold=0.5) == pytest.approx(3.0 / 11.0)
