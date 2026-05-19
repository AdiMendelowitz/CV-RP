"""
Object detection metrics: IoU, NMS, AP, mAP

NumPy only implementation. Box format: [x1, y1, x2, y2] where x1<x2, y1<y2

References:
    Everingham et al. (2010) "The PASCAL Visual Object Classes Challenge"
    https://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf
"""

import numpy as np


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU in [0,1]. 0.0 == union is zero or boxes do not overlap.
    """

    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    Greedy NMS: suppress lower-scoring boxes whose IoU with a kept box exceeds iou_threshold.

    Args:
        boxes: array [x1, y1, x2, y2].
        scores:  array (N,) of confidence scores.
        iou_threshold:  Boxes with IoU > iou_threshold are suppressed.

    Returns:
        Array (K,) of indices of kept boxes, sorted by descending score.
    """

    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    candidates = np.argsort(scores)[::-1]
    kept: list[int] = []

    while candidates.size > 0:
        idx = candidates[0]
        kept.append(int(idx))

        ious = np.array([compute_iou(boxes[idx], boxes[other]) for other in candidates[1:]])
        candidates = candidates[1:][ious <= iou_threshold]

    return np.array(kept, dtype=np.int64)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Average precision via 11-point interpolation (PASCAL VOC 2007 protocol).

    For each recall threshold in [0.0, 0.1, ..., 1.0] take the max precision at recall >= that threshold, then average
    over all 11 thresholds.

    Args:
        recall: array (N,) of recall values.
        precision: array (N,) of precision values.

    Returns:
        AP in [0,1].
    """
    ap = 0.0
    for threshold in np.linspace(0.0, 1.0, 11):
        mask = recall >= threshold
        ap += precision[mask].max() if mask.any() else 0.0

    return float(ap / 11.0)


def compute_map(predictions: dict[int, dict], targets: dict[int, dict], iou_threshold: float = 0.5) -> float:
    """
    Mean Average Precision across all classes present in targets.

    Prediction dictionary keys:
        boxes: np.ndarray shape (N,4) in [x1, y1, x2, y2] format.
        scores: np.ndarray shape (N,) of confidence scores.
        labels: np.ndarray shape (N,) of class indices.

    Target dictionary keys:
        boxes: np.ndarray shape (N,4) in [x1, y1, x2, y2] format.
        labels: np.ndarray shape (N,) of class indices.

    Args:
        predictions: dict mapping image_id to prediction dict.
        targets: dict mapping image_id to target dict.
        iou_threshold: IoU threshold for a detection to count as a true positive.

    Returns:
        mAP in [0,1]. 0.0 == no classes are present in targets.
    """
    class_ids: set[int] = set()
    for t in targets.values():
        class_ids.update(t["labels"].tolist())

    if not class_ids:
        return 0.0

    aps: list[float] = []
    for class_id in sorted(class_ids):
        all_scores: list[float] = []
        all_tp: list[int] = []
        all_fp: list[int] = []
        n_gt = 0

        for image_id, target in targets.items():
            gt_mask = target["labels"] == class_id
            gt_boxes = target["boxes"][gt_mask]
            n_gt += len(gt_boxes)

            pred = predictions.get(image_id)
            if pred is None or len(pred["labels"]) == 0:
                continue

            pred_mask = pred["labels"] == class_id
            if not pred_mask.any():
                continue

            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]
            matched = np.zeros(len(gt_boxes), dtype=bool)

            for i in np.argsort(pred_scores)[::-1]:
                all_scores.append(float(pred_scores[i]))

                if len(gt_boxes) == 0:
                    all_tp.append(0)
                    all_fp.append(1)
                    continue

                ious = np.array([compute_iou(pred_boxes[i], gt_box) for gt_box in gt_boxes])
                best = int(np.argmax(ious))

                if ious[best] >= iou_threshold and not matched[best]:
                    matched[best] = True
                    all_tp.append(1)
                    all_fp.append(0)
                else:
                    all_fp.append(1)
                    all_tp.append(0)

        if n_gt == 0:
            continue

        if not all_scores:
            aps.append(0.0)
            continue

        global_order = np.argsort(all_scores)[::-1]
        tp_cumulative = np.cumsum(np.array(all_tp)[global_order])
        fp_cumulative = np.cumsum(np.array(all_fp)[global_order])

        recall = tp_cumulative / n_gt
        precision = tp_cumulative / (tp_cumulative + fp_cumulative)

        aps.append(compute_ap(recall, precision))

    return float(np.mean(aps)) if aps else 0.0
