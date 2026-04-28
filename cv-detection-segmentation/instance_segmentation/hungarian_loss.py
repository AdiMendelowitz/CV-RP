"""
DETR set prediction loss with hungarian matching.

Implements the bipartite matching loss from:
    "End-to-End Object Detection with Transformers"
    Carion et al., ECCV 2020. https://arxiv.org/abs/2005.12872

All bounding boxes use normalised [x_center, y_center, width, height] format, with coordinates in [0, 1]
relative to image dimensions.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def compute_giou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Generalized IoU (GIoU) between two sets of boxes.

    GIoU extends IoU by adding an enclosing-box penalty that provides a non-zero gradient, even when the boxes don't
    overlap, as defined in Rezatofighi et al., CVPR 2019. https://arxiv.org/abs/1902.09630

    Args:
        boxes_a: Boxes in cx, cy, w, h format, shape (N, 4).
        boxes_b: Boxes in cx, cy, w, h format, shape (M, 4).

    Returns:
        Pairwise GIoU matrix of shape (N, M), values in [-1, 1].
    """

    # Convert cx, cy, w, h -> x1, y1, x2, y2, then broadcast to (N,M)
    a_x1 = (boxes_a[:, 0] - boxes_a[:, 2] / 2).unsqueeze(1)  # (N, 1)
    a_y1 = (boxes_a[:, 1] - boxes_a[:, 3] / 2).unsqueeze(1)
    a_x2 = (boxes_a[:, 0] + boxes_a[:, 2] / 2).unsqueeze(1)
    a_y2 = (boxes_a[:, 1] + boxes_a[:, 3] / 2).unsqueeze(1)

    b_x1 = (boxes_b[:, 0] - boxes_b[:, 2] / 2).unsqueeze(0)  # (1, M)
    b_y1 = (boxes_b[:, 1] - boxes_b[:, 3] / 2).unsqueeze(0)
    b_x2 = (boxes_b[:, 0] + boxes_b[:, 2] / 2).unsqueeze(0)
    b_y2 = (boxes_b[:, 1] + boxes_b[:, 3] / 2).unsqueeze(0)

    area_a = ((a_x2 - a_x1) * (a_y2 - a_y1)).clamp(min=0) # (N, 1)
    area_b = ((b_x2 - b_x1) * (b_y2 - b_y1)).clamp(min=0) # (1, M)

    inter_x1 = torch.max(a_x1, b_x1)
    inter_y1 = torch.max(a_y1, b_y1)
    inter_x2 = torch.min(a_x2, b_x2)
    inter_y2 = torch.min(a_y2, b_y2)
    inter_area = ((inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)) # (N, M)

    union_area = area_a + area_b - inter_area
    iou = inter_area / union_area.clamp(min=1e-7)

    enclose_x1 = torch.min(a_x1, b_x1)
    enclose_y1 = torch.min(a_y1, b_y1)
    enclose_x2 = torch.max(a_x2, b_x2)
    enclose_y2 = torch.max(a_y2, b_y2)
    area_enclose = ((enclose_x2 - enclose_x1).clamp(min=0) * (enclose_y2 - enclose_y1).clamp(min=0))

    giou = iou - (area_enclose - union_area) / area_enclose.clamp(min=1e-7)
    return giou

def build_cost_matrix(pred_logit: torch.Tensor, pred_boxes: torch.Tensor, target_labels: torch.Tensor,
                      target_boxes: torch.Tensor, cost_class: float = 1.0, cost_bbox: float = 5.0,
                      cost_giou: float = 2.0) -> torch.Tensor:
    """
    Construct the NxM assignment cost matrix for a single image.

    Each C[i, j] entry is the cost of assigning prediction i to ground-truth j. The cost combines a classification
    term, an L1 box term and a GIoU box term, matching Carion et al. (2020) equation 2.

    Classification cost uses softmax probability, not log-probability, to keep it bounded and consistent with the GIoU
    and L1 terms in scale.

    Args:
        pred_logit: Predicted class logits for one image, shape (N, num_classes + 1).
                    Last class index is the no-object class, and is excluded from the classification cost.
        pred_boxes: Prediction boxes in cx, cy, w, h format, shape (N, 4).
        target_labels: Ground-truth class indices, shape (M, ).
        target_boxes: Ground-truth boxes in cx, cy, w, h format, shape (M, 4).
        cost_class: Weight for the classification cost term.
        cost_bbox: Weight for the L1 bounding box cost term.
        cost_giou: Weight for the GIoU cost term.

    Returns:
        Cost matrix (N, M).
    """

    # Classification cost: negative softmax probability of the target class.
    # Shape: (N, num_classes + 1) -> select columns for target labels -> (N, M)
    probs = pred_logit.softmax(dim=-1)
    cost_cls = -probs[:, target_labels]

    # L1 box cost: pairwise L1 distance between predicted and target boxes.
    # torch.cdist with p=1 give L1 norm
    cost_l1 = torch.cdist(pred_boxes, target_boxes, p=1) # (N, M)

    cost_giou_matrix = 1.0 - compute_giou(pred_boxes, target_boxes) # (N, M)

    cost_matrix = (cost_class * cost_cls) + (cost_bbox * cost_l1) + (cost_giou * cost_giou_matrix)

    return cost_matrix

def hungarian_match(cost_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find the minimum-cost one-to-one assignment via the hungarian algorithm.

    Args:
        cost_matrix: Shape (N, M). N = number of predictions, M = number of ground-truth objects.
                     N >= M for a valid one-to-one assignment.

    Returns:
        Tuple (pred_indices, tgt_indices), each a longTensor of length M.
        pred_indices[k] is assigned to tgt_indices[k].
    """

    # Cost matrix moved to CPU and converted to numpy for scipy linear_sum_assignment, which only accepts CPU arrays.
    row_idx, col_idx = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    pred_indices = torch.as_tensor(row_idx, dtype=torch.long)
    target_indices = torch.as_tensor(col_idx, dtype=torch.long)
    return pred_indices, target_indices

def compute_giou_paired(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute element-wise GIoU for K aligned box pairs, shape (K,).

    Args:
        boxes_a: Boxes in cx, cy, w, h format, shape (K, 4).
        boxes_b: Boxes in cx, cy, w, h format, shape (K, 4).

    Returns:
        GIoU values of shape (K,), values in [-1, 1].
    """
    a_x1 = boxes_a[:, 0] - boxes_a[:, 2] / 2
    a_y1 = boxes_a[:, 1] - boxes_a[:, 3] / 2
    a_x2 = boxes_a[:, 0] + boxes_a[:, 2] / 2
    a_y2 = boxes_a[:, 1] + boxes_a[:, 3] / 2

    b_x1 = boxes_b[:, 0] - boxes_b[:, 2] / 2
    b_y1 = boxes_b[:, 1] - boxes_b[:, 3] / 2
    b_x2 = boxes_b[:, 0] + boxes_b[:, 2] / 2
    b_y2 = boxes_b[:, 1] + boxes_b[:, 3] / 2

    area_a = ((a_x2 - a_x1) * (a_y2 - a_y1)).clamp(min=0)
    area_b = ((b_x2 - b_x1) * (b_y2 - b_y1)).clamp(min=0)

    inter_area = (
            (torch.min(a_x2, b_x2) - torch.max(a_x1, b_x1)).clamp(min=0) *
            (torch.min(a_y2, b_y2) - torch.max(a_y1, b_y1).clamp(min=0))
    )

    area_enclose = (
        (torch.max(a_x2, b_x2) - torch.min(a_x1, b_x1)).clamp(min=0) *
        (torch.max(a_y2, b_y2) - torch.min(a_y1, b_y1).clamp(min=0))
    )

    union_area = area_a + area_b - inter_area
    iou = inter_area / union_area.clamp(min=1e-7)

    return iou - (area_enclose-union_area) / area_enclose.clamp(min=1e-7)

def set_prediction_loss(pred_logits: torch.Tensor, pred_boxes: torch.Tensor, targets: list[dict[str, torch.Tensor]],
                        cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0,
                        eos_coef: float = 0.1) -> dict[str, torch.Tensor]:
    """
    Compute the full DETR set prediction loss for a batch of images.

    For each image, the Hungarian algorithm finds the optimal 1-to-1 assignment prediction <-> ground-truth objects.
    The loss is computed over the assignments, combining a classification term (cross-entropy over all N predictions)
    and box regression terms (L1 and GIoU, computed only on matched pairs).

    Box losses are normalised by the total number of matched objects across the batch, to make the loss scale invariant
    to the batch size and image density. The no-object class is down-weighted by eos_coef to counteract the large class
    imbalance between matched and unmatched predictions.

    Args:
        pred_logits: Batch of predicted class logits, shape (B, N, num_classes + 1).
        pred_boxes: Batch of predicted boxes in cx, cy, w, h format, shape (B, N, 4).
        targets: List of B dicts, one per image, each containing:
                 - "labels": LongTensor of shape (M,), with ground-truth class indices.
                 - "boxes": FloatTensor of shape (M, 4), with ground-truth boxes in cx, cy, w, h normalised format.
        cost_class: Weight for the classification cost term in the matching step.
        cost_bbox: Weight for the L1 cost term in the matching step and the L1 loss term.
        cost_giou: Weight for the GIoU cost term in the matching step and the GIoU loss term.
        eos_coef: Weight applied to the no-object class in the classification loss.
                 Typical value is 0.1, as used in the original DETR paper.

    Returns:
        A dict with scalar loss tensors:
            - "loss_ce": Weighted cross-entropy loss over all predictions.
            - "loss_bbox": L1 loss on matched box pairs, normalised by num_boxes.
            - "loss_giou": GIoU loss on matched box pairs, normalised by num_boxes.
    """

    batch_size, num_queries, num_classes_plus_one = pred_logits.shape
    num_classes = num_classes_plus_one - 1
    device = pred_logits.device

    # Per-images Hungarian matching
    target_classes = torch.full((batch_size, num_queries), fill_value=num_classes, dtype=torch.long, device=device)

    matched_pred_boxes: list[torch.Tensor] = []
    matched_target_boxes: list[torch.Tensor] = []

    for b_s in range(batch_size):
        target_labels = targets[b_s]["labels"].to(device) # (M,)
        target_boxes = targets[b_s]["boxes"].to(device)

        if target_labels.numel() == 0: # No ground-truth objects in this image, all predictions are no-objects
            continue

        cost_matrix = build_cost_matrix(pred_logits[b_s], pred_boxes[b_s], target_labels, target_boxes,
                                        cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)

        pred_idx, target_idx = hungarian_match(cost_matrix)
        target_classes[b_s, pred_idx] = target_labels[target_idx]
        matched_pred_boxes.append(pred_boxes[b_s][pred_idx])
        matched_target_boxes.append(target_boxes[target_idx])

    # Classification loss: cross-entropy over B*N predictions.
    cls_weights = torch.ones(num_classes_plus_one, device=device)
    cls_weights[num_classes] = eos_coef
    loss_ce = F.cross_entropy(
        pred_logits.flatten(0, 1),   # (B*N, num_classes + 1)
        target_classes.flatten(0, 1),   # (B*N,)
        weight=cls_weights,
    )

    # Box losses: L1 and GIoU, only on matched prediction-target pairs.
    num_boxes = max(sum(t["labels"].numel() for t in targets), 1)

    if matched_pred_boxes:
        all_pred_boxes = torch.cat(matched_pred_boxes, dim=0)   # (total_matched, 4)
        all_target_boxes = torch.cat(matched_target_boxes, dim=0)   # (total_matched, 4)

        loss_bbox = F.l1_loss(all_pred_boxes, all_target_boxes, reduction="sum") / num_boxes

        loss_giou = (1.0 - compute_giou_paired(all_pred_boxes, all_target_boxes)).sum() / num_boxes
    else:
        zero = torch.tensor(0.0, device=device)
        loss_bbox, loss_giou = zero, zero

    return {"loss_ce": loss_ce, "loss_bbox": loss_bbox, "loss_giou": loss_giou}





