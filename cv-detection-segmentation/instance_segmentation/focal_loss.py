"""
Alpha-balanced multi-class focal loss.

Reference: Lin et al. "Focal loss for Dense Object Detection." ICCV 2017.  arXiv:1708.02002.
"""

import torch
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """
    Compute alpha-balanced multi-clas focal loss.
    When gamma=0 this reduces exactly to weighted cross-entropy with the same alpha weights, matching
    nn.CrossEntropyLoss(weight=alpha).

    Args:
        logits: Raw model outputs, shape (N, C). Not softmaxed.
        targets:  Ground-truth class indices, shape (N,). dtype=torch.long.
        alpha: Per-class weights, shape (C,). Must be on the same device as logits.
               Use inverse class frequency to match weighted CE: alpha_c = total / (num_classes * count_c).
        gamma: Focusing parameter. gamma=0 recovers weighted CE.

    Returns:
        Scalar mean focal loss over the batch.
    """

    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (N, C), got shape {tuple(logits.shape)}")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError(f"targets must be 1-D with length N={logits.shape[0]}, got shape {tuple(targets.shape)}")
    if alpha.shape[0] != logits.shape[1]:
        raise ValueError(f"alpha length {alpha.shape[0]} does not match logits length {logits.shape[1]}")
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")

    # log_softmax is numerically stable, avoids log(softmax(x)) cancellation
    log_p = F.log_softmax(logits, dim=1)  # (N, C)
    log_pt = log_p[torch.arange(len(targets)), targets]  # (N,)

    # Per-sample alpha weight from the true class
    alpha_t = alpha[targets]  # (N,)

    if gamma == 0.0:
        return -(alpha_t * log_pt).sum() / alpha_t.sum()

    p_t = log_pt.exp()  # Safe because log_pt<=0 always. (N,) in [0,1]
    focal_weight = (1.0 - p_t) ** gamma  # (N,)

    return -(alpha_t * focal_weight * log_pt).sum() / alpha_t.sum()
