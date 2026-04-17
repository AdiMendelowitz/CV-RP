"""
Segmentation loss functions and evaluation metrics.

Dice loss: Milletari et al., "V-Net", 3DV 2016. https://arxiv.org/abs/1606.04797
BCE + Dice: Standard combination used in medical image segmentation practice.
IoU: Standard overlap metric. Not differentiable, used for evaluation only.

Calling conventions
-------------------
dice_loss: expects probabilities (post sigmoid) shape (B, 1, H, W).
iou_score: expects probabilities (post sigmoid) shape (B, 1, H, W).
combined_loss: expects raw logits, shape (B, 1, H, W). Applies sigmoid internally.
"""

import torch
import torch.nn.functional as F


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss averaged over the batch.

    Differentiable everywhere: the hard argmax is replaced by soft overlap so gradients flow through
    predicted probabilities. Handles class imbalance better than BCE alone because the denominator
    normalises by total predicted and true foreground mass, preventing large background from dominating.

    Args:
        pred: Predicted probabilities (after sigmoid), shape (B, 1, H, W).
        target: Binary ground truth mask, shape (B, 1, H, W).
        eps: Smoothing constant added to both numerator and denominator to handle empty masks without
             division by zero (pred and target both all-zero).

    Returns:
        Scalar Dice loss in [0, 1]. Lower is better.
    """
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = (pred_flat + target_flat).sum(dim=1)

    dice_per_sample = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice_per_sample.mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, esp: float = 1e-6) -> torch.Tensor:
    """IoU averaged over the batch.

    Hard metric: probabilities are thresholded to binary before evaluation. Not suitable as a training
    loss due to the threshold operation having zero gradient almost everywhere.

    Args:
        pred: Predicted probabilities (after sigmoid), shape (B, 1, H, W).
        target: Binary ground truth mask, shape (B, 1, H, W).
        threshold: Decision boundary for converting probabilities to binary masks.
        esp: Smoothing constant to avoid division by zero on empty masks.

    Returns:
        Detached scalar mean IoU in [0, 1]. Higher is better.
    """
    with torch.no_grad():
        pred_binary = (pred >= threshold).float()
        pred_flat = pred_binary.view(pred_binary.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

        iou_per_sample = (intersection + esp) / (union + esp)
        return iou_per_sample.mean()


def combined_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Weighted sum of binary cross-entropy and Dice loss: alpha * BCE + (1 - alpha) * Dice.

    BCE penalizes per-pixel confidence error and provides stable, well-scaled gradients early in
    training. Dice loss directly optimizes the overlap metric used at evaluation time and handles
    class imbalance. Combining both converges faster than either alone.

    BCE is computed via F.binary_cross_entropy_with_logits for numerical stability; sigmoid is
    applied internally before Dice.

    Args:
        pred: Raw logits, shape (B, 1, H, W).
        target: Binary ground truth mask, shape (B, 1, H, W).
        alpha: convex combination factor, default is 0.5.

    Returns:
        Scalar combined loss. Lower is better.
    """
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(torch.sigmoid(pred), target)
    return alpha * bce + (1 - alpha) * dice


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 4, 64, 64

    logits = torch.randn(B, 1, H, W)
    probs = torch.sigmoid(logits)
    target = torch.randint(0, 2, (B, 1, H, W)).float()

    d = dice_loss(probs, target)
    iou = iou_score(probs, target)
    loss = combined_loss(probs, target)

    assert d.shape == torch.Size([]), f"dice_loss must return a scalar, got {d.shape}"
    assert iou.shape == torch.Size([]), f"iou_score must return a scalar, got {iou.shape}"
    assert loss.shape == torch.Size([]), f"combined_loss must return a scalar, got {loss.shape}"
    assert 0.0 <= d.item() <= 1.0, f"dice_loss out of range: {d.item()}"
    assert 0.0 <= iou.item() <= 1.0, f"iou_score out of range: {iou.item()}"

    # iou_score mustn't carry gradient, it's an evaluation metric only
    probs_grad = torch.sigmoid(torch.randn(B, 1, H, W, requires_grad=True))
    iou_grad = iou_score(probs_grad, target)
    assert not iou_grad.requires_grad, "iou_score must return a non-differentiable value for evaluation only"

    # combined_loss must be differentiable w.r.t. raw logits
    logits_grad = torch.randn(B, 1, H, W, requires_grad=True)
    combined_loss(logits_grad, target).backward()
    assert logits_grad is not None, "combined_loss must produce gradients"

    print(f"dice_loss:     {d.item():.4f}")
    print(f"iou_score:     {iou.item():.4f}")
    print(f"combined_loss: {loss.item():.4f}")
    print("\nAll assertions passed.")
