"""
Fast Gradient Sign Method (FGSM).

Goodfellow, Shlens, Szegedy. "Explaining and Harnessing Adversarial Examples."
ICLR 2015. https://arxiv.org/abs/1412.6572

Threat model: L-infinity perturbations on images scaled to [0,1]. Models trained on normalised inputs must carry the
normalisation as an internal first later so that the attack operate in pixel space
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

def fgsm(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, epsilon: float,
         loss_fn: LossFn = F.cross_entropy, targeted: bool = False) -> torch.Tensor:
    """
    Craft FGSM advarserial examples.
    Takes on gradient step of size "epsilon" along the sign of the input gradient.
    In untargeted mode "labels" are the true labels and the step increases the loss.
    In targeted mode "labels" are the desired target classes and the step decreases the loss towards them.

    The model must be in eval mode: with BatchNorm in train mode the batch statistics couple samples together, which
    weakens the attack and makes it depend on batch composition. The model itself is left unchanged, and no paramater
    gradients are populated.

    Gradients are computed under an explicit torch.enable_grad context, so the call is safe inside a torch.no_grad
    evaluation loop. It isn't safe inside torch.inference_mode, which cannot be re-enabled from within.

    Args:
         model: Classifier mapping images to logits.
         images: Clean inputs in [0,1], shape (N, C, H, W).
         labels: True classes (untargeted) or target classes (targets), shape (N,).
         epsilon: L-infinity perturbation budget, must be >= 0.
         loss_fn: Loss taking (logits, labels) and reducing to a scalar, defaults to cross-entropy. The reduction
                  doesn't affect the result, since the sign of the gradient is invariant to positive scaling.
         targeted: If true step towards "labels".

    Returns:
         Adversarial images in [0, 1], same shape and dtype as "images", detached.

    Raises:
        ValueError: if "epsilon" is negative.
    """

    if epsilon < 0:
        raise ValueError(f"Epsilon must be >= 0, got {epsilon}")

    clean = images.detach()
    with torch.enable_grad():
        adv = clean.clone().requires_grad_(True)
        loss = loss_fn(model(adv), labels)
        grad = torch.autograd.grad(loss, adv)[0]

    step = -epsilon if targeted else epsilon
    adv = clean + step * grad.sign()

    return adv.clamp(0.0, 1.0).detach
