"""Projected Gradient Descent (PGD).

Madry, Makelov, Schmidt, Tsipras, Vladu. "Towards Deep Learning Models Resistant
to Adversarial Attacks." ICLR 2018. https://arxiv.org/abs/1706.06083

Threat model: L-infinity perturbations on images scaled to [0, 1]. Models trained on normalised inputs must carry the
normalisation as an internal first later so that the attack operates in pixel space.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

def pgd(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, epsilon: float, alpha: float,
        steps: int, random_start: bool = True, loss_fn: LossFn = F.cross_entropy,
        targeted: bool = False) -> torch.Tensor:
    """
    Craft PGD adversarial examples.

    Iterative FGSM-style sign steps of size "alpha", projecting back into the L-infinity "epsilon"-ball and then into
    the valid pixel range after every step. "random_start" initializes the iteration within the epsilon-ball, and
    without it PGD reduces to iterated FGSM from the clean point.

    The model must be in eval mode: with BatchNorm is in train mode, the batch statistics couple samples together and
    the repeated forward passes overwrite the running statistics with adversarial batches. Adversarial training loops
    must therefore switch the model to *eval mode* for the attack and back to train mode for the parameter update.
    The model itself is left unchanged, and no parameter gradients are populated.

    Gradients are computed under an explicit torch.enable_grad context so the call is safe inside a torch.no_grad
    evaluation loop. It isn't sade inside torch.inference_mode whic cannot be enbaled from within.

    Randomness is drawn from the global torch RNG, so seefing beforehand fixes the initialization.

    Args:
        model: Classifier mapping images to logits.
        images: Clean inputs in [0, 1], shape (N, C, H, W).
        labels: True classes (untargeted) or target classes (targeted), shape (N,).
        epsilon: L-infinity perturbation budget, must be non-negative.
        alpha: Step size per iteration, must be >= 0.
        steps: Number of iterations, must be >= 1.
        random_start: True: initialize uniformly within the epsilon-ball.
        loss_fn: Loss taking (logits, labels) and reducing to scalar
        targeted:

    Returns:

    """
    return