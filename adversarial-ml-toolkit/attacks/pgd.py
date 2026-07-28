"""Projected Gradient Descent (PGD).

Madry, Makelov, Schmidt, Tsipras, Vladu. "Towards Deep Learning Models Resistant
to Adversarial Attacks." ICLR 2018. https://arxiv.org/abs/1706.06083

Threat model: L-infinity perturbations on images scaled to [0, 1]. Models trained on normalised inputs must carry the
normalisation as an internal first layer so that the attack operates in pixel space.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def pgd(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, eps: float, alpha: float, steps: int,
        random_start: bool = True, loss_fn: LossFn = F.cross_entropy, targeted: bool = False) -> torch.Tensor:
    """Craft PGD adversarial examples.

    Iterative FGSM-style sign steps of size alpha, projecting back into the L-infinity eps-ball and then into the valid
    pixel range after every step. random_start initialises the iteration within the eps-ball; without it, PGD reduces to
    iterated FGSM from the clean point.

    The model must be in eval mode: with BatchNorm in train mode the batch statistics couple samples together and the
    repeated forward passes overwrite the running statistics with adversarial batches. Adversarial training loops must
    therefore switch the model to eval mode for the attack and back to train mode for the parameter update. The model
    itself is left unchanged and no parameter gradients are populated.

    Gradients are computed under an explicit torch.enable_grad context so the call is safe inside a torch.no_grad
    evaluation loop. It is not safe inside torch.inference_mode, which cannot be exited from within.

    Randomness is drawn from the global torch RNG, so seeding beforehand fixes the initialisation.

    Args:
        model: Classifier mapping images to logits.
        images: Clean inputs in [0, 1], shape (N, C, H, W).
        labels: True classes (untargeted) or target classes (targeted), shape (N,).
        eps: L-infinity perturbation budget, must be non-negative.
        alpha: Step size per iteration, must be > 0.
        steps: Number of iterations, must be >= 1.
        random_start: If true, initialise uniformly within the eps-ball.
        loss_fn: Takes (logits, labels) and reduces to a scalar. Reduction does not affect the result since the
                 gradient's sign is invariant to positive scaling. Defaults to cross-entropy.
        targeted: If true, step towards labels instead of away from them.

    Returns:
        Adversarial images in [0, 1], same shape and dtype as images, detached.

    Raises:
        ValueError: If eps is negative, alpha is not positive, or steps < 1.
    """
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    clean = images.detach()
    if random_start:
        adv = (clean + torch.empty_like(clean).uniform_(-eps, eps)).clamp(0.0, 1.0)
    else:
        adv = clean.clone()

    step = -alpha if targeted else alpha
    with torch.enable_grad():
        for _ in range(steps):
            adv = adv.detach().requires_grad_(True)
            loss = loss_fn(model(adv), labels)
            grad = torch.autograd.grad(loss, adv)[0]

            # Detach before the update, otherwise the graph chains across iterations.
            adv = adv.detach() + step * grad.sign()
            delta = (adv - clean).clamp(-eps, eps)
            adv = (clean + delta).clamp(0.0, 1.0)

    return adv.detach()