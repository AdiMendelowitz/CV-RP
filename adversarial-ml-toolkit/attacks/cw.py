"""Carlini and Wagner L2 attack (C&W).

Carlini, Wagner. "Towards Evaluating the Robustness of Neural Networks."
IEEE S&P 2017. https://arxiv.org/abs/1608.04644

Threat model: minimum-L2 perturbations on images scaled to [0, 1]. This is the untargeted L2 variant with the f6 margin
objective from the paper, at a fixed penalty c rather than the paper's binary search. Models trained on normalised
inputs must carry the normalisation as an internal first layer so that the attack operates in pixel space.
"""

import torch
import torch.nn.functional as F
from torch import nn


def cw(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, c: float = 1.0, kappa: float = 0.0,
       steps: int = 1000, lr: float = 0.01) -> torch.Tensor:
    """Craft untargeted C&W L2 adversarial examples.

    Optimises a tanh-space perturbation to minimise ||adv - clean||_2^2 + c * f(adv) -- where f is the margin between
    the true-class logit and the best other logit, floored at -kappa. The tanh reparametrisation keeps every iterate in
    [0, 1] by construction, so no clipping is needed. For each sample the lowest-distortion perturbation that actually
    flips the prediction is returned; samples that never flip are returned unchanged. The attack is deterministic given
    the model and input, so no seeding is required.

    The model must be in eval mode: the repeated forward passes would otherwise overwrite BatchNorm running statistics
    with adversarial batches. Its parameters' requires_grad flags are saved and restored, and no parameter gradients are
    left populated, so the model is unchanged on return.

    Gradients are computed under an explicit torch.enable_grad context so the call is safe inside a torch.no_grad
    evaluation loop. It is not safe inside torch.inference_mode, which forbids requires_grad on its tensors.

    Args:
        model: Classifier mapping images to logits.
        images: Clean inputs in [0, 1], shape (N, C, H, W).
        labels: True classes, shape (N,).
        c: Weight on the misclassification term, must be >= 0.
        kappa: Confidence margin, must be >= 0.
        steps: Adam iterations, must be >= 1.
        lr: Adam learning rate, must be > 0.

    Returns:
        Adversarial images in [0, 1], same shape and dtype as images, detached.

    Raises:
        ValueError: If c or kappa is negative, steps < 1, or lr <= 0.
    """
    if c < 0:
        raise ValueError(f"c must be >= 0, got {c}")
    if kappa < 0:
        raise ValueError(f"kappa must be >= 0, got {kappa}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")

    clean = images.detach()

    # Invert the tanh mapping so the first iterate equals the clean input.
    scaled = torch.clamp(2.0 * clean - 1.0, min=-1.0 + 1e-6, max=1.0 - 1e-6)
    w = torch.atanh(scaled).requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)

    best_adv = clean.clone()
    best_dist = torch.full((clean.shape[0],), float("inf"), device=clean.device)

    grad_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        with torch.enable_grad():
            for _ in range(steps):
                adv = 0.5 * (torch.tanh(w) + 1.0)
                logits = model(adv)

                real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
                true_mask = F.one_hot(labels, logits.shape[1]).bool()
                other = logits.masked_fill(true_mask, torch.finfo(logits.dtype).min).amax(dim=1)
                margin = torch.clamp(real - other, min=-kappa)

                dist = (adv - clean).flatten(1).pow(2).sum(dim=1)
                loss = (dist + c * margin).sum()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    dist_det = dist.detach()
                    flipped = logits.argmax(dim=1) != labels
                    improved = flipped & (dist_det < best_dist)
                    best_dist = torch.where(improved, dist_det, best_dist)
                    sample_mask = improved.view(-1, *([1] * (clean.dim() - 1)))
                    best_adv = torch.where(sample_mask, adv.detach(), best_adv)
    finally:
        for p, flag in zip(model.parameters(), grad_flags):
            p.requires_grad_(flag)

    return best_adv.detach()