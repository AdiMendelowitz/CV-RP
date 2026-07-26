"""Input-normalising wrapper for pixel-space adversarial attacks.

The attacks in this toolkit operate on images in [0, 1] and assume the model normalises internally, so that the
L-infinity budget is measured in pixel space. Networks trained with dataset normalisation in the dataloader transform
expect already-normalised inputs; wrapping such a network here moves the normalisation inside the forward pass,
restoring the pixel-space contract without retraining.
"""

import torch
from torch import nn


class NormalizedModel(nn.Module):
    """Normalise inputs in [0, 1] before delegating to the wrapped model.

    The channel mean and STD are registered as buffers, so they move with .to(device) and are saved in the state dict,
    but never receive gradients. Normalisation is differentiable, so input gradients still flow back to pixel space,
    which is what the attacks require.

    Args:
        model: The trained classifier, expecting normalised inputs.
        mean: Per-channel means used during training, length C.
        std: Per-channel standard deviations used during training, length C.
    """

    def __init__(self, model: nn.Module, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)