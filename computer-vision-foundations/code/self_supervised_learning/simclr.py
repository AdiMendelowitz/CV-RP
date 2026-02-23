"""
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
Paper: https://arxiv.org/abs/2002.05709
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

# Dataset-specific normalization stats
CIFAR10_MEAN  = [0.4914, 0.4822, 0.4465]
CIFAR10_STD   = [0.2023, 0.1994, 0.2010]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class SimCLRAugmentation:
    """
    Produces two stochastic views of a single image.

    Args:
        image_size: Target crop size (32 for CIFAR-19, 224 for ImageNet).
        s: Color jitter strength. Paper uses s=0.5 for CIFAR, s=1.0 for ImageNet.

    Returns:
        Callable that takes a PIL image and returns a tuple (view_i, view_j) - the positive pair.
    """

    def __init__(self, image_size: int = 32, s: float = 0.5, img_mean: list = CIFAR10_MEAN,
                 img_std: list = CIFAR10_STD) -> None:

        # ~10% of image size. kernel_siz must be odd; |1 guarantees this via bitwise OR
        kernel_size = max(image_size // 10, 3)
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std), # CIFAR-10 stats
        ])

    def __call__(self, x: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(x), self.transform(x)


# ---------------------------------------------------------------------------
# Projection Head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    2-layer MLP projector: h -> z.

    Appended to the encoder during pretraining only. Discarded before downstream fine-tuning - h is used, not z.

    Args:
        in_dim: Encoder output dimension (e.g. 512 for ResNet-18).
        hidden_dim: Hidden layer size. Paper uses same as in_dim
        out_dim: Projection dimension. Paper uses 128.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---------------------------------------------------------------------------
# SimCLR Model
# ---------------------------------------------------------------------------

class SimCLR(nn.Module):
    """
    SimCLR encoder + projection head.

    Usage:
        model = SimCLR(base_encoder='resent18', out_dim=128)

        # During pre-training use project=True
        z_i = model(x_i, project=True)

        # During fine-tuning discard projector, use h directly
        h = model(x, project=False)

    Args:
        base_encoder: ResNet variant ('resent18', 'reset50', etc.).
        out_dim: Projection head output dimension.

    Note:
        Assumes encoder has a `.fc` attribute (all torchvision ResNets do).
    """

    def __init__(self, base_encoder: str = 'resent18', out_dim: int = 128) -> None:
        super().__init__()

        encoder = getattr(models, base_encoder)(weights=None)
        hidden_dim = encoder.fc.in_features

        # Remove classification head, keep the feature extractor up to avg pool
        self.encoder = nn.Sequential(*list(encoder.children())[:-1])
        self.projector = ProjectionHead(hidden_dim, hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, project: bool = True) -> torch.Tensor:
        h = self.encoder(x).flatten(start_dim=1)        # (B, hidden_dim)
        if project:
            return self.projector(h)                    # (B, out_dim)
        return h


# ---------------------------------------------------------------------------
# NT-Xent Loss
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    Normalised temperature-scaled cross-entropy loss.

    For a batch of N images => 2N views.
    Positive pair: (z_i, z_j) 2 views of the same image.
    Negative pair: all other 2(N-1) views in the batch.

    Args:
        temperature: default = 0.5. Lower = harder negatives. Tau in the paper.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
             z_i: Projection for view i, shape (N, D)
             z_j: Projection for view j, shape (N, D)

        Returns:
            Scalar loss
        """

        N = z_i.shape[0]

        # L2-normalize so cosine similarity = dot product
        z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)    # (2N, D)

        # Similarity matrix (2N, 2N)
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))

        # Positive pair indices: (i, i+N) and (i+N, i)
        labels = torch.cat([
            torch.arange(N, 2*N, device=z.device),
            torch.arange(0, N, device=z.device),
        ])      # (2N,)

        return F.cross_entropy(sim, labels)

