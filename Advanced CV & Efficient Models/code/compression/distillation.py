"""
Knowledge distillation for CIFAR-10

Teacher: ResNet-18 (93.43% checkpoint from benchmark experiment)
Student: SmallCNN (4 conv. blocks, ~0.5 params) or any timm model

Reference: Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"
           https://arxiv.org/abs/1503.02531
"""

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

logger = logging.getLogger(__name__)

__all__ = [
    "distillation_loss",
    "SmallCNN",
    "load_teacher_from_checkpoint",
    "build_student"
]

# ----------------------------------------------------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------------------------------------------------

def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor,
                      T: float, alpha: float) -> torch.Tensor:
    """
    Combined soft-target KL + hard-label CE loss (Hinton et al. 2015)

    L = alpha * T² * KL(softmax(student/T) || softmax(teacher/T)) + (1-alpha) * CE(student, labels)
    Multiplying KL by T² re-scales gradients to be comparable in magnitude to the CE term, making alpha a stable
    hyperparameter regardless of temperature.

    Args:
        student_logits: Raw logits from the student model, shape (B, C).
        teacher_logits: Raw logits from the teacher model, shape (B, C).
        labels: Ground truth class indices, shape (B,).
        T: Softening temperature (higher T => softer probabilities, more emphasis on teacher's relative class probabilities).
        alpha: Weight on the soft-target KL term, (1-alpha) weights CE.

    Returns:
        Scalar combined loss.
    """
    kl_loss = (
        F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean',
        )
        * (T**2)
    )

    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * kl_loss + (1 - alpha) * ce_loss


# ----------------------------------------------------------------------------------------------------------------------
# Student Architecture
# ----------------------------------------------------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """
    Lightweight 4-conv-block CNN designed for CIFAR-10 (32×32 input).

    Architecture:
        Block 1: Conv(3→32)  + BN + ReLU + MaxPool(2) → 16×16
        Block 2: Conv(32→64) + BN + ReLU + MaxPool(2) →  8×8
        Block 3: Conv(64→256)+ BN + ReLU + AdaptiveAvgPool(1) → 1×1
        Linear(256, num_classes)

    ~170K parameters, roughly 65× smaller than ResNet-18.
    """
    def __init__(self, num_classes: int=10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))

# ----------------------------------------------------------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------------------------------------------------------

def load_teacher_from_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load teacher weights into a instantiated model.

    Handles both plain state-dict files and dictionaries keyed by 'model_state_dict'.

    Args:
        model: Model architecture to load weights into.
        checkpoint_path: Path to .pth checkpoint file.
        device: Device to map tensors to.

    Returns:
        Model with loaded weights, frozen and in eval mode.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    # Remap custom ResNet shortcut keys → torchvision downsample keys.
    state_dict = {k.replace("shortcut", "downsample"): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def build_student(architecture: str = "small_cnn", num_classes: int = 10) -> nn.Module:
    """
    Instantiate a student model by name.

    Args:
        architecture: "small_cnn" for the built-in SmallCnn, or any timm model name (e.g. "resnet18", "mobilenetv3_small_100").
        num_classes: Number of output classes.

    Returns:
        Uninitialised student model.
    """

    if architecture == "small_cnn":
        return SmallCNN(num_classes=num_classes)

    try:
        import timm
        return timm.create_model(architecture, pretrained=False, num_classes=num_classes)
    except Exception as e:
        raise ValueError(f"Unknown architecture: {architecture}, install timm or use 'small_cnn'") from e









