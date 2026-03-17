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
    "KnowledgeDistillationTrainer",
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

    L = alpha * (T^2) * KL(softmax(student/T) || softmax(teacher/T)) + (1-alpha) * CE(student, labels)

    Args:
        student_logits:
        teacher_logits:
        labels:
        T:
        alpha:

    Returns:

    """