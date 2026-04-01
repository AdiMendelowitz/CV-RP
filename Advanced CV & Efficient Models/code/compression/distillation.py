"""
Knowledge distillation for CIFAR-10

Teacher: ResNet-18 (93.43% checkpoint from benchmark experiment)
Student: SmallCNN (4 conv. blocks, ~0.5 params) or any timm model

Reference: Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"
           https://arxiv.org/abs/1503.02531
"""

import logging
import random
from ast import Module
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
        Block 1: Conv(3→32)   + BN + ReLU + MaxPool(2) → 16×16
        Block 2: Conv(32→64)  + BN + ReLU + MaxPool(2) →  8×8
        Block 3: Conv(64→128) + BN + ReLU + MaxPool(2) →  4×4
        Block 4: Conv(128→256)+ BN + ReLU + AdaptiveAvgPool(1) → 1×1
        Linear(256, num_classes)

    ~0.5M parameters — roughly 20× smaller than ResNet-18.
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
# Trainer
# ----------------------------------------------------------------------------------------------------------------------

class KnowledgeDistillationTrainer:
    """
    Trains a student model via knowledge distillation from a frozen teacher.

    Args:
        teacher: Pre-trained model, frozen on construction.
        student: Studet model to train.
        device: Compute device.
        temperature: Distillation temperature T (default 4.0).
        alpha: KL loss weight, CE is (1-alpha) (default 0.7).
        lr: Adam learning rate for student (default 1e-3).
        checkpoint_dir: Directory for best student checkpoints.
    """

    def __init__(self, teacher: nn.Module, student: nn.Module, device: torch.device, temperature: float = 4.0,
                 alpha: float = 0.7, lr: float = 1e-3, checkpoint_dir: str = "checkpoints/distillation") -> None:
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        self.teacher = teacher.to(device).eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student = student.to(device)
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int=30) -> dict[str, list[float]]:
        """
        Run the full distillation training loop.

        Args:
            train_loader: Dataloader for training split.
            val_loader: Dataloader for validation/test split.
            epochs: Number of epochs to train.

        Returns:
            History dict with keys: train_loss, train_acc, val_loss, val_acc.
            Accuracy values in [0,1]
        """

        history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0

        for epoch in range(1, epochs+1):
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._val_epoch(val_loader)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            logger.info("Epoch %d/%d: Train Loss=%.4f, Train Acc=%.4f, Val Loss=%.4f, Val Acc=%.4f",
                        epoch, epochs, train_loss, train_acc*100, val_loss, val_acc*100)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc)

            logger.info("Training completed. Best val_acc=%.2f%%`")

        return history

    def _train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        self.student.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.no_grad():
                teacher_logits = self.teacher(images)

            student_logits = self.student(images)
            loss = distillation_loss(student_logits, teacher_logits, labels, self.temperature, self.alpha)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (student_logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        return total_loss / total, correct / total

    def _val_epoch(self, val_loader: DataLoader) -> tuple[float, float]:
        self.student.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                teacher_logits = self.teacher(images)
                student_logits = self.student(images)
                loss = distillation_loss(student_logits, teacher_logits, labels, self.temperature, self.alpha)

                total_loss += loss.item() * images.size(0)
                correct += (student_logits.argmax(dim=1) == labels).sum().item()
                total += images.size(0)

        return total_loss / total, correct / total

    def _save_checkpoint(self, epoch: int, val_acc: float) -> None:
        path = self.checkpoint_dir / "best_student.pth"
        torch.save({"epoch": epoch, "val_acc": val_acc, "model_state_dict": self.student.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   path)

        logger.info("Checkpoint saved to %s (val_acc=%.2f%%)", path, val_acc*100)

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









