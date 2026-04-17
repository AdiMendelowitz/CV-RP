"""
U-Net training script for LGG MRI brain tumour segmentation.

Dataset: LGG MRI Segmentation
  Buda M., Saha A., Mazurowski M.A. "Association of genomic subtypes of lower-grade gliomas with shape
  features automatically extracted by a deep learning algorithm." Computers in Biology and Medicine, 2019.
  https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

Folder layout (default path set in CONFIG):
  data/kaggle_3m/
    TCGA_<ID>/
      <slice>.tif        <- grayscale MRI, uint8
      <slice>_mask.tif   <- binary mask, values 0 or 255

Split is patient-level (one TCGA_* directory = one patient) to prevent anatomy leakage between train and
validation sets.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from segmentation_loss import combined_loss, dice_loss
from unet import UNet

_DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "kaggle_3m"

CONFIG = {
    "data_root": _DATA_ROOT,
    "checkpoint_dir": Path(__file__).resolve().parent / "checkpoints",
    "img_size": 256,
    "in_channels": 1,
    "num_classes": 1,
    "batch_size": 4,
    "epochs": 20,
    "lr": 1e-4,
    "val_fraction": 0.1,
    "alpha": 0.3,  # BCE weight; 1 - alpha = 0.7 goes to Dice
    "seed": 42,
}


# ---------------------------------------------------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------------------------------------------------


class LGGDataset(Dataset):
    """LGG MRI segmentation dataset.

    Each sample is a (image, mask) pair where:
    - image: single-channel float32 tensor, normalised to [0, 1]
    - mask: binary float32 tensor

    Augmentation applies identical spatial transforms to image and mask using
    torchvision.transforms.functional with a shared random state. Mask transforms always use
    NEAREST interpolation to preserve binary values.

    Args:
        slices: List of (image_path, mask_path) pairs.
        img_size: Height and width are resized to this value.
        augment: If True, apply random horizontal flip and rotation.
    """

    def __init__(self, slices: list[tuple[Path, Path]], img_size: int, augment: bool) -> None:
        self.slices = slices
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        """Return the number of image-mask pairs in the dataset."""
        return len(self.slices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load, resize, optionally augment, and return one image-mask pair.

        Args:
            idx: Index into the slices list.

        Returns:
            Tuple of (image, mask) where image is float32 in [0, 1] with shape (1, H, W)
            and mask is a binary float32 tensor with shape (1, H, W).
        """
        img_path, mask_path = self.slices[idx]
        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img, mask = self._resize(img, mask)
        if self.augment:
            img, mask = self._augment(img, mask)

        # Divide uint8 by 255, giving float32 in [0,1].
        img = TF.to_tensor(img)
        # Mask pixels are 0 or 255; divide by 255 to get binary 0.0 or 1.0.
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return img, mask

    def _resize(self, img: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        """Resize image and mask to img_size x img_size."""
        size = (self.img_size, self.img_size)
        img = TF.resize(img, size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size, interpolation=TF.InterpolationMode.NEAREST)
        return img, mask

    def _augment(self, img: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        """Apply random horizontal flip and rotation to image and mask in sync."""
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        return img, mask


# ---------------------------------------------------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------------------------------------------------


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders with a patient-level split.

    Args:
        config: Training configuration dictionary.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    data_root = config["data_root"]
    patient_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())

    if not patient_dirs:
        raise ValueError(f"No patient directories found in {data_root}")

    # Collect all pairs by patient.
    patient_slices: dict[Path, list[tuple[Path, Path]]] = {}
    for patient_dir in patient_dirs:
        pairs = []
        for img_path in sorted(patient_dir.glob("*.tif")):
            if img_path.stem.endswith("_mask"):
                continue
            mask_path = patient_dir / f"{img_path.stem}_mask.tif"
            if mask_path.exists():
                pairs.append((img_path, mask_path))

        if pairs:
            patient_slices[patient_dir] = pairs

    patients = list(patient_slices.keys())
    random.shuffle(patients)

    n_val = max(1, int(len(patients) * config["val_fraction"]))
    val_patients = patients[:n_val]
    train_patients = patients[n_val:]

    train_slices = [pair for p in train_patients for pair in patient_slices[p]]
    val_slices = [pair for p in val_patients for pair in patient_slices[p]]

    train_dataset = LGGDataset(train_slices, config["img_size"], augment=True)
    val_dataset = LGGDataset(val_slices, config["img_size"], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    print(
        f"Patients -- train: {len(train_patients)}, val: {len(val_patients)}\n"
        f"Slices -- train: {len(train_slices):,} val: {len(val_slices):,}"
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------------------------------------------------
# Train / validation loops
# ---------------------------------------------------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    alpha: float,
) -> float:
    """Run one training epoch and return the mean combined loss.

    Args:
        model: U-Net model in training mode.
        loader: DataLoader yielding (image, mask) batches.
        optimizer: Optimizer with parameters already registered.
        device: Device to move tensors to before the forward pass.
        alpha: BCE weight passed to combined_loss.

    Returns:
        Mean combined loss over all batches in the epoch.
    """
    model.train()
    total_loss = 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        loss = combined_loss(model(imgs), masks, alpha=alpha)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    alpha: float,
) -> tuple[float, float]:
    """Evaluate the model on a DataLoader and return mean loss and mean Dice score.

    Args:
        model: U-Net model set to eval mode inside this function.
        loader: DataLoader yielding (image, mask) batches.
        device: Device to move tensors to before the forward pass.
        alpha: BCE weight passed to combined_loss.

    Returns:
        Tuple of (mean_loss, mean_dice) over all batches.
    """
    model.eval()
    total_loss, total_dice = 0.0, 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        logits = model(imgs)
        loss = combined_loss(logits, masks, alpha=alpha)
        dice = 1.0 - dice_loss(torch.sigmoid(logits), masks)

        total_loss += loss.item()
        total_dice += dice.item()

    n = len(loader)
    return total_loss / n, total_dice / n


# ---------------------------------------------------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------------------------------------------------


def save_training_curves(history: dict, output_dir: Path) -> None:
    """Plot and save training loss and validation Dice curves to output_dir/training_curves.png.

    Args:
        history: Dictionary with keys "train_loss", "val_loss", and "val_dice", each a list of
                 per-epoch scalar values.
        output_dir: Directory in which to write training_curves.png.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["val_loss"], label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Combined loss (BCE + Dice)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["val_dice"], color="green", label="Val Dice")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice score")
    ax2.set_title("Validation Dice score")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    path = output_dir / "training_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved to {path}")


if __name__ == "__main__":
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    CONFIG["checkpoint_dir"].mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(CONFIG)

    model = UNet(in_channels=CONFIG["in_channels"], num_classes=CONFIG["num_classes"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_val_loss = float("inf")

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, CONFIG["alpha"])
        val_loss, val_dice = evaluate(model, val_loader, device, CONFIG["alpha"])

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        print(
            f"Epoch {epoch:02d}/{CONFIG['epochs']}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_dice={val_dice:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "config": CONFIG,
                },
                CONFIG["checkpoint_dir"] / "best_unet_lgg.pth",
            )
            print(f"  -> Saved best checkpoint (val_loss={val_loss:.4f})")

    save_training_curves(history, CONFIG["checkpoint_dir"])
    print(f"\nBest val_loss={best_val_loss:.4f}")
