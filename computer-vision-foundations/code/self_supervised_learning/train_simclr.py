"""
SimCLR pretraining on CIFAR0-10.

Usage:
    python train_simclr.py
    python train_simclr.py --epochs 200 --batch-size 521 --temperature 0.5
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from simclr import SimCLR, NTXentLoss, SimCLRAugmentation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset wrapper — applies SimCLR augmentation to return positive pairs
# ---------------------------------------------------------------------------

class SimCLRDataset(Dataset):
    """
    Wraps any image dataset, replacing its transform with SimCLRAugmentation.

    Each __getitem__ returns (view_i, view_j, label).
    Labels are unused during pretraining but kept for downstream evaluation.

    Args:
        dataset: A torchvision dataset with PIL images
        augmentation: SimCLRAugmentation instance
    """

    def __init__(self, dataset: Dataset, augmentation: SimCLRAugmentation) -> None:
        self.dataset = dataset
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        image, label = self.dataset[idx]
        view_i, view_j = self.augmentation(image)
        return view_i, view_j, label


# ---------------------------------------------------------------------------
# Trainin
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    loss_fn: NTXentLoss, device: torch.device) -> float:
    """Runs one training epoch, returns mean loss."""
    model.train()
    total_loss = 0

    for view_i, view_j, _ in loader:
        view_i, view_j = view_i.to(device), view_j.to(device)

        z_i, z_j = model(view_i, project=True), model(view_j, project=True)

        loss = loss_fn(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def pretrain(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # --- DATA ---
    augmentation = SimCLRAugmentation(image_size=32, s=0.5)

    # Load CIFAR-10 without transform
    base_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=None)
    dataset = SimCLRDataset(base_dataset, augmentation)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device.type == 'cuda'), drop_last=True)     # NT-Xent requires consistent batch size

    # --- MODEL ---
    model = SimCLR(base_encoder=args.encoder, out_dim=args.out_dim).to(device)
    logger.info(
        f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    # --- OPTIMIZER + SCHEDULER ---
    # LARS is used in the paper for large batches. Adam is more stable for small batches
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine decay. No warmup needed for Adam at this scale
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    loss_fn = NTXentLoss(temperature=args.temperature)

    # --- CHECKPOINT ---
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, loss_fn, device)
        scheduler.step()

        logger.info(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"loss={loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = save_dir / f"simclr_epoch{epoch:03d}.pt"
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "loss": loss, "args": vars(args),
                 },
                ckpt_path,
            )
            logger.info(f"Checkpoint saved: {ckpt_path}")

    logger.info("Pretraining complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='SimCLR pretraining on CIFAR-10')

    # DATA
    parser.add_argument("--data-dir", type=str, default='./data')
    parser.add_argument("--num-workers", type=int, default=4)

    # Model
    parser.add_argument("--encoder", type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument("--out-dim", type=int, default=512)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)

    # Checkpoint
    parser.add_argument("--save-dir", type=str, default='./checkpoints/simclr')
    parser.add_argument("--save-every", type=int, default=10)

    return parser.parse_args()

if __name__ == "__main__":
    pretrain(parse_args())
