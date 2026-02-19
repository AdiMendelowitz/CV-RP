"""
Train Vision Transformer ViT-Tiny on CIFAR-10
Configuration:
- Patch size: 4x4
- Embedding dim: 192
- Depth: 12 layers
- Heads: 3
- Image size: 32x32
"""
import math

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import time
from pathlib import Path
import json
from typing import Tuple, Optional
import logging
from tqdm import tqdm
from vit import VisionTransformer


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
DEFAULT_SAVE_DIR = Path('./checkpoints/vit_tiny_cifar10')
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 50

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_cifar10_dataloaders(batch_size: int=64, num_workers: int=0, data_dir: str='./data') \
        -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 dataloader with augmentation
    ViT requires stronger augmentation than CNNs to prevent overfitting

    CPU-optimized: num_workers=0 to avoid multiprocessing overhead
    """

    # Training augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),              # 50% chance flip
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD), # CIFAR-10 statistics
    ])

    # Test transform, no augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    return train_loader, test_loader


def create_checkpoint(epoch: int, model: nn.Module, optimizer: optim.Optimizer, test_acc: float,
                      test_loss: float, history: Optional[dict]=None) -> dict:
    """
    Creates a checkpoint dictionary for saving model state
    Args:
        epoch: Current epoch number
        model: Current model being trained
        optimizer: The optimizer
        test_acc: Test accuracy at this checkpoint
        test_loss: Test loss at this checkpoint
        history: Optional training history dict
    Returns:
        Dictionary containing all checkpoint information
    """
    checkpoint = {
        'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc, 'test_loss': test_loss
    }
    if history is not None:
        checkpoint['history'] = history

    return checkpoint

def create_vit_tiny_cifar10():
    """
    ViT-Tiny configured for CIFAR-10:
    - Image size: 32x32
    - Patch size: 4x4 -> (32/4)^2=64 patches
    - 10 classes
    """
    model = VisionTransformer(img_size=32, patch_size=4, in_channels=3, num_classes=10, embed_dim=192, depth=12,
                              num_heads=3, mlp_ratio=4.0, dropout=0.1)
    return model

def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer,
                scheduler: Optional[optim.lr_scheduler._LRScheduler], device: torch.device, epoch: int) -> Tuple[float, float, float]:
    """Train for one epoch
    Returns:
        Tuple of (train_loss, train_acc, epoch_time_seconds)
    """
    model.train()

    running_loss, correct, total = 0.0, 0, 0

    start_time = time.time()
    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()       # Zero gradients

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping (important for transformer training)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if scheduler is not None:       # Update learning rate (if using warmup)
            scheduler.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print progress every 100 batches
        if (batch_idx + 1) % LOG_INTERVAL == 0:
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            lr = optimizer.param_groups[0]['lr']
            print(f"    batch [{batch_idx + 1}/{len(train_loader)}] | "
                  f" Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | Learning Rate: {lr:.6f}")

    epoch_time = time.time() - start_time
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100.0 * correct / total

    return train_loss, train_accuracy, epoch_time

def validate(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device)\
        -> Tuple[float, float]:
    """Validate on test set
    Args:
          model: The model to validate
          test_loader: Dataloader for test data
          criterion: Loss function
          device: Device to run validation on
    Returns:
            Tuple of (test_loss, test_acc)
    """
    model.eval()

    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_accuracy = 100.0 * correct / total

    return test_loss, test_accuracy

def train_vit_tiny(num_epochs=100, batch_size=64, lr=0.001, weight_decay=0.05, warmup_epochs=10,
                   save_dir: str='./checkpoints/vit_tiny_cifar10', early_stopping_patience=10) -> Tuple[dict, VisionTransformer]:
    """
    Train ViT-Tiny on CIFAR-10 (CPU optimized)

    Args:
        num_epochs: Total training epochs (100 for CPU, 200 for GPU)
        batch_size: Batch size (64 for CPU, 128 for GPU)
        lr: Peak learning rate after warmup
        weight_decay: Weight decay for AdamW
        warmup_epochs: Number of linear warmup epochs
        save_dir: Directory to save checkpoints
        early_stopping_patience: Stop if no improvement for this many epochs

    Returns:
        Tuple of (training_history_dict, trained_model)
    """

    assert num_epochs>0, "num_epochs must be positive"
    assert batch_size>0, "batch_size must be positive"
    assert 0<lr<1, "lr must be in (0,1)"
    assert warmup_epochs<num_epochs, "warmup_epochs must be < num_epochs"

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')
    logger.info(f"Using device: {device}")  # Use the module-level logger
    logger.warning("⚠️  CPU training will be slow (~12-24 hours for 100 epochs)")
    logger.info("💡 Consider reducing num_epochs to 50 for faster experimentation\n")


    # Data
    print("\n" + "=" * 70)
    print("Loading CIFAR-10 dataset...")
    print("=" * 70)
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")

    # Model
    print("\n" + "=" * 70)
    print("Creating ViT-Tiny model...")
    print("=" * 70)
    model = create_vit_tiny_cifar10()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nModel configuration:")
    print(f"  Image size: 32×32")
    print(f"  Patch size: 4×4")
    print(f"  Number of patches: 64")
    print(f"  Embedding dim: 192")
    print(f"  Depth: 12")
    print(f"  Attention heads: 3")


    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for regularization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    # Learning rate schedular: linear warmup + cosine decay
    warmup_steps = warmup_epochs * len(train_loader)
    total_steps = num_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step/warmup_steps

        # Cosine decay
        progress = (step - warmup_steps)/(total_steps - warmup_steps)
        return 0.5*(1 + torch.cos(torch.tensor(progress)*math.pi))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print("\n" + "=" * 70)
    print("Training configuration:")
    print("=" * 70)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Peak learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Optimizer: AdamW")
    print(f"LR schedule: Linear warmup + Cosine decay")
    print(f"Label smoothing: 0.1")
    print(f"Gradient clipping: 1.0")

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'learning_rates': []}
    best_acc = 0.0

    epochs_without_improvement = 0

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    for epoch in range(num_epochs):
        print(f"\nEpoch: [{epoch+1}/{num_epochs}]")
        print("=" * 70)

        train_loss, train_acc, epoch_time = train_epoch(
            model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer,
            scheduler=scheduler, device=device, epoch=epoch
        )

        test_loss, test_acc = validate(model=model, test_loader=test_loader, criterion=criterion, device=device)

        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['learning_rates'].append(current_lr)

        print(f"\n  Epoch Summary:")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print(f"    Learning Rate: {current_lr:.6f}")
        print(f"    Time: {epoch_time:.2f}s")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint = create_checkpoint(epoch=epoch+1, model=model, optimizer=optimizer, test_acc=test_acc,
                                           test_loss=test_loss, history=None)
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f"    ✓ New best accuracy! Saved checkpoint.")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
            print(f"    No improvement for {early_stopping_patience} consecutive epochs")
            print(f"    Best test accuracy: {best_acc:.2f}%")
            break  # Exit training loop

        # Save checkpoint every 50 epochs
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint = create_checkpoint(epoch=epoch+1, model=model, optimizer=optimizer, test_acc=test_acc,
                                           test_loss=test_loss, history=history)

            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch + 1}.pth')
            print(f"    ✓ Checkpoint saved.")

    checkpoint = create_checkpoint(epoch=num_epochs, model=model, optimizer=optimizer, test_acc=test_acc,
                                   test_loss=test_loss, history=history)
    torch.save(checkpoint, save_dir / 'final_model.pth')

    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Checkpoints saved to: {save_dir}")

    return history, model

if __name__ == '__main__':
    torch.manual_seed(42)

    # Train ViT-Tiny on CIFAR-10 (CPU-optimized)
    history, model = train_vit_tiny(
        num_epochs=100,  # Reduced for CPU (50 for quick test, 100 for full run)
        batch_size=64,  # Reduced for CPU efficiency
        lr=0.001,  # Peak learning rate
        weight_decay=0.05,  # Weight decay for regularization
        warmup_epochs=10,  # Linear warmup
        early_stopping_patience=10
    )

    print("\n" + "=" * 70)
    print("Training summary saved to ./checkpoints/vit_tiny_cifar10/")
    print("=" * 70)
    print("Expected accuracy after 50 epochs: ~70-75%")
    print("Expected accuracy after 100 epochs: ~75-80%")