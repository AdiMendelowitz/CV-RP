"""
Train ResNet-18 on CIFAR-10 dataset.

CIFAR-10: 60,000 32x32 RGB images, 10 classes
Training: 50,000 images, Test: 10,000 images

Reference: "Deep Residual Learning for Image Recognition" He et al., 2015
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Tuple
from resnet import resnet18

CONFIG = {
    "num_epochs": 10,
    "batch_size": 128,
    "learning_rate": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,      # L2 regularization
    "lr_decay_epochs": [5, 8], # Reduce LR at these epochs
    "lr_decay_factor": 0.1,    # Multiply LR by this
    "num_workers": 0,
    "num_classes": 10,
    "data_dir": "./data",
    "subset_size": 5000
}

CIFAR10_CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and test transforms for CIFAR-10.

    Train transforms augmentation:
    - RandomCrop: randomly crop to 32x32 after padding (shift effect)
    - RandomHorizontalFlip: randomly mirror image
    - Normalize: zero mean, unit variance per channel

    Test transforms: only normalize (no augmentation)

    CIFAR-10 mean/std computed from training set:
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2470, 0.2435, 0.2616]

    Returns:
        (train_transform, test_transform)
    """
    # Standard CIFAR-10 normalization values
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2470, 0.2435, 0.2616]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, test_transform


def get_dataloaders(data_dir: str, batch_size: int, num_workers: int, subset_size=None) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset and create DataLoaders.

    Args:
        data_dir: Directory to save/load dataset
        batch_size: Batch size for training
        num_workers: Parallel data loading workers

    Returns:
        (train_loader, test_loader)
    """
    train_transform, test_transform = get_transforms()

    print("Loading CIFAR-10 dataset...")

    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    if subset_size:
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"  Using subset: {subset_size} samples")

        test_subset = min(1000, len(test_dataset))  # Cap at 1000 for eval
        indices = torch.randperm(len(test_dataset))[:test_subset]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        print(f"  Using test subset:  {test_subset} samples")
    else:
        print(f"  Train samples: {len(train_dataset):,}")
        print(f"  Test samples:  {len(test_dataset):,}")

    # Create DataLoaders, shuffle each epoch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False)



    return train_loader, test_loader


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer,
                device: torch.device, epoch: int,) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: ResNet model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: SGD optimizer
        device: CPU or GPU
        epoch: Current epoch number (for logging)

    Returns:
        (average_loss, accuracy)
    """
    model.train()   # Enables BatchNorm training mode and Dropout
    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Move to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Log every 100 batches
        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {total_loss/(batch_idx+1):.4f} "
                f"Acc: {100.*correct/total:.2f}% "
                f"Time: {elapsed:.1f}s"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model on test set.

    Args:
        model: ResNet model
        test_loader: Test data loader
        criterion: Loss function
        device: CPU or GPU

    Returns:
        (average_loss, accuracy)
    """
    model.eval()    # Disables BatchNorm running stats update

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():   # No gradients needed for evaluation
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def adjust_learning_rate(optimizer: optim.Optimizer, epoch: int, config: dict) -> float:
    """
    Reduce learning rate at scheduled epochs.

    Standard ResNet schedule: reduce by 10x at epochs 5 and 8
    (for our 10 epoch training - original paper uses 30, 60, 90 for 100 epochs)

    Args:
        optimizer: SGD optimizer
        epoch: Current epoch
        config: Training configuration

    Returns:
        Current learning rate
    """
    lr = config["learning_rate"]

    for decay_epoch in config["lr_decay_epochs"]:
        if epoch >= decay_epoch:
            lr *= config["lr_decay_factor"]

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def per_class_accuracy(model: nn.Module, test_loader: DataLoader,device: torch.device) -> dict:
    """
    Compute accuracy per CIFAR-10 class.

    Args:
        model: Trained ResNet model
        test_loader: Test data loader
        device: CPU or GPU

    Returns:
        Dictionary of class_name: accuracy
    """
    model.eval()

    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1

    results = {}
    for i, class_name in enumerate(CIFAR10_CLASSES):
        accuracy = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        results[class_name] = accuracy

    return results


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train(config: dict) -> dict:
    """
    Full training pipeline.

    Args:
        config: Training configuration dictionary

    Returns:
        history: Training metrics per epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, test_loader = get_dataloaders(config["data_dir"], config["batch_size"], config["num_workers"])

    # Model
    print("\nBuilding ResNet-18...")
    model = resnet18(num_classes=config["num_classes"], in_channels=3)
    model = model.to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # Loss - CrossEntropyLoss combines LogSoftmax + NLLLoss
    criterion = nn.CrossEntropyLoss()

    # Optimizer - SGD with momentum and weight decay
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],  # L2 regularization
    )

    # History
    history = {"train_loss": [], "train_acc": [], "test_loss":  [], "test_acc":  [], "lr": []}
    best_test_acc = 0.0

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    total_start = time.time()
    for epoch in range(1, config["num_epochs"] + 1):
        epoch_start = time.time()

        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, config)
        print(f"\nEpoch {epoch}/{config['num_epochs']} (lr={lr:.4f})")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Track best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "best_resnet18_cifar10.pth")

        # Log
        epoch_time = time.time() - epoch_start
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(lr)

        print(
            f"  Summary → "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

    # Final summary
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time:       {total_time/60:.1f} minutes")
    print(f"Best test acc:    {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")

    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 40)
    class_accs = per_class_accuracy(model, test_loader, device)
    for class_name, acc in class_accs.items():
        bar = "█" * int(acc * 20)
        print(f"  {class_name:<8}: {acc:.4f} {bar}")

    return history


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    history = train(CONFIG)

    # Print learning curve
    print("\nLearning Curve:")
    print("-" * 60)
    print(f"{'Epoch':<8} {'Train Acc':<12} {'Test Acc':<12} {'LR':<10}")
    print("-" * 60)
    for i in range(len(history["train_acc"])):
        print(
            f"{i+1:<8} "
            f"{history['train_acc'][i]:.4f}       "
            f"{history['test_acc'][i]:.4f}       "
            f"{history['lr'][i]:.4f}"
        )