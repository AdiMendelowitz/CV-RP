"""
Evaluate trained ViT model and visualize results
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

from vit import VisionTransformer

CIFAR_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint (CPU only)"""

    model = VisionTransformer(img_size=32, patch_size=4, in_channels=3, num_classes=10, embed_dim=192, depth=12,
                              num_heads=3, mlp_ratio=4.0, dropout=0.0)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Test accuracy: {checkpoint['test_acc']: .2f}")
    return model, checkpoint

def evaluate_model(model, test_loader, device):
    """Detailed evaluation with pre-class accuracy"""

    model.eval()

    # Track prediction per class
    class_correct = [0]*10
    class_total = [0]*10

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            intputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Per class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels).item()
                class_total[label] += 1

    overall_accuracy = 100.*sum(class_correct) / sum(class_total)
    print("\n" + "=" * 70)
    print("Per-Class Accuracy:")
    print("=" * 70)
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"{CIFAR_CLASSES[i]:>10s}: {acc:6.2f}% ({class_correct[i]}/{class_total[i]})")
    print("-" * 70)
    print(f"{'Overall':>10s}: {overall_accuracy:6.2f}%")

    return overall_accuracy, class_correct, class_total, all_preds, all_labels


def plot_training_history(history_path, save_path=None):
    """Plot training curves"""

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['test_loss'], label='Test Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Test Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['test_acc'], label='Test Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Test Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate schedule
    axes[1, 0].plot(epochs, history['learning_rates'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # Test accuracy zoomed
    axes[1, 1].plot(epochs, history['test_acc'], linewidth=2, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Test Accuracy (%)')
    axes[1, 1].set_title('Test Accuracy (Zoomed)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([min(history['test_acc']) - 5, max(history['test_acc']) + 2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    plt.show()


def plot_confusion_matrix(all_preds, all_labels, save_path=None):
    """Plot confusion matrix"""

    # Compute confusion matrix
    confusion = np.zeros((10, 10), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        confusion[label][pred] += 1

    # Normalize
    confusion_norm = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(confusion_norm, cmap='Blues', aspect='auto')

    # Labels
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(CIFAR_CLASSES)
    ax.set_yticklabels(CIFAR_CLASSES)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(10):
        for j in range(10):
            text = ax.text(j, i, f'{confusion[i, j]}\n({confusion_norm[i, j]:.2f})',
                           ha="center", va="center", color="black" if confusion_norm[i, j] < 0.5 else "white",
                           fontsize=8)

    ax.set_title("Confusion Matrix (counts and proportions)", fontsize=14, pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def visualize_predictions(model, test_loader, device, num_samples=16):
    """Visualize random predictions"""

    model.eval()

    # Get a batch
    inputs, labels = next(iter(test_loader))
    inputs, labels = inputs.to(device), labels.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        confidences, predicted = probs.max(1)

    # Select random samples
    indices = np.random.choice(len(inputs), num_samples, replace=False)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()


    for i, idx in enumerate(indices):
        # Denormalize image
        img = inputs[idx].cpu().numpy().transpose(1, 2, 0)
        img = img * CIFAR10_STD + CIFAR10_MEAN
        img = np.clip(img, 0, 1)

        true_label = CIFAR_CLASSES[labels[idx].item()]
        pred_label = CIFAR_CLASSES[predicted[idx].item()]
        confidence = confidences[idx].item()

        color = 'green' if labels[idx] == predicted[idx] else 'red'
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}', fontsize=10, color=color)

    plt.tight_layout()
    plt.show()


def main():
    device = torch.device('cpu')
    print(f"Using device: {device}\n")

    checkpoint_dir = Path('./checkpoints/vit_tiny_cifar10')
    checkpoint_path = checkpoint_dir / 'best_model.pth'
    history_path = checkpoint_dir / 'training_history.json'

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first using train_vit_cifar10.py")
        return

    print("=" * 70)
    print("ViT-Tiny CIFAR-10 Evaluation")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model, checkpoint = load_model(checkpoint_path, device)

    # Load test data
    print("\nLoading test data...")
    test_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    test_loader = DataLoader(test_dataset,batch_size=128, shuffle=False, num_workers=0)

    print("\nEvaluating model...")
    overall_acc, class_correct, class_total, all_preds, all_labels = evaluate_model(model, test_loader, device)

    if history_path.exists():
        print("\nPlotting training history...")
        plot_training_history(history_path, save_path=checkpoint_dir / 'training_curves.png')

    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(all_preds, all_labels, save_path=checkpoint_dir / 'confusion_matrix.png')

    print("\nVisualizing predictions...")
    visualize_predictions(model, test_loader, device)

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
























