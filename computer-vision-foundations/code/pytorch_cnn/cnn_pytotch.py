"""
CNN in PyTorch - Matching Numpy Architecture, compare results
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataloader, TensorDataset
import numpy as np
from torchvision import datasets, transforms

# Fix seeds for initializations, makes results reproducible
torch.manual_seed(42)
np.random.seed(42)

# If using GPU:
# torch.backends.cudnn.deterministic = True


class CNNPyTorch(nn.Module):
    """
    CNN matching the Numpy architecture:
    Conv2d(1->8) -> ReLU -> MaxPool2d(2x2) -> Conv2D(8->16) -> ReLU -> MaxPool2d(2x2) -> Flatten ->
    Dense(16*7*7->128) -> ReLU -> Dense(128->10)
    """

    def __init__(self):
        super(CNNPyTorch, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # Output: (8, 28, 28)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # Output: (8, 14, 14)

        # Layer 2
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # Output: (16, 14, 14)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)    # Output: (16, 7, 7)

        # Layer 3
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.relu3 = nn.ReLU()

        # Layer 4
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        # Conv Layer 1
        x = self.conv1(x)  # (batch_size, 8, 28, 28) -> (batch_size, 8, 28, 28)
        x = self.relu1(x)
        x = self.pool1(x)  # (batch_size, 8, 28, 28) -> (batch_size, 8, 14, 14)

        # Conv Layer 2
        x = self.conv2(x)  # (batch_size, 8, 14, 14) -> (batch_size, 16, 14, 14)
        x = self.relu2(x)
        x = self.pool2(x)  # (batch_size, 16, 14, 14) -> (batch_size, 16, 7, 7)

        # Flatten
        x = torch.flatten(x, 1) # (batch_size, 16, 7, 7) -> (batch_size, 16*7*7)

        # Dense Layers
        x = self.fc1(x)  # (batch_size, 16*7*7) -> (batch_size, 128)
        x = self.relu3(x)
        x = self.fc2(x)  # (batch_size, 128) -> (batch_size, 10)
        return x


def train_pytorch(model, train_loader, test_loader, num_epochs=5, lr=0.01):
    """Train PyTroch model"""

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Combines softmax and cross-entropy
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Training history
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct, train_total = 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad() # Clear gradients from previous iterations

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)

            loss.backward()          # Backward pass
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Evaluation phase
        model.eval()
        test_loss = 0.0
        test_correct, test_total = 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        # Test metrics
        avg_test_loss = test_loss / len(test_loader)
        test_acc = test_correct / test_total

        # Save history
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(avg_test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"epoch {epoch+1}/{num_epochs} - "
            f"train_loss: {avg_train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"test_loss: {avg_test_loss:.4f}, test_acc: {test_acc:.4f}"
        )
    return history


def main():
    print("=" * 50)
    print("PyTorch CNN - Matching Numpy Architecture")
    print("=" * 50)

    print("\nLoading MNIST dataset...")
    # Define transforms: convers PIL image to tensor and scales to [0,1]
    transform = transforms.Compose([
        transforms.ToTensor(),       # Automatically scales to [0, 1] and adds channel dim
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    train_dataset_full = datasets.MNIST(
        root='./data',   # Downloads to specified directory
        train=True,
        download=True,      # Downloads if not present
        transform=transform
    )

    test_dataset_full = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_subset = torch.utils.data.Subset(train_dataset_full, range(5000))
    test_subset = torch.utils.data.Subset(test_dataset_full, range(5000))

    print(f"Train: {len(train_subset)} samples, Test: {len(test_subset)} samples")

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

    # Initialize model
    print("\nInitializing PyTorch CNN model...")
    model = CNNPyTorch()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Print model architecture
    print("\nModel architecture:")
    print(model)

    # Train model
    print("\nTraining PyTorch CNN model...")
    print("=" * 50)

    history = train_pytorch(model, train_loader, test_loader, num_epochs=5, lr=0.01)

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)

    print(f"\nfinal Results:")
    print(
        f"Train Loss: {history['train_loss'][-1]:.4f}, Train Acc: {history['train_acc'][-1]:.4f}"
    )
    print(
        f"Test Loss: {history['test_loss'][-1]:.4f}, Test Acc: {history['test_acc'][-1]:.4f}"
    )

    print("=" * 50)
    print("\nSample predictions:")
    print("=" * 50)

    model.eval()
    with torch.no_grad():
        # 10 random samples
        indices = np.random.choice(len(test_subset), 10, replace=False)
        samples = torch.stack([test_subset[i][0] for i in indices])
        labels = torch.stack([test_subset[i][1] for i in indices])

        outputs = model(samples)
        _, predicted = torch.max(outputs.data, 1)

        for i in range(10):
            true_label = labels[i].item()
            predicted_label = predicted[i].item()
            confidence = torch.softmax(outputs[i], dim=0)[predicted_label].item()

            status = "Correct" if true_label == predicted_label else "Incorrect"
            print(
                f"{status} Sample {i+1}: True={true_label}, Predicted={predicted_label}, Confidence={confidence:.4f}"
            )

    return history, model


if __name__ == "__main__":
    history, model = main()
