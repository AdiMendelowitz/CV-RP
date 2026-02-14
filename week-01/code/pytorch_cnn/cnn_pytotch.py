"""
CNN in PyTorch - Matching Numpy Architecture, compare results
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataloader, TensorDataset
import numpy as np
from tensorflow import keras

torch.manual_seed(42)
np.random.seed(42)

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
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        #Layer 3
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.relu3 = nn.ReLU()

        # Layer 4
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        # Conv Layer 1
        x = self.conv1(x) # (batch_size, 8, 28, 28) -> (batch_size, 8, 28, 28)
        x = self.relu1(x)
        x = self.pool1(x) # (batch_size, 8, 28, 28) -> (batch_size, 8, 14, 14)

        # Conv Layer 2
        x = self.conv2(x) # (batch_size, 8, 14, 14) -> (batch_size, 16, 14, 14)
        x = self.relu2(x)
        x = self.pool2(x) # (batch_size, 16, 14, 14) -> (batch_size, 16, 7, 7)

        # Flatten
        x = x.view(x.size(0), -1) # (batch_size, 16, 7, 7) -> (batch_size, 16*7*7)

        # Dense Layers
        x = self.fc1(x) # (batch_size, 16*7*7) -> (batch_size, 128)
        x = self.relu3(x)
        x = self.fc2(x) # (batch_size, 128) -> (batch_size, 10)
        return x

def train_pytorch(model, train_loader, test_loader, num_epochs=5, lr=0.01):
    """Train PyTroch model"""

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss() # Combines softmax and cross-entropy
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Training history
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct, train_total = 0, 0

        for inputs, labels in train_loader:
            outputs = model(inputs)         # Forward pass
            loss = criterion(outputs, labels)

            optimizer.zero_grad()           # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Training metrics
        avg_trian_loss = train_loss / len(train_loader)
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
        history['train_loss'].append(avg_trian_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_acc)

        print(f"epoch {epoch+1}/{num_epochs} - "
              f"train_loss: {avg_trian_loss:.4f}, train_acc: {train_acc:.4f}, "
              f"test_loss: {avg_test_loss:.4f}, test_acc: {test_acc:.4f}")
    return history

def main():
    print("="*50)
    print("PyTorch CNN - Matching Numpy Architecture")
    print("="*50)

    print("\nLoading MNIST dataset...")
    (x_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Same subset as Numpy version for fair comparison
    X_train = x_train[:5000]
    y_train = y_train[:5000]
    X_test = X_test[:1000]
    y_test = y_test[:1000]

    # Preprocess data
    X_train = X_train[:, np.newaxis, :, :].astype(np.float32) / 255.0
    X_test = X_test[:, np.newaxis, :, :].astype(np.float32) / 255.0

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Create PyTorch datasets and loaders
    X_train_torch = torch.from_numpy(X_train)
    y_train_torch = torch.from_numpy(y_train).long()
    X_test_torch = torch.from_numpy(X_test)
    y_test_torch = torch.from_numpy(y_test).long()

    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    test_dataset = TensorDataset(X_test_torch, y_test_torch)

    train_loader = dataloader.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = dataloader.DataLoader(test_dataset, batch_size=64, shuffle=False)

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
    print("="*50)

    history = train_pytorch(model, train_loader, test_loader, num_epochs=5, lr=0.01)

    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)

    print(f"\nfinal Results:")
    print(f"Train Loss: {history['train_loss'][-1]:.4f}, Train Acc: {history['train_acc'][-1]:.4f}")
    print(f"Test Loss: {history['test_loss'][-1]:.4f}, Test Acc: {history['test_acc'][-1]:.4f}")

    print("="*50)
    print("\nSample predictions:")
    print("="*50)

    model.eval()
    with torch.no_grad():
        # 10 random samples
        indices = np.random.choice(len(X_test), 10, replace=False)
        samples = X_test_torch[indices]
        labels = y_test_torch[indices]

        outputs = model(samples)
        _, predicted = torch.max(outputs.data, 1)

        for i in range(10):
            true_label = labels[i].item()
            predicted_label = predicted[i].item()
            confidence = torch.softmax(outputs[i], dim=0)[predicted_label].item()

            status = "Correct" if true_label == predicted_label else "Incorrect"
            print(f"{status} Sample {i+1}: True={true_label}, Predicted={predicted_label}, Confidence={confidence:.4f}")

    return history, model

if __name__ == "__main__":
    history, model = main()