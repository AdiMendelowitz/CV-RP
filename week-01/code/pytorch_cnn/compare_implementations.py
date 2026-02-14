"""
Compare Numpy and PyTorch implementations
"""

import numpy as np
import torch
from tensorflow import keras
import sys
import os
from cnn_pytotch import CNNPyTorch, train_pytorch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cnn_scratch"))
from layers import Conv2D, MaxPool2D, Flatten, Dense, ReLU, Softmax
from network import Network, CrossEntropyLoss
from train import SGD, train as train_numpy
from torch.utils.data import DataLoader, TensorDataset

print("=" * 50)
print("Comparing Numpy and PyTorch CNN Implementations")

print("\nLoading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Same subset as Numpy version for fair comparison
X_train = X_train[:5000]
y_train = y_train[:5000]
X_test = X_test[:1000]
y_test = y_test[:1000]

# Preprocess data
X_train = X_train[:, np.newaxis, :, :].astype(np.float32) / 255.0
X_test = X_test[:, np.newaxis, :, :].astype(np.float32) / 255.0

print(f"Dataset: Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ==============================================================================
# Train Numpy CNN
# ==============================================================================
print("=" * 50)
print("\nTraining Numpy CNN...")
print("=" * 50)

# Create Numpy model
model_numpy = Network(
    [
        Conv2D(1, 8, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(pool_size=2),
        Conv2D(8, 16, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(pool_size=2),
        Flatten(),
        Dense(16 * 7 * 7, 128),
        ReLU(),
        Dense(128, 10),
        Softmax(),
    ]
)

optimizer_numpy = SGD(learning_rate=0.01, momentum=0.9)
loss_fn_numpy = CrossEntropyLoss()

history_numpy = train_numpy(
    model=model_numpy,
    optimizer=optimizer_numpy,
    loss_fn=loss_fn_numpy,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    num_epochs=5,
    batch_size=64,
    verbose=True,
)


# ==============================================================================
# Train PyTorch CNN
# ==============================================================================
print("=" * 50)
print("\nTraining PyTorch CNN...")
print("=" * 50)

x_train_pytorch = torch.from_numpy(X_train)
y_train_pytorch = torch.from_numpy(y_train).long()
x_test_pytorch = torch.from_numpy(X_test)
y_test_pytorch = torch.from_numpy(y_test).long()

train_dataset = TensorDataset(x_train_pytorch, y_train_pytorch)
test_dataset = TensorDataset(x_test_pytorch, y_test_pytorch)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model_pytorch = CNNPyTorch()
history_pytorch = train_pytorch(
    model=model_pytorch,
    train_loader=train_loader,
    test_loader=test_loader,
    num_epochs=5,
    lr=0.01,
)

# ============================================================
# COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

print("\n{:<20} {:<15} {:<15} {:<10}".format("Metric", "NumPy", "PyTorch", "Diff"))
print("-" * 70)


# Final epoch comparison
numpy_train_acc = history_numpy["train_acc"][-1]
pytorch_train_acc = history_pytorch["train_acc"][-1]
numpy_test_acc = history_numpy["test_acc"][-1]
pytorch_test_acc = history_pytorch["test_acc"][-1]

print(
    "{:<20} {:<15.4f} {:<15.4f} {:<10.4f}".format(
        "Train Accuracy",
        numpy_train_acc,
        pytorch_train_acc,
        abs(numpy_train_acc - pytorch_train_acc),
    )
)

print(
    "{:<20} {:<15.4f} {:<15.4f} {:<10.4f}".format(
        "Test Accuracy",
        numpy_test_acc,
        pytorch_test_acc,
        abs(numpy_test_acc - pytorch_test_acc),
    )
)

numpy_train_loss = history_numpy["train_loss"][-1]
pytorch_train_loss = history_pytorch["train_loss"][-1]
numpy_test_loss = history_numpy["test_loss"][-1]
pytorch_test_loss = history_pytorch["test_loss"][-1]

print(
    "{:<20} {:<15.4f} {:<15.4f} {:<10.4f}".format(
        "Train Loss",
        numpy_train_loss,
        pytorch_train_loss,
        abs(numpy_train_loss - pytorch_train_loss),
    )
)

print(
    "{:<20} {:<15.4f} {:<15.4f} {:<10.4f}".format(
        "Test Loss",
        numpy_test_loss,
        pytorch_test_loss,
        abs(numpy_test_loss - pytorch_test_loss),
    )
)

# Analysis
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

diff_test_acc = abs(numpy_test_acc - pytorch_test_acc)

if diff_test_acc < 0.02:  # Within 2%
    print("✓ Results MATCH! Implementations are equivalent.")
    print(f"  Difference: {diff_test_acc*100:.2f}% (within 2% tolerance)")
elif diff_test_acc < 0.05:  # Within 5%
    print("⚠ Results are SIMILAR but show some variance.")
    print(f"  Difference: {diff_test_acc*100:.2f}% (acceptable, due to randomness)")
else:
    print("✗ Results DIFFER significantly.")
    print(f"  Difference: {diff_test_acc*100:.2f}%")
    print("  Possible causes:")
    print("  - Different random initialization")
    print("  - Numerical precision differences")
    print("  - Bug in one implementation")

print("\n" + "=" * 70)
