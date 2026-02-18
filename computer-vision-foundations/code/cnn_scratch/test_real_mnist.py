"""
Test CNN on real MNIST handwritten digits
"""

import numpy as np
from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Softmax
from network import Network, CrossEntropyLoss
from train import SGD, train

# Download and load MNIST
print("Loading MNIST dataset...")
try:
    from tensorflow import keras

    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    print("✓ MNIST loaded successfully!")
except ImportError:
    print("❌ TensorFlow not found. Install with: pip install tensorflow")
    print("   Or use Option 2 below for manual download.")
    exit(1)

print(f"Original shapes: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"                 X_test={X_test.shape}, y_test={y_test.shape}")

# Preprocess data
print("\nPreprocessing...")

# 1. Add channel dimension: (N, 28, 28) → (N, 1, 28, 28)
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]

# 2. Normalize to [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# 3. Use subset for faster training (optional)
# Full MNIST: 60,000 train, 10,000 test - takes ~30 min
# Subset: 5,000 train, 1,000 test - takes ~3 min
USE_SUBSET = True

if USE_SUBSET:
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    X_test = X_test[:1000]
    y_test = y_test[:1000]
    print(f"Using subset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
else:
    print(f"Using full MNIST: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

print(f"Final shapes: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"              X_test={X_test.shape}, y_test={y_test.shape}")
print(f"Value range: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"Classes: {np.unique(y_train)} (digits 0-9)")

# Build CNN for MNIST (28x28 images, 10 classes)
print("\nBuilding CNN...")
model = Network(
    [
        # Input: (batch, 1, 28, 28)
        Conv2D(1, 8, kernel_size=3, padding=1),  # → (batch, 8, 28, 28)
        ReLU(),
        MaxPool2D(pool_size=2),  # → (batch, 8, 14, 14)
        Conv2D(8, 16, kernel_size=3, padding=1),  # → (batch, 16, 14, 14)
        ReLU(),
        MaxPool2D(pool_size=2),  # → (batch, 16, 7, 7)
        Flatten(),  # → (batch, 784)
        Dense(16 * 7 * 7, 128),  # → (batch, 128)
        ReLU(),
        Dense(128, 10),  # → (batch, 10)
        Softmax(),
    ]
)

print("✓ Model created")
print("\nArchitecture:")
print("  Conv2D(1→8, 3x3, pad=1) → ReLU → MaxPool(2x2)")
print("  Conv2D(8→16, 3x3, pad=1) → ReLU → MaxPool(2x2)")
print("  Flatten → Dense(784→128) → ReLU → Dense(128→10) → Softmax")

# Setup training
optimizer = SGD(learning_rate=0.01, momentum=0.9)
loss_fn = CrossEntropyLoss()

print("\n" + "=" * 70)
print("Starting training on MNIST...")
print("=" * 70)

# Train
history = train(
    model,
    optimizer,
    loss_fn,
    X_train,
    y_train,
    X_test,
    y_test,
    num_epochs=5,  # Start with 5 epochs
    batch_size=64,  # Larger batch for MNIST
    verbose=True,
)

print("=" * 70)
print("Training complete!\n")

# Results
print("Final Results:")
print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
print(f"  Train Acc:  {history['train_acc'][-1]:.4f}")
print(f"  Test Loss:  {history['test_loss'][-1]:.4f}")
print(f"  Test Acc:   {history['test_acc'][-1]:.4f}")

# Expected results (subset):
# Epoch 5: Train Acc ~95%, Test Acc ~93-95%

# Visualize some predictions (optional)
print("\n" + "=" * 70)
print("Sample Predictions:")
print("=" * 70)

# Get 10 random test samples
indices = np.random.choice(len(X_test), 10, replace=False)
X_sample = X_test[indices]
y_sample = y_test[indices]

# Predict
y_pred = model.forward(X_sample)
predictions = np.argmax(y_pred, axis=1)

# Show results
for i in range(10):
    true_label = y_sample[i]
    pred_label = predictions[i]
    confidence = y_pred[i, pred_label]

    status = "✓" if pred_label == true_label else "✗"
    print(
        f"{status} Sample {i + 1}: True={true_label}, Predicted={pred_label}, Confidence={confidence:.4f}"
    )

# Compute per-class accuracy (optional)
print("\n" + "=" * 70)
print("Per-Class Accuracy:")
print("=" * 70)

y_pred_all = model.forward(X_test)
predictions_all = np.argmax(y_pred_all, axis=1)

for digit in range(10):
    mask = y_test == digit
    if mask.sum() > 0:
        accuracy = (predictions_all[mask] == digit).mean()
        print(f"  Digit {digit}: {accuracy:.4f} ({mask.sum()} samples)")


import matplotlib.pyplot as plt

# Plot learning curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["test_loss"], label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Over Time")

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["test_acc"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Over Time")

plt.savefig("mnist_training.png")
print("✓ Saved plot to mnist_training.png")


# Find all mistakes
y_pred_all = model.forward(X_test)
predictions_all = np.argmax(y_pred_all, axis=1)
mistakes = np.where(predictions_all != y_test)[0]

print(f"\nTotal mistakes: {len(mistakes)}/1000")

# Visualize first 10 mistakes
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, idx in enumerate(mistakes[:10]):
    ax = axes[i // 5, i % 5]
    ax.imshow(X_test[idx, 0], cmap="gray")
    ax.set_title(f"True: {y_test[idx]}, Pred: {predictions_all[idx]}")
    ax.axis("off")
plt.savefig("mistakes.png")
