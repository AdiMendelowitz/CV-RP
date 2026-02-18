"""
Test CNN on real MNIST-like data
"""

import numpy as np
from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Softmax
from network import Network, CrossEntropyLoss
from train import SGD, train

# Generate synthetic "digit-like" data
# (Replace with real MNIST later)
np.random.seed(42)


def generate_simple_patterns(num_samples, img_size=28):
    """Generate simple vertical/horizontal/diagonal line patterns"""
    X = np.zeros((num_samples, 1, img_size, img_size))
    y = np.zeros(num_samples, dtype=int)

    for i in range(num_samples):
        label = i % 3
        img = np.zeros((img_size, img_size))

        if label == 0:  # Vertical line
            img[:, img_size // 2] = 1.0
        elif label == 1:  # Horizontal line
            img[img_size // 2, :] = 1.0
        else:  # Diagonal line
            np.fill_diagonal(img, 1.0)

        # Add noise
        img += np.random.randn(img_size, img_size) * 0.1
        X[i, 0] = img
        y[i] = label

    return X.astype("float32"), y


# Create dataset with patterns (not random!)
X_train, y_train = generate_simple_patterns(600, img_size=14)
X_test, y_test = generate_simple_patterns(150, img_size=14)

print(f"Train set: {X_train.shape}, Labels: {y_train.shape}")
print(f"Test set:  {X_test.shape}, Labels: {y_test.shape}")
print(f"Classes: 0=vertical, 1=horizontal, 2=diagonal")

# Build CNN
model = Network(
    [
        Conv2D(1, 8, kernel_size=3, padding=1),  # (1,14,14) → (8,14,14)
        ReLU(),
        MaxPool2D(pool_size=2),  # (8,14,14) → (8,7,7)
        Conv2D(8, 16, kernel_size=3, padding=1),  # (8,7,7) → (16,7,7)
        ReLU(),
        MaxPool2D(pool_size=2),  # (16,7,7) → (16,3,3)
        Flatten(),  # (16,3,3) → (144,)
        Dense(144, 32),  # (144,) → (32,)
        ReLU(),
        Dense(32, 3),  # (32,) → (3,)
        Softmax(),
    ]
)

# Setup training
optimizer = SGD(learning_rate=0.01, momentum=0.9)
loss_fn = CrossEntropyLoss()

print("\nStarting training on patterned data...")
print("=" * 60)

# Train
history = train(
    model,
    optimizer,
    loss_fn,
    X_train,
    y_train,
    X_test,
    y_test,
    num_epochs=10,
    batch_size=32,
    verbose=True,
)

print("=" * 60)
print("Training complete!")
print(f"\nFinal Results:")
print(f"  Train Accuracy: {history['train_acc'][-1]:.4f}")
print(f"  Test Accuracy:  {history['test_acc'][-1]:.4f}")

# Expected: Should reach 90%+ accuracy!
