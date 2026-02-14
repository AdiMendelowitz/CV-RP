import numpy as np
from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Softmax
from network import Network, CrossEntropyLoss
from train import SGD, train

np.random.seed(42)

X_train = np.random.rand(100, 1, 8, 8).astype(
    "float32"
)  # 100 samples, 1 channel, 8x8 images
y_train = np.random.randint(0, 3, 100)

X_test = np.random.rand(20, 1, 8, 8).astype("float32")
y_test = np.random.randint(0, 3, 20)

# Tiny CNN model
model = Network(
    [
        Conv2D(1, 4, kernel_size=3, padding=1),  # (1, 8, 8) -> (4, 8, 8)
        ReLU(),
        MaxPool2D(pool_size=2),  # (4, 8, 8) -> (4, 4, 4)
        Flatten(),  # (4, 4, 4) -> (64,)
        Dense(64, 3),  # (64,) -> (3,)
        Softmax(),
    ]
)

# Setup training
optimizer = SGD(learning_rate=0.01, momentum=0.9)
loss_fn = CrossEntropyLoss()

print("Starting training...")
print("=" * 30)

history = train(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    num_epochs=5,
    batch_size=16,
    verbose=True,
)

print("=" * 30)
print("Starting testing...")
print(
    f"\nFinal Test Loss: {history['test_loss'][-1]:.4f}, Final Test Accuracy: {history['test_acc'][-1]:.4f}"
)
