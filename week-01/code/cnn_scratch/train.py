"""
Training utilities: Optimizer and training loop
"""

import numpy as np
from typing import List, Tuple, Optional

from numpy.ma.extras import average

from network import Network, CrossEntropyLoss


class SGD:
    """Stochastic Gradient Descent optimizer
    Update rule: param -= learning_rate * grad
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}  # For momentum

    def step(self, params: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Update parameters using their gradients
        Args:
            params: List of (parameter, gradient) tuples
        """

        for i, (param, grad) in enumerate(params):
            if grad is None:
                continue  # Skip if no gradient

            if self.momentum > 0:  # initialize velocity if needed
                if i not in self.velocities:
                    self.velocities[i] = np.zeros_like(param)
                self.velocities[i] = (
                    self.momentum * self.velocities[i] - self.learning_rate * grad
                )
                param += self.velocities[i]
            else:
                param -= self.learning_rate * grad


def compute_loss_and_accuracy(
    model: Network,
    loss_fn: CrossEntropyLoss,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    num_batches: int,
    optimizer: Optional[SGD] = None,
    train_mode: bool = False,
) -> Tuple[float, float]:
    """Compute loss and accuracy for given data"""

    if train_mode and optimizer is None:
        raise ValueError(
            "Optimizer must be provided in train mode to update parameters."
        )

    total_loss, correct = 0.0, 0

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch, y_batch = X[start:end], y[start:end]

        y_pred = model.forward(X_batch)
        loss = loss_fn.forward(y_pred, y_batch)
        total_loss += loss

        predictions = np.argmax(y_pred, axis=1)
        correct += np.sum(predictions == y_batch)

        if optimizer is not None:
            grad = loss_fn.backward()
            model.backward(grad)
            optimizer.step(model.get_params())

    average_loss = total_loss / num_batches
    accuracy = correct / (num_batches * batch_size)
    return average_loss, accuracy


def train_epoch(
    model: Network,
    optimizer: SGD,
    loss_fn: CrossEntropyLoss,
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = 32,
) -> Tuple[float, float]:
    """Train for one epoch
    Args:
        model: Network to train
        optimizer: Optimizer to update parameters (SGD)
        loss_fn: Cross-entropy loss function
        X_train: Training data (num_samples, channels, height, width)
        y_train: Training labels (num_samples,) as class indices
        batch_size: Size of each training batch

    Returns:
        (average_loss, accuracy)
    """
    num_samples: int = X_train.shape[0]
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}")
    if batch_size > num_samples:
        raise ValueError(
            f"Batch size {batch_size} must be smaller than number of samples {num_samples}"
        )

    num_batches = num_samples // batch_size
    if num_batches == 0:
        raise ValueError(
            f"Not enough samples ({num_samples}) for batch size {batch_size}. Need at least {batch_size} samples."
        )

    indices = np.random.permutation(num_samples)

    average_loss, accuracy = compute_loss_and_accuracy(
        model=model,
        loss_fn=loss_fn,
        X=X_train[indices],
        y=y_train[indices],
        batch_size=batch_size,
        num_batches=num_batches,
        optimizer=optimizer,
        train_mode=True,
    )

    return average_loss, accuracy


def evaluate(
    model: Network,
    loss_fn: CrossEntropyLoss,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
) -> Tuple[float, float]:
    """Evaluate model on test set
    Args:
        model: NN
        loss_fn: Cross-entropy loss function
        X_test: Test data
        y_test: Test labels

    Returns:
        (average_loss, accuracy)
    """
    num_batches: int = X_test.shape[0] // batch_size
    avg_loss, accuracy = compute_loss_and_accuracy(
        model=model,
        loss_fn=loss_fn,
        X=X_test,
        y=y_test,
        batch_size=batch_size,
        num_batches=num_batches,
        optimizer=None,
        train_mode=False,
    )

    return avg_loss, accuracy


def train(
    model: Network,
    optimizer: SGD,
    loss_fn: CrossEntropyLoss,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_epochs: int = 10,
    batch_size: int = 32,
    verbose: bool = True,
) -> dict:
    """Full training loop
    Args:
        model: NN
        optimizer: Optimizer (SGD)
        loss_fn: Loss function (CrossEntropyLoss)
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        num_epochs: Number of training epochs
        batch_size: Size of each training batch
    Returns:
        history: Dictionary with training and test loss/accuracy per epoch
    """

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, optimizer, loss_fn, X_train, y_train, batch_size
        )
        test_loss, test_acc = evaluate(model, loss_fn, X_test, y_test, batch_size)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if verbose:
            print(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
            )
    return history
