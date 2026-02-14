"""
CNN network class - Stacks layers and manages forward/backward passes
"""

import numpy as np
from typing import List
from layers import Layer


class Network:
    """
    NN class that stacks layers

    Usage:
        From layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Softmax

        model = Network([
            Conv2D(1, 16, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2),
            Flatten(),
            Dense(16*14*14, 128),
            Softmax()
        ])
    """

    def __init__(self, layers: List[Layer]) -> None:
        """Initialize the network with a list of layers"""
        self.layers = layers

    def forward(self, input: np.ndarray) -> np.ndarray:
        """Forward pass through all layers
        Args:
            input: Input data (batch,...)
        Returns:
            Output: network output (batch, num_classes)
        """
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output: np.ndarray) -> None:
        """Backward pass through all layers in reverse order
        Args:
            grad_output: Gradient from loss function
        Returns:
            grad_input: Gradient w.r.t. input
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_params(self) -> List:
        """Get all trainable parameters from all layers
        Returns:
              List of (parameter, gradient) tuples
        """
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """Allow network(input) syntax for forward pass"""
        return self.forward(input)


class CrossEntropyLoss:
    """Cross-entropy loss for classification
    Combines with softmax for numerical stability

    Loss = -sum(y_true * log(y_pred))
    When used with softmax output, gradient simplifies to (y_pred - y_true)
    """

    def __init__(self) -> None:
        self.y_true = None
        self.y_pred = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute cross-entropy loss
        Args:
            y_pred: Predicted probabilities (batch, num_classes)
            y_true: True labels (batch, num_classes) one-hot encoded or class indices (batch,)
        Returns:
            loss: Scalar loss value
        """

        self.y_pred = y_pred
        batch_size = y_pred.shape[0]

        # Convert class indices to one-hot if necessary
        if y_true.ndim == 1:
            self.y_true = np.zeros_like(y_pred)
            self.y_true[np.arange(batch_size), y_true] = 1
        else:
            self.y_true = y_true

        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-12, 1.0 - 1e-12)

        # Cross-entropy loss
        loss = -np.sum(self.y_true * np.log(y_pred_clipped)) / batch_size
        return loss

    def backward(self) -> np.ndarray:
        """Compute gradient of loss w.r.t. predictions
        Returns:
            grad: Gradient (batch, num_classes)
        """
        if self.y_pred is None or self.y_true is None:
            raise ValueError(
                "Must call forward() before backward() to compute gradients."
            )

        batch_size = self.y_pred.shape[0]
        grad = (self.y_pred - self.y_true) / batch_size
        return grad

    def summary(self) -> None:
        """Print model architecture summary"""
        print("=" * 50)
        print(f"{'layer':<30} {'Output shape':<30} {'Params':<10}")
        print("=" * 50)

        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_name = f"{i+1}. {layer.__class__.__name__}"
            params = sum(p.size for p, _ in layer.get_params())
            total_params += params
            print(f"{layer_name:<30} {'N/A':<20} {params:<10}")

        print("=" * 50)
        print(f"Total parameters: {total_params:,}")
        print("=" * 50)
