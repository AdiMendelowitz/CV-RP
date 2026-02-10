"""Neural network layers implemented from scratch using NumPy."""

import numpy as np
from typing import Tuple, Optional, List

def _pad_input(input: np.ndarray, padding: int) -> np.ndarray:
    """Pad the input with zeros on the borders."""
    if padding > 0:
        return np.pad(
            input,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant', constant_values=0
        )
    return input

class Layer:
    """Base class for all layers"""

    def forward(self, input: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        raise NotImplementedError("Must be implemented by subclass.")

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """Backward pass through the layer."""
        raise NotImplementedError("Must be implemented by subclass.")

    def get_params(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return a list of (parameter, gradient) tuples for optimizer."""
        return []

class Conv2D(Layer):
    """ 2D Convolutional layer"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, padding: int=0) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization of weights for ReLu
        std = np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * std
        self.bias = np.zeros(out_channels)

        # Cache for backward pass
        self.input = None
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer
        Args:
            input: (batch, in_channels, height, width)
        Returns:
            output: (batch, out_channels, height_out, width_out)
        """
        if input.ndim != 4:
            raise ValueError(f"Expected 4D input (batch, in_channels, height, width),"
                             f" got {input.shape}D with shape {input.shape}")
        if input.shape[1] != self.in_channels:
            raise ValueError(f"Expected input with {self.in_channels} channels, got {input.shape[1]} channels")

        self.input = input
        batch_size, in_channels, height_in, width_in = input.shape

        input_padded = _pad_input(input, self.padding)

        height_out = (height_in + 2*self.padding - self.kernel_size) // self.stride+1
        width_out = (width_in + 2*self.padding - self.kernel_size) // self.stride+1
        output = np.zeros((batch_size, self.out_channels, height_out, width_out))

        # Convolution operation
        for b in range(0, batch_size):
            for oc in range(0, self.out_channels):
                for ho in range(0, height_out):
                    for wo in range(0, width_out):
                        h_start = ho * self.stride
                        w_start = wo * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        # Extract patch (all input channels for this patch)
                        patch = input_padded[b, :, h_start:h_end, w_start:w_end]

                        # Convolve: patch shape (in_channels, kernel_size, kernel_size),
                        #          weights shape (out_channels, in_channels, kernel_size, kernel_size)
                        output[b, oc, ho, wo] = (
                                np.sum(patch * self.weights[oc]) + self.bias[oc]
                        )

        return output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """Backward pass for convolutional layer
        Args:
            output_gradient: Gradient for next layer (batch, out_channels, height_out, width_out)
        Returns:
            input_gradient: Gradient w.r.t. input (batch, in_channels, height_in, width_in)
        """
        batch_size, _, height_out, width_out = output_gradient.shape
        _, in_channels, height_in, width_in = self.input.shape

        input_padded = _pad_input(self.input, self.padding)

        # Initialize gradients
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        grad_input_padded = np.zeros_like(input_padded)

        # Compute gradients
        for bs in range(0, batch_size):
            for oc in range(0, self.out_channels):
                for ho in range(0, height_out):
                    for wo in range(0, width_out):
                        h_start = ho * self.stride
                        w_start = wo * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        # Current gradient value
                        grad = output_gradient[bs, oc, ho, wo]

                        # Gradient w.r.t. bias: sum over all positions
                        self.grad_bias[oc] += grad

                        # Extract patch
                        patch = input_padded[bs, :, h_start:h_end, w_start:w_end]

                        # Gradient w.r.t. weights
                        self.grad_weights[oc] += grad * patch

                        # Gradient w.r.t. input
                        grad_input_padded[bs, :, h_start:h_end, w_start:w_end] += (
                                grad * self.weights[oc]
                        )

        # Remove padding from input gradient
        if self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_padded
        return grad_input

    def get_params(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """ Prints number of params and returns (param, grad) pairs for weights and bias"""
        return [(self.weights, self.grad_weights), (self.bias, self.grad_bias)]

    def __repr__(self) -> str:
        return (f"Conv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")

class MaxPool2D(Layer):
    """ 2D Max Pooling layer
    Forward: take max value in each pool_size x pool_size window
    Backward: propagate gradient only to the max value in the window

    Input/Output shape: (batch, channels, height, width)
    """

    def __init__(self, pool_size: int=2, stride: Optional[int]=None) -> None:
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.input = None
        self.max_indices = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """ Forward max pooling
        Args:
            input: (batch, channels, height, width)
        Returns:
            output: (batch, channels, height_out, width_out)
        """
        self.input = input
        batch_size, channels, height_in, width_in = input.shape

        height_out = (height_in - self.pool_size) // self.stride + 1
        width_out = (width_in - self.pool_size) // self.stride + 1
        output = np.zeros((batch_size, channels, height_out, width_out))

        self.max_indices = np.zeros((batch_size, channels, height_out, width_out, 2), dtype=int)

        # Max pooling operation
        for b in range(0, batch_size):
            for c in range(0, channels):
                for ho in range(0, height_out):
                    for wo in range(0, width_out):
                        h_start = ho * self.stride
                        w_start = wo * self.stride
                        h_end = h_start + self.pool_size
                        w_end = w_start + self.pool_size

                        # Extract pooling window
                        window = input[b, c, h_start:h_end, w_start:w_end]

                        # Find max value and its index
                        max_val = np.max(window)
                        max_pos = np.unravel_index(np.argmax(window), window.shape)

                        output[b, c, ho, wo] = max_val
                        self.max_indices[b, c, ho, wo] = [
                            h_start + max_pos[0],
                            w_start + max_pos[1]
                        ]
        return output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """ Backward max pooling.
        Gradient is propagated to the position with max value, rest get zero gradient
        Args:
            output_gradient: (batch, channels, height_out, width_out)
        Returns:
            input_gradient: (batch, channels, height_in, width_in)
        """

        batch_size, channels, height_out, width_out = output_gradient.shape
        grad_input = np.zeros_like(self.input)

        # Propagate gradients to max positions
        for b in range(0, batch_size):
            for c in range(0, channels):
                for ho in range(0, height_out):
                    for wo in range(0, width_out):
                        max_pos = self.max_indices[b, c, ho, wo]
                        grad_input[b, c, max_pos[0], max_pos[1]] += output_gradient[b, c, ho, wo]
        return grad_input

class Dense(Layer):
    """ Fully connected dense layer
    Forward: output = input @ weights + bias
    Backward: compute gradients w.r.t. weights, bias and input

    Input shape: (batch, input_size)
    Output shape: (batch, output_size)
    Weights shape: (input_size, output_size)
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.output_size = output_size

        # He initialization for ReLU
        std = np.sqrt(2. / input_size)
        self.weights = np.random.randn(input_size, output_size) * std
        self.bias = np.zeros(output_size)

        # Cache for backward pass
        self.input = None
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """ Forward pass for dense layer"""
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """ Backward dense layer
        Given: d_loss/d_output
        Compute: d_loss/d_input, d_loss/d_weights, d_loss/d_biases

        Args:
            output_gradient: (batch, output_size)
        Returns:
            input_gradient: (batch, input_size)
        """
        self.grad_weights = np.dot(self.input.T, output_gradient)
        self.grad_bias = np.sum(output_gradient, axis=0)
        return np.dot(output_gradient, self.weights.T)

    def get_params(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """ Return (param, grad) pairs"""
        return [(self.weights, self.grad_weights), (self.bias, self.grad_bias)]


class ReLU(Layer):
    """ ReLU activation layer
    Forward: output = max(0, input)
    Backward: gradient is propagated only for positive input values
    """

    def __init__(self) -> None:
        self.input = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """ Forward ReLU"""
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """ Backward ReLU
        Gradient is propagated only for positive input values
        """
        return output_gradient * (self.input > 0)

class Softmax(Layer):
    """ Softmax activation layer
    Forward: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    Backward: Jacobian of softmax is used to compute gradient w.r.t. input
    """

    def __init__(self) -> None:
        self.output = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """ Forward softmax
        Args:
            input: (batch, num_classes)
        Returns:
            output: (batch, num_classes) - probabilities sum to 1
        """
        exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))  # for numerical stability
        self.output = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """ Backward softmax
        NOTE: When used with CrossEntropyLoss, the combined gradient simplifies to (softmax_output - one_hot_labels).
        This general implementation is kept for flexibility but is less efficient.
        """
        batch_size = self.output.shape[0]
        input_gradient = np.zeros_like(output_gradient)

        for i in range(batch_size):
            y = self.output[i].reshape(-1, 1)  # (num_classes, 1)
            jacobian = np.diagflat(y) - np.dot(y, y.T)  # (num_classes, num_classes)
            input_gradient[i] = np.dot(output_gradient[i], jacobian)

        return input_gradient


class Flatten(Layer):
    """
    Flatten layers: Reshape from (batch, channels, height, width) to (batch, channels*height*width) for fully connected layers
    """

    def __init__(self) -> None:
        self.input_shape = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        """ Forward flatten
        Args:
            input: (batch, channels, height, width)
        Returns:
            output: (batch, channels*height*width)
        """
        self.input_shape = input.shape
        batch_size = input.shape[0]
        return input.reshape(batch_size, -1)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """ Backward flatten
        Args:
            output_gradient: (batch, channels*height*width)
        Returns:
            input_gradient: (batch, channels, height, width)
        """
        return output_gradient.reshape(self.input_shape)
