# CNN from Scratch: NumPy Only

A complete Convolutional Neural Network implementation using only NumPy, built to understand deep learning from first principles.

## Features

- Full backpropagation through convolutional and dense layers
- He weight initialization
- SGD optimizer with momentum
- Achieves 100% accuracy on line orientation classification (3-class synthetic dataset)

## Line Classification Results

Trained on 600 14x14 images (vertical/horizontal/diagonal lines):

- **Test Accuracy:** 100%
- **Train Accuracy:** 100%
- Converges in 2 epochs

## Line Classification Architecture

```
Conv2D(1->8) -> ReLU -> MaxPool2D ->
Conv2D(8->16) -> ReLU -> MaxPool2D ->
Flatten -> Dense(144->32) -> ReLU -> Dense(32->3) -> Softmax
```

This network targets 3 output classes. The MNIST adaptation uses a different
final dense layer configuration with 10 output classes and a larger hidden layer.

## Implementation Highlights

- No deep learning libraries (PyTorch, TensorFlow, or similar)
- Manual gradient computation for all layers
- Full forward and backward pass implemented from scratch

## Usage

```python
from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Softmax
from network import Network
from train import SGD, train

model = Network([...])
optimizer = SGD(learning_rate=0.01, momentum=0.9)
history = train(model, optimizer, loss_fn, X_train, y_train, X_test, y_test)
```

---

## MNIST Results

Trained on 5,000 samples, tested on 1,000 samples:

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **90.94%** |
| Train Accuracy | 93.97% |
| Training Time | 5 epochs |
| Parameters | ~103,000 |

**Learning Curve:**

- Epoch 1: 72.29%
- Epoch 5: 90.94%

**Per-Digit Accuracy:**

- Best: Digit 4 (95.45%)
- Hardest: Digit 5 (83.91%)
- Average: 90.94%

---

## Author

**Adi Mendelowitz**
Machine Learning Engineer