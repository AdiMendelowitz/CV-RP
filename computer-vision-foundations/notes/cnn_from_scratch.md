# CNN from Scratch: NumPy Only

A complete Convolutional Neural Network implementation using only NumPy, built to understand deep learning from first principles.[file:130]

## Features

- Full backpropagation through convolutional and dense layers (no autograd).[file:130]
- He (Kaiming) weight initialization for layers with ReLU activations.[file:130][web:63]
- SGD optimizer with momentum (classical heavy‑ball momentum).[file:130]
- Achieves 100% accuracy on a 3‑class synthetic line‑orientation dataset (vertical, horizontal, diagonal) under the experimental setup described below.[file:130]

---

## Line Classification Results

Trained on 600 synthetic 14×14 grayscale images (3 classes: vertical, horizontal, diagonal lines):[file:130]

- **Test accuracy:** 100%  
- **Train accuracy:** 100%  
- Converges in 2 epochs

Given the simplicity and separability of the patterns, perfect accuracy on a held‑out synthetic set is realistic for a small CNN with well‑tuned training.[file:130]

### Line Classification Architecture

```text
Conv2D(1 -> 8) -> ReLU -> MaxPool2D ->
Conv2D(8 -> 16) -> ReLU -> MaxPool2D ->
Flatten -> Dense(144 -> 32) -> ReLU -> Dense(32 -> 3) -> Softmax
```

- 1 input channel (grayscale 14×14).  
- Two conv–ReLU–pool stages extract increasingly abstract features and reduce spatial size.  
- A small MLP head maps flattened features to 3 logits, followed by softmax for classification.[file:130]

This network targets 3 output classes. The MNIST adaptation reuses the same backbone idea but changes the final dense layer for 10 classes and increases hidden dimensionality.[file:130]

---

## Implementation Highlights

- No deep learning libraries (PyTorch, TensorFlow, JAX, etc.). All tensors and operations are NumPy arrays and functions.[file:130]
- Manual gradient computation for all layers, including Conv2D, MaxPool2D (with argmax/indices for backward), Dense, and Softmax + cross‑entropy.[file:130]
- Full forward and backward passes implemented from scratch to expose all intermediate shapes and operations.[file:130]

This implementation is suitable as a correctness‑oriented reference for understanding gradient flow through convolutional networks, not as an optimised production framework.

---

## Usage

```python
from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Softmax
from network import Network
from train import SGD, train

model = Network([...])
optimizer = SGD(learning_rate=0.01, momentum=0.9)
history = train(model, optimizer, loss_fn, X_train, y_train, X_test, y_test)
```

- `Network([...])` composes layers into a simple sequential CNN.  
- `SGD` implements mini‑batch SGD with momentum (β ≈ 0.9).  
- `train` handles epochs, mini‑batch iteration, forward/backward passes, and accuracy logging.[file:130]

---

## MNIST Results

Trained on a subset of MNIST:[file:130][web:63]

- **Training set:** 5,000 samples  
- **Test set:** 1,000 samples  

| Metric          | Value      |
|-----------------|-----------:|
| **Test accuracy**  | **90.94%** |
| Train accuracy  | 93.97%     |
| Training time   | 5 epochs   |
| Parameters      | ~103,000   |

On full MNIST (60k/10k) and with more epochs, CNNs of similar size typically reach ≈98–99% accuracy using optimised frameworks and training recipes; your 90.94% on 5k/1k with a pure‑NumPy implementation and 5 epochs is therefore a reasonable result for an educational implementation.[file:130][web:63]

**Learning curve (test accuracy by epoch):**[file:130]

- Epoch 1: 72.29%  
- Epoch 5: 90.94%

**Per‑digit accuracy:**[file:130]

- Best: digit **4** — 95.45%  
- Hardest: digit **5** — 83.91%  
- Overall average: 90.94%

These class‑wise numbers are consistent with the general observation that some MNIST digits (e.g., 5) are more confusable than others (e.g., 0 or 4), especially with limited data and training time.[web:63]

---

## Author

**Adi Mendelowitz**  
Machine Learning Engineer / Researcher (NumPy CNN implementation and experiments).[file:130]
