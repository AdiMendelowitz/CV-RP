# CNN from Scratch - NumPy Only

A complete Convolutional Neural Network implementation using only NumPy, built to understand deep learning fundamentals.

## Features
- Full backpropagation through convolutional and dense layers
- He weight initialization
- SGD optimizer with momentum
- Achieves 100% accuracy on line orientation classification

## Results
Trained on 600 14×14 images (vertical/horizontal/diagonal lines):
- **Training Accuracy:** 100%
- **Test Accuracy:** 100%
- Converges in 2 epochs

## Architecture
```
Conv2D(1→8) → ReLU → MaxPool2D → 
Conv2D(8→16) → ReLU → MaxPool2D → 
Flatten → Dense(144→32) → ReLU → Dense(32→3) → Softmax
```

## Implementation Highlights
- No deep learning libraries used (PyTorch, TensorFlow)
- Manual gradient computation for all layers
- Educational purpose: understand neural networks from first principles

## Usage
```python
from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Softmax
from network import Network
from train import SGD, train

model = Network([...])
optimizer = SGD(learning_rate=0.01, momentum=0.9)
history = train(model, optimizer, loss_fn, X_train, y_train, X_test, y_test)
```


## MNIST Results (Real Handwritten Digits)

Trained on 5,000 samples, tested on 1,000 samples:

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **90.94%** |
| Train Accuracy | 93.97% |
| Training Time | 5 epochs |
| Parameters | ~13,000 |

**Learning Curve:**
- Epoch 1: 72.29%
- Epoch 5: 90.94%

**Per-Digit Accuracy:**
- Best: Digit 4 (95.45%)
- Hardest: Digit 5 (83.91%)
- Average: 90.94%

Built with **NumPy only** - no deep learning frameworks!

----------------------
## 👤 Author

**Adi Mendelowitz**  
Machine Learning Engineer  
Specialization: Computer Vision & Image Processing