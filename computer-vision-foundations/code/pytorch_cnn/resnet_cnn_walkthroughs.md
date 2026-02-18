# PyTorch & Deep Learning Masterclass — Complete Reference

> A senior ML researcher's walkthrough of PyTorch fundamentals, CNNs, and ResNet architecture.  
> Based on `cnn_pytorch.py` and `resnet.py` implementations.

**Date:** February 2026  
**Topics:** PyTorch fundamentals, Convolutional Neural Networks, ResNet, Backpropagation, Training loops

---

## Table of Contents

- [Part 1: CNN PyTorch - Foundations](#part-1-cnn-pytorch---foundations)
  - [Reproducibility](#reproducibility)
  - [nn.Module Contract](#nnmodule-contract)
  - [Convolutional Layers](#convolutional-layers)
  - [ReLU Activation](#relu-activation)
  - [MaxPool](#maxpool)
  - [Fully Connected Layers](#fully-connected-layers)
  - [The Forward Pass](#the-forward-pass)
  - [Training Loop](#training-loop)
  - [Evaluation Pattern](#evaluation-pattern)
  - [Data Loading](#data-loading)
- [Part 2: ResNet - Modern Architecture](#part-2-resnet---modern-architecture)
  - [The Problem ResNet Solved](#the-problem-resnet-solved)
  - [BasicBlock](#basicblock)
  - [BatchNorm](#batchnorm)
  - [Skip Connections](#skip-connections)
  - [ResNet Architecture](#resnet-architecture)
  - [Weight Initialization](#weight-initialization)
  - [Global Average Pooling](#global-average-pooling)
- [Critical Patterns](#critical-patterns)
- [Interview Prep](#interview-prep)

---

## PART 1: CNN PyTorch - Foundations

### Reproducibility

```python
torch.manual_seed(42)
np.random.seed(42)
# torch.backends.cudnn.deterministic = True
```

**Why This Matters:**

Neural network training involves randomness at multiple critical points:
- **Weight initialization**: Different random seeds → different starting points in the loss landscape
- **Data shuffling**: Different batch orderings → different gradient descent trajectories  
- **Dropout** (if used): Different neurons dropped → different training dynamics

Without fixing seeds, two "identical" training runs produce different final accuracies. This makes debugging impossible — you can't tell if a performance change came from your code modification or random variance.

**The GPU Caveat:**

cuDNN (CUDA Deep Neural Network library) uses non-deterministic algorithms by default for speed. Even with seeds fixed, GPU training can produce different results.

```python
torch.backends.cudnn.deterministic = True  # Forces deterministic algorithms
```

**When to use:**
- ✅ Publishing research (reproducibility requirement)
- ✅ Debugging (isolating real bugs from random variance)
- ❌ Production training (10-20% slower, speed matters more)

---

### nn.Module Contract

```python
class CNNPyTorch(nn.Module):
    def __init__(self):
        super(CNNPyTorch, self).__init__()
```

**`nn.Module` is not just inheritance — it's a contract with PyTorch's autograd system.**

**What you get:**

1. **Automatic parameter registration**  
   Any `nn.Layer` assigned as an attribute gets registered in `_modules` OrderedDict.

   ```python
   # This gets registered automatically:
   self.conv1 = nn.Conv2d(1, 8, 3)
   
   # This does NOT:
   conv_layer = nn.Conv2d(1, 8, 3)  # Not an attribute
   ```

2. **Training/eval mode switching**  
   `model.train()` and `model.eval()` recursively set flags on every module.
   - **BatchNorm**: Training uses batch stats, eval uses running stats
   - **Dropout**: Training randomly zeros neurons, eval passes all through

3. **Device management**  
   `model.to('cuda')` recursively moves all parameters and buffers to GPU.

4. **State persistence**  
   `model.state_dict()` returns OrderedDict for checkpointing.

**The Mandatory Handshake:**

```python
super(CNNPyTorch, self).__init__()
```

This boots up `nn.Module`'s internal parameter registry. **Without it, parameter registration silently fails** — `model.parameters()` returns empty, nothing trains. This bug wastes hours.

---

### Convolutional Layers

```python
self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
```

**What a Convolution Does Mathematically:**

1. You have 8 different 3×3 kernels (filters)
2. At each spatial position (i, j) in the input image:
   - Extract the 3×3 neighborhood centered at (i, j)
   - Compute dot product between this patch and kernel k
   - Store result in output feature map k at position (i, j)
3. Repeat for all 8 kernels → 8 output feature maps

**Why Convolution Instead of Fully Connected?**

| Property | Fully Connected | Convolutional |
|----------|----------------|---------------|
| **Parameters** | 28×28×128 = 100,352 | 3×3×1×8 = 72 |
| **Translation invariance** | No | Yes |
| **Spatial structure** | Ignores | Preserves |

**Parameter Sharing:**  
The same 72 weights are reused at every spatial position. This is why CNNs are efficient.

**Translation Invariance:**  
If a cat shifts 5 pixels right, the same neurons still fire — the network automatically generalizes.

**Padding Calculation:**

```
output_size = (input_size + 2*padding - kernel_size) / stride + 1
           = (28 + 2*1 - 3) / 1 + 1
           = 28 ✓
```

`padding=1` preserves spatial dimensions. Without it, each conv layer loses 1 pixel on each border. Stack 5 conv layers with `kernel_size=3, padding=0` and a 28×28 image shrinks to 18×18.

**Parameter Count:**

```
Parameters = (kernel_h × kernel_w × in_channels × out_channels) + out_channels
          = (3 × 3 × 1 × 8) + 8
          = 80
```

72 weights + 8 biases.

---

### ReLU Activation

```python
self.relu1 = nn.ReLU()
```

**ReLU: `f(x) = max(0, x)`**

**Why Nonlinearity is Mandatory:**

Without nonlinearity, deep networks collapse to shallow ones.

```python
# Two linear layers:
h = W1 @ x
y = W2 @ h = W2 @ (W1 @ x) = (W2 @ W1) @ x
```

You can always collapse multiple linear transformations into a single matrix multiply. A 100-layer network of linear layers is mathematically equivalent to 1 layer.

**Why ReLU Over Sigmoid/Tanh?**

**Sigmoid:** `σ(x) = 1 / (1 + e^(-x))`  
- For large |x|, gradient approaches 0 (saturation)
- In a 50-layer network: multiply 50 gradients all < 0.25 → gradient vanishes
- Early layers barely train

**ReLU:** `f(x) = max(0, x)`  
- For x > 0, gradient = 1 (no saturation)
- For x < 0, gradient = 0 ("dead ReLU" less severe than vanishing)
- Gradient flows cleanly through deep networks

**Historical Context:**

Neural networks existed since the 1980s but didn't work well until 2010s. Three key enablers:
1. GPUs for computation
2. Large datasets (ImageNet)
3. **ReLU replacing sigmoid**

Krizhevsky et al.'s AlexNet (2012) used ReLU and crushed ImageNet. Sigmoid networks of the same depth couldn't train.

---

### MaxPool

```python
self.pool1 = nn.MaxPool2d(kernel_size=2)
```

MaxPool slides a 2×2 window, takes the max value, shifts by stride=2 (default equals kernel_size).  
Input 28×28 → output 14×14. **Zero learnable parameters.**

**Two Purposes:**

1. **Translation Invariance**  
   If an edge detector fires at position (10, 10) or (10, 11), after max pooling they both contribute to the same output cell. Small shifts don't change the representation.

2. **Computational Efficiency**  
   After pooling: 4× fewer spatial positions. Next conv layer operates on 14×14 instead of 28×28 — 4× fewer computations.

**Modern Alternative:**

Strided convolutions: `Conv(stride=2)` instead of `Conv → Pool`. The conv layer learns the downsampling. Many recent architectures (ResNet for certain blocks, MobileNet) prefer this.

---

### Fully Connected Layers

```python
self.fc1 = nn.Linear(16 * 7 * 7, 128)
```

**This hardcoded `16 * 7 * 7 = 784` is fragile.**  
Change any earlier layer's stride or pooling → this number is wrong → runtime shape error.

**Why it's 784:**
- Start: 28×28, 1 channel
- After conv1 + pool1: 14×14, 8 channels
- After conv2 + pool2: 7×7, 16 channels
- Flatten: 7 × 7 × 16 = 784

**Better Pattern (ResNet uses this):**

```python
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Always outputs 1×1
self.fc = nn.Linear(512, num_classes)        # Works for any input size
```

`AdaptiveAvgPool2d((1, 1))` averages each channel's entire spatial map into a single value. Works regardless of input image resolution.

**Why Fully Connected Layers?**

CNNs extract spatial features hierarchically:
- Early layers: edges
- Middle layers: textures
- Late layers: object parts

Final classification isn't spatial — it's a global judgment: "this image contains a dog". Fully connected layers aggregate all spatial information into a class decision.

**Parameter Count:**

```python
# nn.Linear(784, 128)
Parameters = 784 × 128 + 128 = 100,480
```

Compare to `conv1`'s 80 parameters — fully connected layers are expensive.

---

### The Forward Pass

```python
x = torch.flatten(x, 1)
```

**`torch.flatten(x, 1)` vs `x.view(x.size(0), -1)` — What's the Difference?**

Both flatten all dimensions after the batch dimension:
- `x.view()` requires the tensor to be contiguous in memory. If you've transposed or sliced in certain ways, it fails.
- `torch.flatten()` handles non-contiguous tensors by copying if needed.

**Use `torch.flatten()` — newer, clearer, safer.**

**Shape Flow Through Network:**

```python
# Input:  (batch, 1, 28, 28)
x = self.conv1(x)    # (batch, 8, 28, 28)  - 8 feature maps
x = self.relu1(x)    # (batch, 8, 28, 28)  - nonlinearity, same shape
x = self.pool1(x)    # (batch, 8, 14, 14)  - spatial downsampling

x = self.conv2(x)    # (batch, 16, 14, 14) - 16 feature maps
x = self.relu2(x)    # (batch, 16, 14, 14)
x = self.pool2(x)    # (batch, 16, 7, 7)

x = torch.flatten(x, 1)  # (batch, 784)  - collapse spatial dims
x = self.fc1(x)      # (batch, 128)      - first classifier layer
x = self.relu3(x)    # (batch, 128)
x = self.fc2(x)      # (batch, 10)       - logits (raw scores)
```

**The Pattern:** Channels double (8→16) while spatial dims halve (28→14→7). This is the canonical CNN design.

---

### Training Loop

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
```

**CrossEntropyLoss = LogSoftmax + NLLLoss (Numerically Stable)**

Manual implementation (unstable):
```python
probs = torch.softmax(logits, dim=1)
loss = -torch.log(probs[range(batch_size), labels]).mean()
```

What PyTorch does (stable):
```python
log_probs = torch.log_softmax(logits, dim=1)  # log-sum-exp trick
loss = -log_probs[range(batch_size), labels].mean()
```

**Critical Rule: Never put softmax before CrossEntropyLoss**

```python
# ❌ WRONG - Double softmax
outputs = torch.softmax(logits, dim=1)
loss = criterion(outputs, labels)

# ✅ CORRECT - Raw logits
outputs = model(inputs)
loss = criterion(outputs, labels)
```

**SGD with Momentum:**

Standard SGD: `params -= lr * gradient`

Problems:
- Noisy gradients (each batch is a small random sample)
- Oscillates in narrow valleys of loss landscape
- Slow convergence

Momentum:
```python
velocity = 0.9 * velocity + gradient
params -= lr * velocity
```

Velocity is an exponential moving average of past gradients:
- Consistent gradient direction → velocity builds up, accelerates convergence
- Oscillating gradients → positive/negative components cancel, smooths path

`momentum=0.9` means 90% of previous velocity carries forward. Typical values: 0.9 or 0.99.

---

### The Critical Training Loop Pattern

```python
for inputs, labels in train_loader:
    optimizer.zero_grad()        # 1. Clear old gradients
    
    outputs = model(inputs)      # 2. Forward pass
    loss = criterion(outputs, labels)
    
    loss.backward()              # 3. Compute gradients
    optimizer.step()             # 4. Update weights
```

**This is the heartbeat of deep learning.** Every framework implements some variant of these four steps.

**1. `optimizer.zero_grad()`**

PyTorch accumulates gradients by default. Without zeroing, gradients from batch N contaminate batch N+1.

Why accumulate by default? Intentional design for gradient accumulation:

```python
optimizer.zero_grad()
for i in range(4):  # Simulate 4× larger batch
    outputs = model(inputs[i])
    loss = criterion(outputs, labels[i])
    loss.backward()  # Accumulate gradients
optimizer.step()  # Update once with accumulated gradients
```

This simulates batch size 256 on a GPU that only fits 64 samples.

**2. Forward Pass**

`outputs = model(inputs)` builds a computation graph tracking every operation. This graph makes autograd possible.

**3. `loss.backward()`**

Traverses the computation graph in reverse (backpropagation). For every parameter `p`, computes `∂loss/∂p` using the chain rule, stores in `p.grad`.

**The chain rule is why deep learning works:**
```
∂loss/∂w₁ = (∂loss/∂w₃) × (∂w₃/∂w₂) × (∂w₂/∂w₁)
```

PyTorch knows how to differentiate every primitive operation (matmul, ReLU, conv). It chains these derivatives automatically. **You never write a single derivative by hand.**

**4. `optimizer.step()`**

Uses computed gradients to update weights. For SGD with momentum:

```python
for param in model.parameters():
    if param.grad is None:
        continue
    velocity[param] = momentum * velocity[param] + param.grad
    param.data -= lr * velocity[param]
```

---

### Evaluation Pattern

```python
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
```

**Two Separate Mechanisms:**

**`model.eval()`**  
Sets a boolean flag on every module. Affects:
- **BatchNorm**: switches from batch statistics to running statistics
- **Dropout**: turns off (no random zeroing)

**`torch.no_grad()`**  
Disables autograd — PyTorch doesn't build computation graph.

Benefits:
- **Memory**: ~50% reduction (no intermediate activations stored)
- **Speed**: ~30% faster (no graph building overhead)

**Common Bug:** Forgetting `model.eval()`. Test accuracy will be different (usually worse) because BatchNorm uses wrong statistics.

**Always use both during evaluation.**

---

### Metrics Computation

```python
_, predicted = torch.max(outputs.data, 1)
train_correct += (predicted == labels).sum().item()
```

**`torch.max(outputs, dim=1)` returns `(values, indices)`**

For (batch, 10) output:
- `values`: (batch,) — max logit for each sample
- `indices`: (batch,) — which class had the max (0-9 for MNIST)

Discard values with `_`, keep indices (predicted labels).

**`(predicted == labels).sum().item()`**
- `predicted == labels`: (batch,) boolean tensor
- `.sum()`: counts True values → scalar tensor
- `.item()`: extracts Python int, **detaches from graph**

**Why `.item()`?**

Without it: `train_loss += loss` keeps the entire computation graph in memory for every batch. After 100 batches → out of memory crash.

`.item()` extracts the value and discards the graph. **Always use it for metrics logging.**

---

### Data Loading

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

**`transforms.ToTensor()` does three things:**

1. Converts PIL Image (H, W, C) to tensor (C, H, W) — PyTorch uses channels-first
2. Converts uint8 [0, 255] to float32 [0, 1] by dividing by 255
3. Copies data (non-intrusive)

**`transforms.Normalize((0.1307,), (0.3081,))`**

Applies `(x - mean) / std` per channel.

For MNIST (1 channel):
- Mean = 0.1307
- Std = 0.3081

These are empirical statistics of the entire MNIST training set.

**Why Normalize?**

1. **Gradient scale**: If inputs span [0, 255], gradients for early layers are huge. Normalized inputs → stable gradients.
2. **Weight initialization**: Kaiming/Xavier initialization assumes zero-mean, unit-variance inputs.

**For new datasets, compute mean/std:**

```python
mean = train_data.mean(dim=(0, 2, 3))
std = train_data.std(dim=(0, 2, 3))
```

---

### DataLoader

```python
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)
```

**Why Batch Training?**

| Aspect | Single Sample | Batch (64) |
|--------|--------------|------------|
| **Gradient estimate** | Extremely noisy | More stable |
| **Computation** | Cannot parallelize | Leverages GPU parallelism |
| **Convergence** | Slower, erratic | Faster, smoother |

**Why Not Use Entire Dataset as One Batch?**

**Generalization!**

Large batches converge to **sharp minima** (narrow valleys in loss landscape). Sharp minima generalize poorly — slight distribution shift and loss spikes.

Small batches add noise → implicit regularization → **flat minima**. Flat minima are robust.

**Research (Keskar et al., 2016):** Large batch training achieves lower training loss but worse test accuracy.

**Typical batch sizes:** 32-256 for most tasks.

**`shuffle=True` for training, `False` for test:**

Shuffling prevents the model from learning batch order (e.g., all 0s, then all 1s). Test set is never shuffled — you want deterministic, comparable results.

---

### Sample Predictions

```python
confidence = torch.softmax(outputs[i], dim=0)[predicted_label].item()
```

**Here softmax is correct** — converting logits to probabilities for human interpretation, not for loss computation.

**Important Caveat:**

Neural network softmax probabilities are **overconfident by default**. A network outputting `[0.99, 0.01, ...]` isn't truly 99% confident — it learned to push probabilities to extremes during training (minimizing cross-entropy encourages this).

For calibrated confidence, use temperature scaling or other calibration techniques.

---

---

## PART 2: ResNet - Modern Architecture

### The Problem ResNet Solved

**Before 2015: deeper networks performed *worse* than shallower ones.**

Not overfitting — pure optimization failure. Gradients vanished as they propagated backward through 20+ layers.

```
Multiply: 0.1 × 0.1 × ... × 0.1 (twenty times) → effectively zero
```

**ResNet's Insight (He et al., 2015):**

Don't learn `H(x)` directly. Learn the **residual** `F(x) = H(x) - x`.

The network learns what to **add** to the input, not the full transformation.

**Why This Helps:**

During backpropagation, gradients flow through two paths:

1. **Main path** (through learned layers): gradients can vanish
2. **Skip connection** (identity): gradients flow directly, **undiminished**

Even if main path gradients vanish, the skip connection provides a clean **gradient highway** to early layers.

**Result:** ResNet-152 (152 layers) trains successfully and outperforms ResNet-18.

---

### BasicBlock

```python
self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                       stride=stride, padding=1, bias=False)
self.bn1 = nn.BatchNorm2d(out_channels)
```

**`bias=False` — Why?**

BatchNorm immediately follows:
```python
y = γ * (x - μ) / σ + β
```

Where:
- `γ` (gamma): learnable scale
- `β` (beta): learnable shift
- `μ`, `σ`: batch mean and std

The `β` parameter is functionally identical to a bias term. Having both `conv.bias` and `bn.beta` wastes parameters (one will learn to be zero).

**This is a standard optimization** in every modern architecture.

---

### BatchNorm

```python
self.bn1 = nn.BatchNorm2d(out_channels)
```

**BatchNorm Does Three Things:**

**During Training:**
```python
μ_batch = x.mean(dim=(0, 2, 3))  # Mean per channel across batch and spatial dims
σ_batch = x.std(dim=(0, 2, 3))   # Std per channel
x_normalized = (x - μ_batch) / (σ_batch + ε)
output = γ * x_normalized + β
```

**During Evaluation:**
```python
x_normalized = (x - μ_running) / (σ_running + ε)
output = γ * x_normalized + β
```

`μ_running` and `σ_running` are exponential moving averages:
```python
μ_running = 0.9 * μ_running + 0.1 * μ_batch
```

**Why This Helps:**

1. **Reduces internal covariate shift**: As weights update, input distributions to each layer shift. BatchNorm stabilizes these distributions.

2. **Allows higher learning rates**: Normalization prevents activations from exploding/vanishing.

3. **Acts as regularization**: Noise from using batch statistics during training adds randomness, reducing overfitting.

**Critical Bug Prevention:**

Always call `model.eval()` before evaluation. Otherwise BatchNorm uses batch statistics (different for small test batches) instead of running statistics → inconsistent results.

---

### Skip Connections

```python
self.shortcut = nn.Sequential()
if stride != 1 or in_channels != out_channels:
    self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    )
```

**Goal:** Add input `x` to output `F(x)`. For this to work, both must have identical shape.

**Two Cases Break Shape Matching:**

1. **Spatial mismatch (`stride != 1`)**: Main path downsamples (56×56 → 28×28), input is still 56×56.

2. **Channel mismatch (`in_channels != out_channels`)**: Main path goes 64→128 channels, input is 64.

**Solution: 1×1 Projection Convolution**

A 1×1 conv changes channel count without spatial reasoning:
- **Kernel size = 1**: No spatial context, just per-pixel linear transformation
- **Stride = stride**: Matches main path's downsampling
- **Output channels = out_channels**: Matches main path's channel count

This is called a **"projection shortcut"** — it projects input into the correct dimensional space.

**When is it Identity?**

When `stride=1` and `in_channels == out_channels`, shortcut is empty `nn.Sequential()` = `f(x) = x`. Zero parameters, pure passthrough.

**Example Flow:**
- `layer1`: 64→64 channels, stride=1 → **identity shortcuts**
- `layer2` first block: 64→128, stride=2 → **projection shortcut**
- `layer2` remaining blocks: 128→128, stride=1 → **identity shortcuts**

---

### Forward Pass — Pre-activation vs Post-activation

```python
def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    
    out = self.conv2(out)
    out = self.bn2(out)
    
    out += self.shortcut(x)  # Add before final ReLU
    out = self.relu(out)
    return out
```

**Original ResNet Ordering:**
- ReLU after first conv
- **No ReLU after second conv**
- Addition
- ReLU after addition

**Why ReLU After Addition?**

The residual `F(x)` can be negative (learning to subtract features). If ReLU applied before adding, you'd lose this expressiveness.

ReLU after addition allows both positive and negative residuals, then applies nonlinearity to the combined signal.

**Modern Variant (Pre-activation ResNet):**

```python
out = self.bn1(x)
out = self.relu(out)
out = self.conv1(out)  # BN-ReLU-Conv

out = self.bn2(out)
out = self.relu(out)
out = self.conv2(out)

return out + self.shortcut(x)  # No final ReLU
```

Research showed this trains slightly better for very deep networks (100+ layers).

---

### ResNet Architecture

```python
self.current_channels = 64
```

**Stateful counter** for building variable-depth networks. As you construct layers, this tracks current channel count. `_make_layer` modifies it, next call reads the updated value.

**Initial Convolution:**

```python
self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
```

**Two aggressive downsampling steps:**  
224×224 → 112×112 (conv) → 56×56 (pool)

Reduces spatial resolution by 16× before residual blocks. Why?

1. **Computational efficiency**: Expensive residual layers operate on smaller spatial maps.

2. **Receptive field**: 7×7 kernel at original resolution captures textures, edges, large-scale structure.

**Padding Math:**
```
(224 + 2*3 - 7) / 2 + 1 = 112 ✓
(112 + 2*1 - 3) / 2 + 1 = 56 ✓
```

---

### Layer Structure — Geometric Progression

```python
self.layer1 = self._make_layer(block, 64, layers[0], stride=1)   # 56×56, 64ch
self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 28×28, 128ch
self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 14×14, 256ch
self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 7×7, 512ch
```

**The Pattern is Intentional:**

At each layer:
- Spatial dimensions halve (4× fewer positions)
- Channels double (2× more)

**Result:** Computational cost stays roughly constant.

**FLOPs for 3×3 conv:**
```
FLOPs ≈ H × W × C_in × C_out × 9
```

If H and W halve while C doubles:
```
(H/2) × (W/2) × (2C) × (2C) × 9 = H × W × C × C × 9
```

Same computational cost per layer.

---

### _make_layer — Builder Pattern

```python
def _make_layer(self, block, out_channels, num_blocks, stride):
    layers = [block(self.current_channels, out_channels, stride)]
    self.current_channels = out_channels
    
    for _ in range(1, num_blocks):
        layers.append(block(self.current_channels, out_channels, stride=1))
    
    return nn.Sequential(*layers)
```

**First block is special** — handles channel transition and downsampling.  
**Remaining blocks are uniform** — same channels, no downsampling.

**Example for `layer2` with `[2, 2, 2, 2]` config:**
- Block 1: 64→128 channels, stride=2 (projection shortcut, downsampling)
- Block 2: 128→128 channels, stride=1 (identity shortcut)

**`nn.Sequential(*layers)`** unpacks the list. During forward pass, Sequential calls each block in order.

---

### Weight Initialization

```python
def _initialize_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
```

**Kaiming (He) Initialization — Designed for ReLU Networks**

For a layer with `fan_out` outputs, initialize weights from:
```
N(0, sqrt(2 / fan_out))
```

**Why?**

ReLU zeros out half the neurons (negative activations → 0). This reduces output variance by half. To maintain consistent signal magnitude through the network, scale up initial weights by sqrt(2).

**What Happens with Wrong Initialization?**

1. **Too small (N(0, 0.01))**: Activations shrink through layers → near-zero outputs → gradients vanish.

2. **Too large (N(0, 1))**: Activations explode → gradients explode → training diverges.

3. **Xavier (designed for tanh)**: Assumes symmetric activations. ReLU violates this (negative → 0) → shrinking activations.

**Kaiming initialization ensures:** At initialization, activation and gradient magnitudes stay roughly constant across layers. Gives optimizer a good starting point.

**BatchNorm Initialized to Identity:**

```python
elif isinstance(module, nn.BatchNorm2d):
    nn.init.constant_(module.weight, 1)
    nn.init.constant_(module.bias, 0)
```

With `γ=1, β=0`, BatchNorm starts as pure normalization: `output = (x - μ) / σ`. As training proceeds, `γ` and `β ` are learned.

---

### Global Average Pooling

```python
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
self.fc = nn.Linear(512, num_classes)
```

**`AdaptiveAvgPool2d((1, 1))` outputs 1×1 spatial map regardless of input size.**

Input: `(batch, 512, 7, 7)` → Output: `(batch, 512, 1, 1)`  
Input: `(batch, 512, 14, 14)` → Output: `(batch, 512, 1, 1)`

**How?** Computes average over entire spatial extent of each channel.

**Why This Matters:**

Your CNN hardcodes `16 * 7 * 7`. If input is 56×56 instead of 28×28, the math breaks.

ResNet works on **any input resolution** because `AdaptiveAvgPool2d` collapses spatial dims to 1×1 before linear layer. Train on 224×224, test on 448×448 — no architecture modification needed.

---

### Factory Functions

```python
def resnet18(num_classes=1000, in_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)
```

**API Design:** Users call `resnet18()`, not `ResNet(BasicBlock, [2, 2, 2, 2], ...)`.

If you refactor internals, public API stays stable.

**ResNet-18 Layer Count:**

1 (initial conv) + 2×2 (layer1) + 2×2 (layer2) + 2×2 (layer3) + 2×2 (layer4) + 1 (fc) = **18 weight layers**

---

### Testing — Shape Assertions

```python
assert out_block.shape == x_block.shape
```

**Why Shape Assertions Matter:**

Most bugs in deep learning are silent shape errors. A network expecting `(batch, 128, 7, 7)` but receiving `(batch, 128, 14, 14)` won't crash — it produces wrong results.

**Professional Practice:**
- Add shape assertions at every layer transition during development
- Remove in production for speed

---

## Critical Patterns

### The Standard Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
```

### Common Bugs to Avoid

❌ **Putting softmax before CrossEntropyLoss**
```python
outputs = torch.softmax(logits, dim=1)  # Wrong!
loss = criterion(outputs, labels)
```

❌ **Forgetting to zero gradients**
```python
outputs = model(inputs)  # Gradients accumulate!
loss.backward()
optimizer.step()
```

❌ **Forgetting model.eval() during evaluation**
```python
with torch.no_grad():
    outputs = model(inputs)  # BatchNorm uses wrong stats!
```

❌ **Not using .item() for metrics**
```python
train_loss += loss  # Memory leak!
```

### Design Patterns to Follow

✅ **Shape comments in forward**
```python
x = self.conv1(x)  # (batch, 8, 28, 28)
```

✅ **Type hints**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
```

✅ **Factory functions for models**
```python
def resnet18(num_classes=1000) -> ResNet:
```

✅ **Adaptive pooling for flexibility**
```python
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
```

---

## Interview Prep

### Questions You Should Be Ready to Answer

**1. Why do skip connections solve vanishing gradients?**

Gradients flow through two paths: the main path (through learned layers) and the skip connection (identity). Even if main path gradients vanish, the skip connection provides a clean gradient highway to early layers.

**2. Why does BatchNorm help training?**

Three reasons:
- Reduces internal covariate shift (stabilizes layer input distributions)
- Allows higher learning rates (prevents activation explosion/vanishing)
- Acts as regularization (batch statistics add noise)

**3. What's the tradeoff between batch size and generalization?**

Large batches converge to sharp minima (narrow valleys) that generalize poorly. Small batches add noise → implicit regularization → flat minima that are more robust to distribution shift.

**4. Why are 1×1 convolutions useful?**

- Dimension matching in skip connections (no spatial reasoning needed)
- Cheap channel mixing (change channel count without spatial computation)
- Network-in-network (add nonlinearity without changing spatial dims)

**5. AdaptiveAvgPool vs hardcoded flatten?**

AdaptiveAvgPool outputs fixed spatial size (e.g., 1×1) regardless of input resolution → network works on any image size. Hardcoded flatten breaks if input resolution changes.

**6. Why ReLU over sigmoid?**

Sigmoid saturates (gradient → 0 for large |x|). In deep networks, multiplying many near-zero gradients causes vanishing gradients. ReLU has gradient=1 for positive inputs → gradients flow cleanly.

**7. Why normalize inputs?**

- Stable gradients (unnormalized [0, 255] inputs → huge gradients in early layers)
- Matches weight initialization assumptions (Kaiming/Xavier assume zero-mean, unit-variance)

**8. What does .item() do and why use it?**

Extracts Python scalar and detaches from computation graph. Without it, accumulating tensors keeps entire computation history in memory → memory leak.

---

## Quick Reference

### Common Shapes in ResNet-18 (ImageNet)

| Layer | Input | Output |
|-------|-------|--------|
| Input | (batch, 3, 224, 224) | |
| conv1 | (batch, 3, 224, 224) | (batch, 64, 112, 112) |
| maxpool | (batch, 64, 112, 112) | (batch, 64, 56, 56) |
| layer1 | (batch, 64, 56, 56) | (batch, 64, 56, 56) |
| layer2 | (batch, 64, 56, 56) | (batch, 128, 28, 28) |
| layer3 | (batch, 128, 28, 28) | (batch, 256, 14, 14) |
| layer4 | (batch, 256, 14, 14) | (batch, 512, 7, 7) |
| avgpool | (batch, 512, 7, 7) | (batch, 512, 1, 1) |
| fc | (batch, 512) | (batch, 1000) |

### Parameter Counts

- **ResNet-18**: ~11.7M parameters
- **ResNet-34**: ~21.8M parameters
- **ResNet-50**: ~25.6M parameters

### Typical Hyperparameters

- **Batch size**: 32-256
- **Learning rate**: 0.001-0.1 (with warmup and decay)
- **Momentum**: 0.9
- **Weight decay**: 1e-4 to 5e-4
- **Epochs**: 90-200 for ImageNet

---

## References

**ResNet Paper:**
- He, Kaiming, et al. "Deep residual learning for image recognition." CVPR 2016.
- https://arxiv.org/abs/1512.03385

**Batch Normalization:**
- Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." ICML 2015.

**Large Batch Training:**
- Keskar, Nitish Shirish, et al. "On large-batch training for deep learning: Generalization gap and sharp minima." ICLR 2017.

**Weight Initialization:**
- He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." ICCV 2015.

---

**End of Reference Document**