# ResNet Theory – Why Residual Connections Solve Vanishing Gradients

**Topic:** Mathematical and practical explanation of how skip connections enable deep network training  
**Status:** Completed

---

## The Vanishing Gradient Problem

In deep neural networks (dozens of layers or more), gradients can become exponentially smaller as they propagate backward, especially with saturating nonlinearities (sigmoid, tanh).

During backpropagation, gradients are computed via the chain rule:

```text
∂L/∂w₁ = (∂L/∂w_n) × (∂w_n/∂w_{n-1}) × ... × (∂w₂/∂w₁)
```

For a network with sigmoid activations:

- Sigmoid derivative: `sigma'(x) = sigma(x)(1 - sigma(x))`.  
- Maximum value: `sigma'(0) = 0.25`.  
- For large |x|, `sigma'(x) ~= 0` (saturation).

**Example: 20‑layer network with sigmoids**

```text
gradient ≈ 0.25 × 0.25 × ... × 0.25   (20 times)
        = 0.25²⁰
        ≈ 9 × 10⁻¹³   # essentially zero
```

Even if weights are well‑behaved, multiplying many derivatives in (0, 1) drives the gradient magnitude toward zero.

### Consequences

1. **Early layers barely learn.** Gradients arriving at first layers are near zero, leading to very slow updates.  
2. **Optimization failure.** Networks can have enough capacity but fail to fit even the training set well.  
3. **Deeper ≠ better (pre‑ResNet).** Simply stacking more layers can increase training error, not just test error.

### Pre‑ResNet Partial Solutions

- **ReLU activations:** derivative is 1 for positive inputs (no saturation in that region).  
- **Batch Normalization:** stabilises activations and gradients, reducing internal covariate shift.  
- **Careful initialization (Xavier/He):** keeps variance of activations/gradients roughly constant at start.

These helped but did not fully solve training very deep plain networks (e.g., 50+ layers) on ImageNet or CIFAR‑10.

---

## How Residual Connections Help

### Core Idea

Instead of learning a mapping `H(x)` directly, a residual block learns a **residual** function:

```text
F(x) = H(x) - x
```

and the block outputs:

```python
# Standard block (plain network)
output = H(x)

# Residual block
output = F(x) + x
```

Here, `F(x)` is the learned residual (typically 2–3 conv–BN–ReLU layers), and `x` is the skip/identity connection.

### The Gradient Highway

Forward pass for one residual unit:

```text
y = F(x, W) + x
```

Backward:

```text
∂L/∂x = ∂L/∂y · (∂F/∂x + I)
      = ∂L/∂y · ∂F/∂x + ∂L/∂y
```

**Key insight:** The term `d L / d y` flows directly through the identity addition (the `+ x`) and does not get multiplied by small derivatives in F. Even if `d F / d x` is small or ill‑conditioned, the additive identity term provides a robust gradient path.

### Layer‑wise Formulation

Consider a sequence of residual blocks:

```text
x_{l+1} = x_l + F_l(x_l, W_l)
```

Backward at layer `l`:

```text
∂L/∂x_l = ∂L/∂x_{l+1} · (I + ∂F_l/∂x_l)
```

Stacking across L blocks:

```text
∂L/∂x_0 = ∂L/∂x_L · Π_{l=0}^{L-1} (I + ∂F_l/∂x_l)
```

Contrast:

- Plain network: `d L/ d x_0 = d L/d x_L * prod_l d H_l / d x_l`, a pure product of Jacobians (easy to vanish or explode).  
- Residual network: each factor is `I + J_l`, where `J_l = d F_l / d x_l`.

If residual branches are small perturbations (‖`J_l`‖ not too large), then eigenvalues of `I + J_l` cluster around 1, making gradient norms more stable across depth. In the extreme case when `F_l ~= 0`, we have `I + d F_l / d x_l ~= I`, so:

```text
∂L/∂x_0 ≈ ∂L/∂x_L
```

This is the “gradient highway”: adding identity paths mitigates vanishing gradients even when the residual branch itself is poorly conditioned.

### Visual Intuition

```text
Plain network (20 layers):
Input → Layer1 → Layer2 → ... → Layer20 → Output
       (×0.3)   (×0.3)           (×0.3)   → gradient ≈ 0.3²⁰ → ~0

ResNet (20 residual blocks):
Input → [F₁ + identity] → [F₂ + identity] → ... → [F₂₀ + identity] → Output
          ↘ grad ↖          ↘ grad ↖                        ↘ grad ↖

Skip paths give a near‑constant gradient route from output to early layers.
```

---

## Intuitive Explanation

### Learning Incremental Changes

**Plain network:**  

- Each layer must contribute to the full mapping from input to output.  
- It is hard to optimise, because layers interact in complicated ways.  

**Residual network:**  

- Each block learns a small *correction* to the current representation: “what should I add to x?”.  
- If a block is unnecessary, it can approximate `F(x) ~= 0`, effectively becoming an identity mapping.  
- This makes optimisation easier: the identity function is always in the function class and easy to realise by pushing residual weights toward zero.

### Identity Hypothesis (He et al.)

The ResNet paper observes:

> If the added layers can be constructed as identity mappings, a deeper model should produce no higher training error than its shallower counterpart.

Interpretation:

- Start with a shallower network (e.g., 18 layers).  
- Add additional residual blocks (e.g., to reach 34 layers).  
- In the worst case, those extra blocks learn identity (F ≈ 0), so the deep network can match the training error of the shallow one.  
- In practice, deeper residual nets yield lower training and test error.

Without skips, coaxing a stack of layers to implement exact identity is non‑trivial; with residuals, identity corresponds simply to “learn F ≈ 0.”

---

## Experimental Evidence

### CIFAR‑10 (from the ResNet paper)

On CIFAR‑10, He et al. show that deeper plain networks can be *worse* than shallower ones, whereas deeper ResNets improve:

- A 56‑layer **plain** CNN has higher training and test error than a 20‑layer plain CNN (optimisation difficulty).  
- Corresponding 56‑layer **ResNet** achieves substantially lower training and test error than its 20‑layer ResNet counterpart.  

The key observation is that residual connections allow increasing depth to reduce training error, whereas plain nets with the same depth get stuck at higher training error.

### ImageNet (ILSVRC 2015)

- ResNet‑152 (152 layers) achieves **3.57% top‑5 error** on ImageNet test set using an ensemble, winning ILSVRC 2015 classification.  
- This was about 8× deeper than VGG‑19 yet with lower FLOPs and significantly better accuracy.

This demonstrates that residual connections make it possible to train very deep networks in practice.

---

## Implementation Details

### Basic Residual Block (Post‑activation ResNet‑v1 style)

```python
def forward(self, x):
    # Main path: F(x)
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    # Skip connection: add identity or projection
    out += self.shortcut(x)

    # Activation after addition
    out = self.relu(out)
    return out
```

- When `self.shortcut` is identity, this implements `x + F(x)`.  
- When shapes differ (e.g., stride 2 or channel mismatch), a 1×1 conv + BN projection is used to align dimensions.

### Identity vs Projection Shortcuts

**Identity shortcut (no change in shape):**

```python
if stride == 1 and in_channels == out_channels:
    self.shortcut = nn.Sequential()   # returns x unchanged
```

**Projection shortcut (when downsampling or channel changes):**

```python
else:
    self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1,
                  stride=stride, bias=False),
        nn.BatchNorm2d(out_channels)
    )
```

The projection introduces weights on the skip path, but still provides a relatively direct gradient route; He et al. show that identity skips with after‑addition activation work best when feasible.

---

## Why ReLU Alone Was Not Enough

ReLU avoids saturation on positive inputs, but gradients still traverse **weight matrices** and nonlinearities:

```text
∂L/∂x₁ = ∂L/∂x₂ · W₂ · ReLU'(z₁)
```

Even with ReLU'(z₁) = 1, small weights or ill‑conditioned weight matrices can shrink the gradient. Over many layers, the product of weight matrices can still yield vanishing or exploding gradients.

Residual connections change the structure:

```text
x₂ = x₁ + F(x₁; W₂)
∂L/∂x₁ = ∂L/∂x₂ · (I + ∂F/∂x₁)
```

Even if `W_2` and `d F / d x_1` are small, the identity term `I` ensures a baseline gradient path. This is why residual connections plus ReLU and BN enable much deeper networks than ReLU alone.

---

## Practical Implications

### 1. Network Depth

**Before ResNet (≈2015):**  

- ConvNet depth for classification typically ≤ 20–30 layers (e.g., VGG‑19); deeper plain nets were very hard to train.

**After ResNet:**  

- ResNet‑50, ‑101, ‑152 and even 1,001‑layer variants (on CIFAR) are trainable.  
- Depth became a tunable hyperparameter rather than a hard barrier.

### 2. Training Dynamics

Residual networks:

- Optimise more easily and reach lower training error for the same depth.  
- Are less sensitive to learning rate choices (within reasonable ranges).  
- Show smoother, “more convex‑like” loss landscapes when visualised, compared to plain nets of similar depth.

Li et al. (2018) explicitly show that ResNet‑style skip connections make the loss surface wider and less chaotic than VGG‑style architectures, which correlates with better optimisation and generalisation.

### 3. Transfer Learning

ResNet backbones have been widely adopted for:

- Object detection (Faster R‑CNN, Mask R‑CNN).  
- Semantic segmentation (DeepLab, FCN variants).  
- Instance segmentation and other downstream CV tasks.

Deep, robust feature extractors enabled by residual connections became standard starting points for many vision pipelines.

---

## Common Misconceptions

### “Skip connections just copy features.”

**Reality:** They do not just copy features; they **add** the learned residual to the input. The skip path gives a clean gradient route, but the main branch still transforms features; the network learns when and how to modify or preserve information.

### “Skip connections always help.”

**Reality:** Benefits are most pronounced when depth is large enough that optimisation is a bottleneck. For very shallow networks (e.g., 5–10 layers), residual connections bring limited gains and sometimes unnecessary complexity.

### “Identity mappings don’t learn anything.”

**Reality:** Learning `F(x) ~= 0` is still a **learned decision** that certain layers should pass features through; the network uses this flexibility to allocate representational capacity where it’s most useful.

---

## Advanced Topics

### Pre‑activation ResNet (He et al., 2016)

The follow‑up paper “Identity Mappings in Deep Residual Networks” proposes moving BN and ReLU **before** convolutions (pre‑activation):

- Original (post‑activation): `Conv → BN → ReLU → Conv → BN → Add → ReLU`.  
- Pre‑activation: `BN → ReLU → Conv → BN → ReLU → Conv → Add`.

Benefits:

- Cleaner forward/backward paths (activation is not between addition and identity).  
- Better performance for very deep nets (e.g., 1001‑layer ResNets on CIFAR‑10).

### ResNeXt (Aggregated Residual Transformations)

ResNeXt adds **grouped/parallel** paths within each residual block, increasing “cardinality”:

```text
x → [Path 1] → 
  → [Path 2] →  Σ → + identity → output
  → [Path 3] →
```

### DenseNet (Dense Skip Connections)

DenseNets connect each layer to all subsequent layers via concatenation:

```text
x₀ → x₁ → x₂ → x₃
 ↓    ↓    ↓
 └────┴────┴──→ concat [x₀, x₁, x₂, x₃]
```

This creates many gradient and information paths, at the cost of higher memory usage.

---

## Key Takeaways

- **Vanishing gradients** arise from repeated multiplication of Jacobians with singular values < 1 across many layers.  
- **Residual/skip connections** change the recurrence to include an identity term: gradients propagate via both the residual branch and a direct path.  
- Even when the residual branch contributes small or unstable derivatives, the identity term keeps gradients from collapsing completely.  
- This yields **easier optimisation**, enabling very deep networks (50–150+ layers) with strong empirical performance.  
- Residual connections are not a minor tweak; they effectively restructure the optimisation landscape and have shaped modern backbone design.

---

## Further Reading

1. **Deep Residual Learning for Image Recognition** – He et al., 2015.  
   Introduces ResNet, residual blocks, and shows deep networks (up to 152 layers) on ImageNet and 100/1000‑layer variants on CIFAR‑10.

2. **Identity Mappings in Deep Residual Networks** – He et al., 2016.  
   Analyses propagation in residual blocks, motivates pre‑activation units, and reports 1001‑layer ResNets.

3. **Visualizing the Loss Landscape of Neural Nets** – Li et al., 2018.  
   Shows that skip connections (ResNet, DenseNet) significantly “smooth” the loss landscape compared to plain VGG‑style nets.

4. Related architectures:  
   - **Highway Networks** (Srivastava et al., 2015) – gated skip connections.  
   - **DenseNet** (Huang et al., 2017) – dense concatenated connections.  
   - **ResNeXt** (Xie et al., 2017) – aggregated residual transformations.
