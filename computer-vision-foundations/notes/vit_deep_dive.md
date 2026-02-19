# Vision Transformer (ViT) Deep Dive

**Reference Implementation:** `vit.py`  
**Paper:** "An Image is Worth 16x16 Words" (Dosovskiy et al., 2020)

---

## Table of Contents

- [The Paradigm Shift](#the-paradigm-shift)
- [Component 1: PatchEmbedding](#component-1-patchembed)
- [Component 2: MultiHeadAttention](#component-2-multiheadattention)
- [Component 3: TransformerBlock](#component-3-transformerblock)
- [Component 4: VisionTransformer](#component-4-visiontransformer)
- [Why This Works](#why-this-works)
- [ViT vs CNNs](#vit-vs-cnns)
- [Training Considerations](#training-considerations)
- [Interview Questions](#interview-questions)

---

## The Paradigm Shift

**CNNs:** Spatial inductive bias built into architecture
- Convolutions process local neighborhoods
- Translation equivariance by design
- Hierarchical feature learning

**Transformers:** Pure sequence modeling with global receptive field
- Treat image as sequence of patches
- No spatial inductive bias
- Self-attention learns relationships between all patches

**Key insight:** With enough data, transformers can learn spatial relationships that CNNs get "for free" — and often learn better representations.

---

## Component 1: PatchEmbedding

### What It Does

Converts a 2D image into a 1D sequence of patch embeddings.

```python
# Input:  (batch, 3, 224, 224)     RGB image
# Output: (batch, 197, 768)        Sequence of embedded patches
#         197 = 196 patches + 1 CLS token
#         196 = (224/16)² patches for patch_size=16
```

### The Process

**Step 1: Split into patches**

```python
# Image: 224×224
# Patch size: 16×16
# Number of patches: (224/16) × (224/16) = 14 × 14 = 196
```

Each patch is a 16×16×3 = 768-dimensional vector (flattened).

**Step 2: Linear projection (via Conv2d)**

```python
self.projection = nn.Conv2d(
    in_channels=3,
    out_channels=768,  # embed_dim
    kernel_size=16,
    stride=16
)
```

**Why Conv2d?** A convolution with `kernel_size=patch_size` and `stride=patch_size` simultaneously:
1. Splits the image into non-overlapping patches
2. Projects each patch to embedding dimension

This is computationally equivalent to:
```python
# Manual approach (slower):
patches = split_into_patches(image)  # (196, 768)
embeddings = linear_projection(patches)
```

**Step 3: Add positional embeddings**

```python
self.positional_embedding = nn.Parameter(torch.randn(1, 197, 768))
```

**Why needed?** Unlike CNNs, transformers have no notion of spatial position. Positional embeddings tell the model where each patch came from in the original image.

**Learnable vs Fixed:** ViT uses learnable positional embeddings (unlike original Transformer's sinusoidal). Research shows both work, but learnable is slightly better for vision.

**Step 4: Prepend [CLS] token**

```python
self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
```

The [CLS] token is a learnable embedding prepended to the sequence. After passing through all transformer layers, it aggregates information from all patches and is used for classification.

**Why [CLS]?** Borrowed from BERT. Alternative: global average pooling over all patch embeddings. [CLS] works better empirically.

---

## Component 2: MultiHeadAttention

### The Attention Mechanism

**Core equation:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- Q (Query): "What am I looking for?"
- K (Key): "What do I contain?"
- V (Value): "What information do I have?"

### Step-by-Step Breakdown

**Step 1: Generate Q, K, V**

```python
# Single linear projection generates all three
qkv = self.qkv(x)  # (batch, seq_len, 3 * embed_dim)
```

For each token, we create three different representations:
- Q: What this token is querying for
- K: What this token offers (for other tokens to query)
- V: The actual information this token carries

**Step 2: Split into multiple heads**

```python
# embed_dim = 768, num_heads = 12
# head_dim = 768 / 12 = 64

# Reshape: (batch, 197, 768) -> (batch, 12, 197, 64)
```

**Why multiple heads?** Each head can learn to attend to different types of relationships:
- Head 1: Spatial proximity
- Head 2: Color similarity
- Head 3: Texture patterns
- etc.

**Step 3: Scaled dot-product attention**

```python
# Q @ K^T: (batch, 12, 197, 64) @ (batch, 12, 64, 197)
#        -> (batch, 12, 197, 197)
attn = (q @ k.transpose(-2, -1)) * self.scale
```

**What does this matrix represent?**
- Shape: (197, 197) for each head
- attn[i, j] = similarity between patch i and patch j
- Each row: "how much should token i attend to all other tokens?"

**Why scale by √d_k?**

Dot products grow with dimensionality. For large d_k, dot products have large magnitude → softmax saturates → tiny gradients.

Scaling by √d_k keeps dot products in a reasonable range.

**Example:**
```python
# Without scaling:
q = [1, 1, ..., 1]  # 64 dimensions
k = [1, 1, ..., 1]
q @ k = 64  # Large!

# With scaling:
(q @ k) / √64 = 64 / 8 = 8  # Reasonable
```

**Step 4: Softmax (attention weights)**

```python
attn = attn.softmax(dim=-1)  # (batch, 12, 197, 197)
```

Softmax converts similarities to probabilities. Each row sums to 1:
- attn[i, :] = [0.3, 0.05, 0.02, ..., 0.15, 0.01]
- Interpretation: Token i should pay 30% attention to token 0, 5% to token 1, etc.

**Step 5: Apply attention to values**

```python
# (batch, 12, 197, 197) @ (batch, 12, 197, 64)
# -> (batch, 12, 197, 64)
x = attn @ v
```

This is a weighted sum. For token i:
```
output[i] = 0.3 * v[0] + 0.05 * v[1] + ... + 0.01 * v[196]
```

Token i's output is a weighted combination of all other tokens' values, where weights come from attention scores.

**Step 6: Concatenate heads and project**

```python
# (batch, 12, 197, 64) -> (batch, 197, 768)
x = x.transpose(1, 2).reshape(batch, seq_len, embed_dim)
x = self.proj(x)
```

### Attention Visualization

For a 224×224 image with 16×16 patches (196 patches):

```
Attention matrix for one head: (197, 197)

     CLS  P1  P2  P3  ...  P196
CLS [0.01 0.02 0.03 0.01 ... 0.01]
P1  [0.05 0.30 0.20 0.05 ... 0.02]  <- Patch 1 attends to itself (0.30) 
P2  [0.03 0.25 0.35 0.10 ... 0.01]     and its neighbors (0.20, 0.25)
P3  [0.02 0.05 0.15 0.40 ... 0.03]
...
P196[0.01 0.01 0.02 0.03 ... 0.50]
```

Early layers: attention is local (patches attend to neighbors)  
Later layers: attention is global (patches attend across entire image)

---

## Component 3: TransformerBlock

### Architecture

```python
x -> LayerNorm -> MultiHeadAttention -> Add (residual)
  -> LayerNorm -> MLP -> Add (residual) -> output
```

This is the "Pre-LN" (Pre-LayerNorm) architecture, different from the original Transformer.

### Why Pre-LayerNorm?

**Original Transformer (Post-LN):**
```python
x = x + Attention(x)
x = LayerNorm(x)
x = x + MLP(x)
x = LayerNorm(x)
```

**ViT (Pre-LN):**
```python
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

**Benefits of Pre-LN:**
1. **More stable training**: Gradients flow more cleanly through residuals
2. **Can skip final LayerNorm** (though ViT keeps it)
3. **Easier to train deep models** (100+ layers)

Research (Xiong et al., 2020) shows Pre-LN converges better for very deep Transformers.

### The MLP Block

```python
Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768)
```

**Why 4× expansion?** Following original Transformer design. The hidden layer has 4× the embedding dimension.

**Why GELU instead of ReLU?**

GELU (Gaussian Error Linear Unit): `x * Φ(x)` where Φ is the cumulative distribution function of the standard normal distribution.

Approximation: `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`

Properties:
- Smooth (differentiable everywhere)
- Non-monotonic (slightly negative for small negative inputs)
- Empirically better for Transformers

**ReLU:** Hard cutoff at 0  
**GELU:** Smooth transition around 0

### Residual Connections

```python
x = x + self.attn(self.norm1(x))
x = x + self.mlp(self.norm2(x))
```

Same purpose as in ResNet: enable gradient flow to early layers. Without residuals, 12-layer Transformer would suffer from vanishing gradients.

---

## Component 4: VisionTransformer

### Full Forward Pass

```python
Input: (batch, 3, 224, 224)

1. PatchEmbedding:
   (batch, 3, 224, 224) -> (batch, 197, 768)
   
2. Transformer Blocks (×12):
   (batch, 197, 768) -> (batch, 197, 768)  # Same shape
   
3. LayerNorm:
   (batch, 197, 768) -> (batch, 197, 768)
   
4. Extract CLS token:
   (batch, 197, 768) -> (batch, 768)  # Take first token
   
5. Classification head:
   (batch, 768) -> (batch, num_classes)
```

### Why Extract CLS Token?

**Alternative 1: Average all patch tokens**
```python
cls_token = x.mean(dim=1)  # (batch, 197, 768) -> (batch, 768)
```

**Alternative 2: Max pooling**
```python
cls_token = x.max(dim=1)[0]
```

**CLS token approach:**
- Dedicated learnable token
- Attends to all patches through self-attention
- Aggregates global information
- Empirically works best

### Weight Initialization

```python
nn.init.trunc_normal_(module.weight, std=0.02)
```

**Truncated normal:** Normal distribution with values outside 2σ resampled.

**Why std=0.02?** Empirically found to work well for Transformers. Smaller than typical initialization (std=0.1) because:
1. LayerNorm follows, which normalizes anyway
2. Residual connections accumulate activations

---

## Why This Works

### The Inductive Bias Question

**CNNs have strong inductive biases:**
- Locality (convolutions operate on local neighborhoods)
- Translation equivariance (shift input → shift output)
- Hierarchical composition (early: edges, late: objects)

**Transformers have minimal inductive bias:**
- Only constraint: sequence of tokens
- Must learn spatial relationships from data

**So why do Transformers work for vision?**

### Answer: Data Scale

**Small datasets (< 100K images):**
- CNNs outperform ViT
- Inductive bias helps when data is scarce
- Example: CIFAR-10 (50K images) → ResNet > ViT

**Medium datasets (100K - 1M images):**
- CNNs ≈ ViT
- ImageNet (1.3M images) → Similar performance

**Large datasets (> 10M images):**
- ViT outperforms CNNs
- JFT-300M (300M images) → ViT significantly better
- Transformers can learn spatial relationships given enough examples

### What Transformers Learn

Visualizing attention maps shows Transformers learn:

**Early layers:**
- Local patterns (similar to early CNN layers)
- Attend to spatially nearby patches

**Middle layers:**
- Semantic groupings (patches with similar content)
- Object boundaries

**Late layers:**
- Global relationships
- Object parts across entire image
- Scene understanding

**Key difference from CNNs:** Transformers can attend to long-range dependencies from layer 1. CNNs need to stack many layers to achieve this.

---

## ViT vs CNNs

### Computational Complexity

**CNN (ResNet-50):**
- Complexity: O(n²) per layer (for n×n image)
- But: depth is limited (practical max ~200 layers)

**ViT:**
- Self-attention complexity: O(n²) where n = number of patches
- For 224×224 image, 16×16 patches → 196 patches
- Attention: (197, 197) matrix → manageable

**Comparison:**
```
ResNet-50:  25.6M parameters,  4.1 GFLOPs
ViT-Base:   86M parameters,   17.6 GFLOPs  (4× more compute)
```

ViT is more computationally expensive but often more accurate.

### When to Use Each

**Use CNNs when:**
- Small dataset (< 100K images)
- Need computational efficiency
- Strong spatial structure matters
- Limited compute budget

**Use ViT when:**
- Large dataset (> 1M images)
- Transfer learning from pretrained ViT
- Need global relationships early
- Have sufficient compute

**Hybrid approaches:**
- ConvNeXt: CNN designed to match ViT performance
- Swin Transformer: ViT with local windows (more efficient)
- CoAtNet: Combine conv stem + transformer stages

---

## Training Considerations

### ViT Training is Different

**Compared to CNNs, ViT requires:**

1. **More data**
   - Underfits on ImageNet alone
   - Needs pretraining on larger datasets (JFT-300M, ImageNet-21K)

2. **Heavier augmentation**
   - RandAugment, CutMix, MixUp
   - More regularization than CNNs

3. **Longer training**
   - 300 epochs (vs 90 for ResNet)
   - Higher batch sizes (4096 vs 256)

4. **Different optimization**
   - AdamW optimizer (vs SGD for CNNs)
   - Warmup + cosine decay learning rate schedule

### Typical Training Recipe

```python
# Optimizer
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.05)

# Learning rate schedule
warmup_steps = 10000
scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup)

# Augmentation
transforms = [
    RandomResizedCrop(224),
    RandAugment(),
    MixUp(alpha=0.8),
    CutMix(alpha=1.0)
]

# Regularization
- Dropout: 0.1
- Stochastic depth: 0.1
- Label smoothing: 0.1
```

### Why So Much Regularization?

Transformers have **high capacity** — 86M parameters for ViT-Base.

Without strong regularization, they overfit easily on datasets smaller than ImageNet-21K.

CNNs have built-in regularization via inductive bias (locality, weight sharing). ViT must compensate with explicit regularization.

---

## Model Variants

### ViT Sizes

| Model | Layers | Hidden Dim | Heads | Params | ImageNet Acc |
|-------|--------|------------|-------|--------|--------------|
| ViT-Ti | 12 | 192 | 3 | 5.7M | ~72% |
| ViT-S | 12 | 384 | 6 | 22M | ~80% |
| ViT-B | 12 | 768 | 12 | 86M | ~85% |
| ViT-L | 24 | 1024 | 16 | 307M | ~88% |
| ViT-H | 32 | 1280 | 16 | 632M | ~89% |

### Patch Sizes

**ViT-B/16:** Base model, 16×16 patches (most common)  
**ViT-B/32:** Base model, 32×32 patches (faster, fewer patches)  
**ViT-B/8:** Base model, 8×8 patches (more expensive, finer detail)

**Trade-off:**
- Smaller patches → more tokens → better accuracy but slower
- Larger patches → fewer tokens → faster but less detail

---


---

## Key Takeaways

✅ **ViT treats images as sequences** of patch embeddings, enabling standard Transformer architecture

✅ **Self-attention** learns which patches are related without spatial bias

✅ **CLS token** aggregates global information for classification

✅ **Multi-head attention** allows learning different types of relationships simultaneously

✅ **Pre-LayerNorm** provides more stable gradients than Post-LN

✅ **Data scale matters**: ViT needs large datasets to outperform CNNs

✅ **Compute tradeoff**: ViT is more expensive than CNNs but often more accurate

✅ **Heavy regularization required**: Transformers overfit without strong augmentation

---

## Further Reading

**Original Paper:**
- Dosovskiy et al., "An Image is Worth 16x16 Words" (2020)
- https://arxiv.org/abs/2010.11929

**Related Work:**
- DeiT: Data-efficient image transformers (training ViT on ImageNet-1K only)
- Swin Transformer: Hierarchical vision transformer with shifted windows
- ConvNeXt: CNN designed to match ViT performance
- BEiT: BERT pre-training for image transformers

**Implementation Details:**
- "How to train your ViT?" (Steiner et al., 2021)
- ViT training recipes and ablations

---

**Status:** Implementation complete ✅  
**Next:** Train ViT on CIFAR-10 and compare with ResNet performance