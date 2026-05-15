# Vision Transformer (ViT) Deep Dive

**Reference Implementation:** `vit.py`  
**Paper:** “An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale” (Dosovitskiy et al., 2020)[file:129][web:148][web:151]

---

## Table of Contents

- [The Paradigm Shift](#the-paradigm-shift)
- [Component 1: PatchEmbedding](#component-1-patchembedding)
- [Component 2: MultiHeadAttention](#component-2-multiheadattention)
- [Component 3: TransformerBlock](#component-3-transformerblock)
- [Component 4: VisionTransformer](#component-4-visiontransformer)
- [Why This Works](#why-this-works)
- [ViT vs CNNs](#vit-vs-cnns)
- [ViT vs "Attention Is All You Need"](#vit-vs-attention-is-all-you-need-2017)
- [Training Considerations](#training-considerations)
- [Model Variants](#model-variants)

---

## The Paradigm Shift

**CNNs: strong spatial inductive bias.**  

- Convolutions operate on local neighbourhoods and share weights spatially.  
- Translation equivariance is built in: shifting the input shifts the feature maps.  
- Stacking layers yields hierarchical features (edges → parts → objects).[file:129][web:137]

**Transformers: minimal spatial inductive bias.**  

- Treat an image as a sequence of patch tokens and use self‑attention over tokens.  
- No inherent locality or translation equivariance; all relations are learned.  
- Global receptive field from the first layer via attention over all patches.[file:129][web:148]

**Key insight (ViT):** With sufficient data (e.g., JFT‑300M), a pure transformer applied to image patches can match or surpass CNNs on classification tasks, despite lacking convolutional inductive biases.[web:148][web:151]

---

## Component 1: PatchEmbedding

### What It Does

Converts a 2D image into a 1D sequence of patch embeddings.[file:129][web:148]

```python
# Input:  (batch, 3, 224, 224)     RGB image
# Output: (batch, 197, 768)        patch tokens + CLS
#         197 = 196 patches + 1 CLS token
#         196 = (224/16)² patches for patch_size = 16
```

### The Process

**Step 1: Split into patches**

```python
# Image: 224 × 224
# Patch size: 16 × 16
# Number of patches: (224 / 16) × (224 / 16) = 14 × 14 = 196
```

Each patch covers 16×16×3 = 768 scalar values; in ViT these are linearly projected to an embedding of dimension \(D\) (often 768 for ViT‑B), not kept as raw 768‑D.[file:129][web:148]

**Step 2: Linear projection via Conv2d**

```python
self.projection = nn.Conv2d(
    in_channels=3,
    out_channels=embed_dim,  # e.g. 768
    kernel_size=patch_size,  # e.g. 16
    stride=patch_size
)
```

A conv layer with `kernel_size = stride = patch_size` performs, in one operation:[file:129][web:148][web:156]

1. Non‑overlapping patch extraction.  
2. Linear projection of each patch to `embed_dim`.

This is equivalent to manually flattening patches and applying a fully connected layer to each patch, but more efficient and idiomatic in PyTorch.[file:129][web:156]

**Step 3: Add positional embeddings**

```python
self.positional_embedding = nn.Parameter(
    torch.randn(1, n_patches + 1, embed_dim)
)
```

Transformers are permutation‑invariant over tokens, so they need explicit position information.[file:129][web:148] ViT uses *learnable* 1D positional embeddings over the patch sequence (plus the CLS position), as opposed to fixed sinusoidal encodings used in Vaswani et al.[web:148][web:151] The ViT paper reports small or negligible accuracy differences between fixed and learned positions, but learnable embeddings are now standard in vision transformers.[web:148]

**Step 4: Prepend [CLS] token**

```python
self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
```

A learnable [CLS] token is prepended to the patch sequence. After passing through all transformer layers, its final representation is used as a compact summary for classification.[file:129][web:148]

Alternatives like global average pooling over patch tokens are viable, but empirically [CLS] performs slightly better for ViT‑style models and aligns with BERT‑style pretraining recipes.[file:129][web:148][web:151]

---

## Component 2: MultiHeadAttention

### The Attention Mechanism

Standard scaled dot‑product attention:[file:129][web:148]

```text
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
```

- Q (queries): what each token is looking for.  
- K (keys): what each token offers.  
- V (values): the information to aggregate.

### Step‑by‑Step Breakdown

**Step 1: Generate Q, K, V**

```python
qkv = self.qkv(x)  # (batch, seq_len, 3 * embed_dim)
```

A single linear layer produces concatenated Q, K, V for each token, which are then reshaped and split.[file:129][web:148]

**Step 2: Split into multiple heads**

```python
# Example: embed_dim = 768, num_heads = 12 → head_dim = 64
# (batch, seq_len, 768) -> (batch, 12, seq_len, 64)
```

Multiple heads allow the model to attend to different relational patterns in parallel (e.g., spatial neighbours, similar colour/texture, long‑range dependencies).[file:129][web:148]

**Step 3: Scaled dot‑product attention**

```python
attn = (q @ k.transpose(-2, -1)) * self.scale  # self.scale = 1/√d_k
# Shape: (batch, num_heads, seq_len, seq_len)
```

The attention matrix per head has shape (tokens × tokens), including the CLS token (so 197×197 for ViT‑B/16).[file:129][web:148]

Scaling by \(1/\sqrt{d_k}\) prevents inner products from growing too large with dimension, which would make softmax saturate and harm gradients.[file:129][web:148]

**Step 4: Softmax (attention weights)**

```python
attn = attn.softmax(dim=-1)
```

Each row is a probability distribution over all tokens: for token i, attn[i, :] encodes how much it attends to every other token.[file:129][web:148]

**Step 5: Apply attention to values**

```python
x = attn @ v  # (batch, num_heads, seq_len, head_dim)
```

For each token i:

```text
output[i] = Σ_j attn[i, j] * v[j]
```

So the new representation of token i is a learned weighted average of other tokens’ value vectors.[file:129][web:148]

**Step 6: Concatenate heads and project**

```python
x = x.transpose(1, 2).reshape(batch, seq_len, embed_dim)
x = self.proj(x)
```

This re‑merges head outputs and applies a final linear projection.[file:129][web:148]

### Attention Patterns (Qualitative)

Empirically, attention maps in ViT exhibit:[file:129][web:148]

- Early layers: mainly local attention to neighbouring patches (CNN‑like behaviour).  
- Middle layers: attention that clusters semantically similar regions (object parts).  
- Late layers: more global attention across distant regions and to the CLS token.

---

## Component 3: TransformerBlock

### Architecture

```python
x -> LayerNorm -> MultiHeadAttention -> Add (residual)
  -> LayerNorm -> MLP -> Add (residual) -> output
```

ViT uses a **Pre‑LayerNorm** (Pre‑LN) architecture: the LN is applied before each sublayer (attention/MLP), not after the residual, which differs from the original Transformer.[file:129][web:148][web:153]

### Why Pre‑LayerNorm?

Original Vaswani et al. used **Post‑LN**:

```python
# Post-LN (original Transformer)
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + MLP(x))
```

ViT uses **Pre‑LN**:

```python
# Pre-LN (ViT-style)
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

Pre‑LN improves gradient flow, especially in deeper models, by providing a clean residual path that does not pass through LayerNorm, making training more stable and less sensitive to warmup and learning‑rate schedules.[file:129][web:153][web:158]

### The MLP Block

```python
MLP: Linear(embed_dim → 4 * embed_dim) → GELU → Linear(4 * embed_dim → embed_dim)
```

- Hidden dimension is typically 4× the embedding dimension, mirroring Vaswani et al.[file:129][web:148]  
- GELU is used instead of ReLU; it is smoother and has empirically worked better for Transformer‑style architectures.[file:129][web:148]

GELU is approximated as:

\[
\operatorname{GELU}(x) \approx 0.5 x \big(1 + \tanh\big(\sqrt{2/\pi} (x + 0.044715 x^3)\big)\big)
\]

### Residual Connections

```python
x = x + self.attn(self.norm1(x))
x = x + self.mlp(self.norm2(x))
```

Residual connections play the same role as in ResNets: they help gradients flow to early layers and mitigate vanishing gradients; without them, a 12‑layer transformer would be significantly harder to train.[file:129][web:148]

---

## Component 4: VisionTransformer

### Full Forward Pass

```python
Input: (batch, 3, 224, 224)

1. PatchEmbedding:
   (batch, 3, 224, 224) -> (batch, 197, embed_dim)

2. Transformer blocks (×L, e.g. 12):
   (batch, 197, embed_dim) -> (batch, 197, embed_dim)

3. LayerNorm:
   (batch, 197, embed_dim) -> (batch, 197, embed_dim)

4. Extract CLS token:
   (batch, 197, embed_dim) -> (batch, embed_dim)

5. Classification head:
   (batch, embed_dim) -> (batch, num_classes)
```

### Why Extract the CLS Token?

Alternatives include averaging or max‑pooling patch tokens:

```python
# Mean pooling
cls_like = x.mean(dim=1)        # (batch, embed_dim)

# Max pooling
cls_like, _ = x.max(dim=1)      # (batch, embed_dim)
```

ViT, however, designates the first token as a learnable [CLS] token, which attends to all patches and is then used as the input to the classifier.[file:129][web:148] This mechanism is borrowed from BERT and works well in practice; many ViT variants retain it even when adding alternative pooling strategies.

### Weight Initialization

ViT uses truncated normal initialisation for linear layers, with standard deviation around 0.02:[file:129][web:148]

```python
nn.init.trunc_normal_(module.weight, std=0.02)
```

- A relatively small std helps stabilise early training in deep residual networks.  
- LayerNorm and residual connections compensate, but small initial weights avoid large activations and gradients at start‑up.[file:129][web:148]

---

## Why This Works

### Inductive Bias vs Data Scale

**CNNs** provide strong inductive biases:[file:129][web:148]

- Locality and translation equivariance.  
- Hierarchical feature extraction built directly into the architecture.  
- Good performance on relatively small datasets (e.g., CIFAR‑10, ImageNet‑1k) with moderate training schedules.

**ViT** has weaker inductive biases:[file:129][web:148]

- It only assumes a token sequence; locality and translation behaviour are learned.  
- As a result, ViT is data‑hungry and underperforms CNNs when trained from scratch on modest datasets like ImageNet‑1k without special tricks.[web:148][web:154]

Dosovitskiy et al. show that, when pre‑trained on very large datasets (e.g., JFT‑300M) and then fine‑tuned, ViT matches or surpasses state‑of‑the‑art CNNs across several benchmarks, while being competitive in training efficiency at scale.[web:148][web:151]

### What Transformers Learn in Vision

Attention visualisation reveals that:[file:129][web:148]

- Early layers: focus mainly on local neighbourhoods (akin to conv receptive fields).  
- Middle layers: organise semantically similar patches and follow object boundaries.  
- Later layers: capture global relations and scene‑level semantics, with CLS attention spreading over relevant object regions.

Transformers can attend to long‑range dependencies from the first layer; CNNs must stack many layers or add explicit non‑local blocks to achieve comparable receptive fields.[file:129][web:148]

---

## ViT vs CNNs

### Computational Complexity

For a fixed input size, both CNNs and transformers have roughly quadratic cost in spatial resolution, but with different constants and scaling behaviour.

**ResNet‑50 (typical CNN):**

- ~25.6M parameters, ≈4 GFLOPs on 224×224 inputs.[web:63]  

**ViT‑Base/16 (as reported in implementations and follow‑up work):**

- 12 layers, 768‑D embeddings, 12 heads, MLP ratio 4.  
- ~86M parameters.[file:129][web:148][web:156]  
- FLOPs depend on implementation; some references report ≈16–18 GFLOPs at 224×224 resolution for ViT‑B/16.[web:156][web:160]

So ViT‑B/16 is substantially larger and more computationally intensive than ResNet‑50 at 224×224, though efficient kernels and hardware can narrow the gap.[file:129][web:156]

### When to Use Each

**CNNs are often preferable when:**[file:129][web:148][web:154]

- Data is limited (≲10^5 images).  
- Compute/budget constraints are tight.  
- Deployment targets have optimised convolution kernels.

**ViT is attractive when:**

- You have access to large‑scale pretraining (e.g., ImageNet‑21k, JFT) or strong distillation recipes (e.g., DeiT).[web:148][web:154]  
- You want strong transfer performance across diverse downstream tasks.  
- You can afford higher compute and parameter counts.

**Hybrid approaches:**  

- **ConvNeXt:** CNNs redesigned to match ViT performance using modern training recipes.  
- **Swin Transformer:** Hierarchical ViT with local  window attention (better scaling in resolution).  
- **CoAtNet:** Hybrid conv + attention backbone.[file:129][web:155]

---

## ViT vs "Attention Is All You Need" (2017)

The original Transformer was designed for sequence‑to‑sequence tasks (e.g., machine translation). ViT borrows the encoder stack and drops the decoder and cross‑attention.[file:129][web:148]

### Original Transformer Overview

```text
Encoder:                         Decoder:
Input tokens                     Shifted output tokens
  ↓                                  ↓
Token + Positional Encoding      Token + Positional Encoding
  ↓                                  ↓
[Multi-Head Self-Attention]      [Masked Multi-Head Self-Attention]
[Add & Norm]                     [Add & Norm]
[Feed-Forward]                       ↓
[Add & Norm]                     [Multi-Head Cross-Attention] ← attends to encoder output
  ↓ (×N layers)                  [Add & Norm]
Encoder output                   [Feed-Forward]
                                 [Add & Norm]
                                     ↓ (×N layers)
                                 Linear + Softmax
                                 Output sequence
```

ViT uses only the **encoder** side: stacked self‑attention + MLP blocks with positional encodings.[file:129][web:148]

### Attention Types and What ViT Uses

- **Self‑attention (encoder):** tokens attend to one another within the same sequence — used by ViT.  
- **Masked self‑attention (decoder):** prevents peeking at future tokens — irrelevant for ViT.  
- **Cross‑attention (decoder→encoder):** lets output tokens attend to encoded inputs — not needed for classification.[file:129][web:148]

### LayerNorm: Post‑LN vs Pre‑LN

Original Transformer:[file:129][web:148]

```python
# Post-LN
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + MLP(x))
```

ViT and most modern transformers:[file:129][web:153][web:158]

```python
# Pre-LN
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

Pre‑LN improves stability, especially in deeper networks, by ensuring a clear residual path for gradients and reducing the need for aggressive learning‑rate warmup.[web:153][web:158]

### Positional Encoding: Fixed vs Learnable

Original Transformer uses fixed sinusoidal positional encodings:[file:129][web:148]

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

ViT uses **learnable** positional embeddings:[file:129][web:148]

```python
self.positional_embedding = nn.Parameter(
    torch.randn(1, n_patches + 1, embed_dim)
)
```

Sinusoidal encodings have nice extrapolation properties, but ViT shows that learnable embeddings work at least as well within the training resolution, with interpolation used to adapt to different resolutions at fine‑tuning time.[web:148][web:151]

### Summary Table

|                      | Transformer (2017)             | ViT (2020)                         |
|----------------------|---------------------------------|------------------------------------|
| Architecture         | Encoder + decoder               | Encoder only                       |
| Attention types      | Self, masked self, cross        | Self only                          |
| LayerNorm position   | Post‑LN                         | Pre‑LN                             |
| Positional encoding  | Fixed sinusoidal                | Learnable                          |
| Input tokens         | Word/subword embeddings         | Patch embeddings (+ CLS)           |
| Output               | Token sequence                  | Class logits                       |
| Main task            | Seq‑to‑seq (e.g., translation)  | Image classification               |

[file:129][web:148]

---

## Training Considerations

### ViT Training vs CNN Training

Compared to typical CNN schedules, ViT training recipes differ in several ways:[file:129][web:148][web:155]

1. **More data / pretraining.**  
   - Original ViT models are pre‑trained on JFT‑300M or ImageNet‑21k before fine‑tuning on ImageNet‑1k.[web:148][web:151]  
   - Training from scratch on ImageNet‑1k usually underperforms unless using DeiT‑style data‑efficient recipes and distillation.[web:154][web:159]

2. **Stronger augmentation.**  
   - RandAugment, MixUp, CutMix and related techniques are commonly used.  
   - These act as explicit regularisation to compensate for weaker inductive biases.

3. **Longer training and larger batches.**  
   - 300+ epochs and batch sizes in the thousands are typical in the original ViT work.  
   - Cosine decay with warmup is standard.[web:148]

4. **Different optimiser and regularisation.**  
   - AdamW (Adam with decoupled weight decay) instead of SGD with momentum.  
   - Non‑trivial weight decay, label smoothing, dropout, and stochastic depth are used.[file:129][web:148][web:155]

These choices are documented further in follow‑up work such as “How to Train Your ViT” and DeiT.[file:129][web:154][web:155]

---

## Model Variants

### ViT Sizes (Canonical Family)

Common ViT configurations (numbers approximate and may vary slightly across implementations):[file:129][web:148][web:160]

| Model    | Layers | Hidden dim | Heads | Params (M) | Typical ImageNet‑1k top‑1* |
|----------|--------|------------|-------|-----------:|----------------------------:|
| ViT‑Ti   | 12     | 192        | 3     | ≈5–6       | ≈72% (with strong recipe)   |
| ViT‑S    | 12     | 384        | 6     | ≈22        | ≈80%                        |
| ViT‑B    | 12     | 768        | 12    | ≈86        | ≈84–86% (with large‑scale pretraining) |
| ViT‑L    | 24     | 1024       | 16    | ≈300       | ≈87–89%                     |
| ViT‑H    | 32     | 1280       | 16    | ≈630       | ≈88–89%+                    |

\*Exact accuracies depend heavily on dataset (ImageNet‑1k vs ImageNet‑21k vs JFT), training recipe, and distillation; the table shows representative ranges from ViT and follow‑up scaling papers.[web:148][web:155][web:160]

### Patch Sizes

ViT models are often denoted as ViT‑X/P, where X is size (B, L, etc.) and P is patch size:[file:129][web:148]

- **ViT‑B/32:** Base model, 32×32 patches (fewer tokens → faster, coarser).  
- **ViT‑B/16:** Base model, 16×16 patches (default).  
- **ViT‑B/8:** Base model, 8×8 patches (more tokens → slower, finer detail).

**Trade‑off:**  

- Smaller patches → more tokens → higher compute and memory → finer spatial granularity, higher accuracy.  
- Larger patches → fewer tokens → lower cost → potentially less detailed representations.[web:148][web:155]

---

## Key Takeaways

- ViT treats images as sequences of **patch embeddings**, enabling direct reuse of Transformer encoder architectures.[file:129][web:148]  
- **Self‑attention** learns which patches are related, without hard‑coded spatial bias.  
- A **CLS token** aggregates global information and feeds a classifier head.  
- **Multi‑head attention** captures diverse relationships simultaneously.  
- **Pre‑LayerNorm** stabilises deep transformer training and is now standard.  
- ViT needs **large‑scale data or strong training recipes** to outperform CNNs.  
- ViT models are more computationally expensive than ResNet‑class CNNs at the same resolution, but offer excellent transfer and state‑of‑the‑art accuracy when scaled.[web:148][web:155][web:160]

---

## Further Reading

**Core papers:**

- Dosovitskiy et al., “An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.”[web:148][web:151]  
- Touvron et al., “DeiT: Data‑Efficient Image Transformers.” (ViT‑style models trained on ImageNet‑1k from scratch with distillation).[web:154][web:159]  
- Zhai et al., “Scaling Vision Transformers.” (Large‑scale ViT training, up to billions of parameters).[web:155][web:160]

**Training recipes and implementations:**

- “How to Train Your ViT?” (Steiner et al.) — ablations and best practices.  
- Official and open‑source ViT codebases (e.g., timm, Google’s JAX/Flax implementations).[web:156]
