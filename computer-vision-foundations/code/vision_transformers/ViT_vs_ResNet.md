# ViT vs ResNet — CIFAR-10 Comparison

*Both models trained on CIFAR-10 (50K train / 10K test). My actual results.*

---

## Results at a Glance

| | ViT-Tiny | ResNet-18 |
|---|---|---|
| **Best test accuracy** | 86.99% | 93.43% |
| **Parameters** | 5.3M | 11.7M |
| **Training time (T4 GPU)** | 44 min / 100 epochs | ~15 min / 100 epochs |
| **Epochs to converge** | ~96 | ~60-70 |
| **Optimizer** | AdamW | SGD + momentum |
| **Batch size** | 128 | 128 |
| **LR schedule** | Linear warmup + cosine decay | Cosine decay |
| **Augmentation** | MixUp + CutMix + TrivialAugment + Random Erasing | RandomCrop + HorizontalFlip |

---

## Training Curves

### What ViT Curves Show

Looking at the actual training plot from Kaggle:

**Loss curves**: Train loss oscillates heavily the entire run (~1.4-1.8 range), while test loss smoothly decreases from ~2.0 to ~0.8. The gap between them is large. This is *expected behaviour*, not a problem — MixUp and CutMix make the training loss artificially inflated (the model is scored against a mix of two labels simultaneously).

**Accuracy curves**: Train accuracy oscillates wildly (20-80% swings between epochs). Test accuracy climbs smoothly from ~50% at epoch 10 to 86.99% at epoch 96. Same cause: the oscillating train metric is meaningless when mixing labels. **Only test accuracy matters.**

**LR schedule**: Clean warmup to 0.001 at epoch 10, then smooth cosine decay to ~0 at epoch 100. The accuracy jump at epoch 10 is visible — that's warmup ending and full learning rate kicking in.

**ResNet training curves look fundamentally different:**
- Train and test accuracy track each other closely (no wild oscillation)
- Train accuracy climbs above test accuracy smoothly — classic overfitting gap
- Converges earlier, around epoch 60-70
- Loss curves are smooth from the start

The visual difference reflects the architectural difference: ResNet trains stably with minimal augmentation, ViT requires heavy augmentation which makes the training metrics unreliable.

---

## Accuracy

**ResNet-18: 93.43% | ViT-Tiny: 86.99%**

ResNet wins by 6.44 percentage points. This matches the theoretical prediction exactly: on 50K images, CNNs' built-in spatial inductive biases (locality, translation equivariance) give them an inherent advantage over Transformers that must learn these relationships from scratch.

### Per-class breakdown from ViT results:

| Class | ViT-Tiny | ResNet-18 (typical) | Gap |
|-------|----------|---------------------|-----|
| airplane | 89.7% | 95% | -5.3% |
| automobile | 94.9% | 97% | -2.1% |
| bird | 83.3% | 93% | -9.7% |
| **cat** | **68.4%** | **87%** | **-18.6%** |
| deer | 80.9% | 93% | -12.1% |
| dog | 81.8% | 88% | -6.2% |
| frog | 93.1% | 95% | -1.9% |
| horse | 92.3% | 96% | -3.7% |
| ship | 92.5% | 95% | -2.5% |
| truck | 90.1% | 96% | -5.9% |

Cat is the biggest gap by far. **Why?** Cats have highly variable poses, appearances, and orientations on 32×32 images. CNNs handle this via translation equivariance — a cat-ear pattern is detected regardless of where it appears. ViT has to learn "cat-ear-at-position-12 is the same as cat-ear-at-position-7" entirely from data. At 32×32 resolution with only 50K examples, it never fully learns this.

The classes where ViT gets close to ResNet (automobile 94.9%, frog 93.1%, horse 92.3%) are classes with distinctive global shapes that are consistent across examples — exactly the kind of global pattern where attention-based models are at their best.

---

## Parameter Count

**ResNet-18: 11.7M | ViT-Tiny: 5.3M**

ViT-Tiny has less than half the parameters yet gets within 6.5% accuracy. On a per-parameter basis, ViT-Tiny is actually more efficient. The performance gap comes from architecture design, not model capacity.

### Where the parameters live:

**ResNet-18 (11.7M):**
- Conv layers: ~9.5M (mostly in the later residual blocks where channel count is 256/512)
- BatchNorm: ~0.1M
- Fully connected head: ~2.1M (512 → 10 with 512-dim global avg pool)

**ViT-Tiny (5.3M):**
- Patch embedding conv: 192 × 3 × 4 × 4 = **9,216 params** — shockingly small
- QKV projections (×12 layers): 12 × (192 × 576) = **1.33M**
- Output projections (×12 layers): 12 × (192 × 192) = **0.44M**
- MLP layers (×12 layers): 12 × (192×768 + 768×192) = **2.36M**
- LayerNorm (×24): ~9,216
- Positional embeddings: 65 × 192 = **12,480 params**
- CLS token: **192 params**
- Classification head: 192 × 10 = **1,920 params**

The CLS token and positional embeddings that seem architecturally important are actually tiny in parameter count. The bulk of ViT is in the QKV and MLP projections, which scale with `embed_dim²`.

---

## Training Time

**ResNet-18: ~15 min | ViT-Tiny: 44 min (both T4 GPU, 100 epochs)**

ViT is ~3× slower despite fewer parameters. The bottleneck is the attention operation.

### Why attention is slower than convolution:

**Convolution complexity**: O(k² × C_in × C_out × H × W) where k is kernel size. For ResNet, k=3, so most compute is O(9 × C²) per spatial position.

**Self-attention complexity**: O(n² × d) where n is sequence length and d is embedding dim. For 32×32 image with patch_size=4: n=65 tokens. The 65×65 attention matrix must be computed for every head, every layer, every batch.

At 32×32 with 4×4 patches, n=65 is actually small. At 224×224 with 16×16 patches, n=197 — the quadratic cost becomes the dominant factor and ViT becomes much slower relative to ResNets.

**Also**: PyTorch's `nn.Conv2d` is heavily optimised with cuDNN kernels specifically tuned for GPUs. The attention operations, while GPU-accelerated, don't benefit from the same decades of low-level optimisation that convolutions have.

---

## The Core Trade-off

Both models are 5-12M parameters. On 50K images, ResNet wins clearly. But this reverses at scale:

| Dataset size | Winner | Why |
|---|---|---|
| CIFAR-10 (50K) | ResNet | CNN inductive bias compensates for data scarcity |
| ImageNet (1.3M) | Roughly equal | Enough data for ViT to learn spatial patterns |
| ImageNet-21K (14M) | ViT | ViT representations generalise better |
| JFT-300M (300M) | ViT by large margin | Full potential of attention at scale |

**For transfer learning** the gap flips even on small data: a ViT pretrained on ImageNet-21K and finetuned on CIFAR-10 for 10 epochs reaches 98%+, well above any CNN trained from scratch on CIFAR-10. The representations ViT learns from large data are more transferable.

---

## What This Means Architecturally

The accuracy gap is a direct measurement of inductive bias value on small datasets.

ResNet's 3×3 convolutions enforce that nearby pixels are related. Every filter is shared across all spatial positions. The model never has to "discover" that patch position 3 and patch position 4 are adjacent — the kernel sees both simultaneously.

ViT starts with no such assumption. The positional embeddings give it the *opportunity* to learn spatial relationships, but learning which of the 65×65 = 4,225 pairwise relationships matter takes a lot of data. On 50K examples distributed across 10 classes, that's only 5K examples per class — not enough for the attention mechanism to fully specialise.

This is why the ViT paper's main claim is specifically about scale: the whole point is that at >10M images, the flexibility of attention produces better representations than the rigidity of convolution. Testing it in exactly the regime where it was predicted to underperform, and it underperforms by the predicted amount.