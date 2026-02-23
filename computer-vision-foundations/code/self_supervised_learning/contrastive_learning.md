# Contrastive Self-Supervised Learning (SSL) & SimCLR

**Papers:**
- SimCLR v1: Chen et al., 2020 — https://arxiv.org/abs/2002.05709
- SimCLR v2: Chen et al., 2020b — https://arxiv.org/abs/2006.10029

---

## The Core Idea

Learn image representations **without labels** by teaching the model: *two augmented views of the same image should have similar embeddings; views from different images should not.*

No labels. No generative model. No clustering.

```
Image x
   ├── augment t  → x_i → encoder f(·) → h_i → projector g(·) → z_i ─┐
   └── augment t' → x_j → encoder f(·) → h_j → projector g(·) → z_j ─┴→ NT-Xent Loss
```

Downstream tasks use `h` (encoder output), **not** `z`. The projector is discarded after pretraining.

---

## Framework Components

### 1. Data Augmentation
Produces two views of each image. The single most important design decision in SimCLR.

```
RandomCrop + Resize → HorizontalFlip → ColorJitter → RandomGrayscale → GaussianBlur → Normalize
```

**Color distortion is non-negotiable.** Without it the model cheats by matching color histograms rather than learning semantics — the paper ablates this extensively.

---

### 2. Encoder f(·)
Any backbone (ResNet-50 in the original). Outputs representation `h`.  
This is what you keep and fine-tune downstream.

---

### 3. Projection Head g(·)
Small 2-layer MLP mapping `h → z`. Used **only during pretraining**, then discarded.

**Why it helps:** The projector absorbs augmentation-specific information, freeing `h` to encode general semantics. Without it, the encoder must satisfy the contrastive objective directly, degrading representation quality.

This insight — keeping `h` not `z` — recurs throughout the SSL literature.

---

### 4. NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)

For a batch of N images → 2N augmented views total:
- **Positive pair:** the two views of the same image `(x_i, x_j)`
- **Negative pairs:** all other 2(N−1) views in the batch — no explicit mining needed

$$\ell(i,j) = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

Where:
- `sim(u,v) = uᵀv / (‖u‖ ‖v‖)` — cosine similarity
- `τ` — temperature hyperparameter (default 0.5)
- Denominator sums over all 2N−1 other views
- Final loss averages over both directions (i→j and j→i)

**Intuitively:** Softmax cross-entropy where the "correct class" is the positive pair. Temperature controls sharpness — lower τ = harder negatives = harder problem.

**Critical constraint:** More negatives = better signal → SimCLR requires large batches (4096–8192). This is its primary weakness vs. MoCo.

---

## SimCLR v2 Additions

Same loss and framework as v1. Three changes:

1. **Larger encoder** — ResNet-152 (3×wider) gives substantially better representations
2. **Deeper projector** — 3 layers instead of 2; unlike v1, **keep the first layer** when fine-tuning
3. **Knowledge distillation** — pretrained model as teacher for semi-supervised learning; specifically targets the low-label regime

---

## Where SimCLR Fits in SSL

| Method | Year | Negatives | Momentum Encoder | Key Idea |
|--------|------|-----------|-----------------|----------|
| **SimCLR** | 2020 | ✅ large batch | ❌ | Simple, clean baseline |
| **MoCo** | 2020 | ✅ queue | ✅ | Memory-efficient negatives — no giant batches |
| **BYOL** | 2020 | ❌ | ✅ | No negatives at all; stop-gradient prevents collapse |
| **SimSiam** | 2020 | ❌ | ❌ | No negatives, no momentum — just stop-gradient |
| **DINO** | 2021 | ❌ | ✅ | ViT + self-distillation; best transferable features |
| **MAE** | 2021 | ❌ | ❌ | Masked reconstruction; current dominant paradigm |

**Trajectory:** The field moved away from requiring large batches (SimCLR's bottleneck) → then away from negatives entirely (BYOL, SimSiam) → then away from contrastive learning altogether toward masked autoencoders (MAE).

---

## Practical Notes

**Data efficiency:** SSL pretrained on unlabeled data + fine-tune on 1–10% labeled data often matches or exceeds fully supervised training on the full labeled set. This is the core value proposition.

**What to use today:**
- **DINO** — best off-the-shelf ViT features, excellent for downstream tasks
- **MAE** — scalable, state of the art for pretraining
- **SimCLR** — study and implement to understand contrastive learning; not for production

---

## Key Takeaways

1. SSL works by maximizing agreement between views of the same image
2. Augmentation design is as important as architecture
3. The projector head trick (discard at fine-tune time) is a broadly applicable insight
4. Large batch requirement is SimCLR's fundamental weakness — solved by MoCo's queue
5. The field has largely moved to masked autoencoders (MAE), but contrastive methods remain highly relevant in multimodal settings (CLIP is contrastive)