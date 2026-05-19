# Four papers that shaped visual recognition

These reading notes trace a **23‑year arc** from early convolutional networks to self‑supervised vision transformers. LeNet (1998) showed end‑to‑end learning from pixels could work at industrial scale on document recognition. VGG (2014/2015) established that depth with small filters can outperform shallow architectures with large filters. DeiT (2021) made vision transformers practical on ImageNet‑1k without proprietary 300M‑image pretraining. DINO (2021) showed that self‑supervised ViTs can develop strong spatial and semantic understanding, including emergent segmentation, without labels. Together, these papers encode key design lessons for modern vision architectures.

---

## LeNet‑5: the blueprint every CNN still follows

**Paper:** “Gradient‑Based Learning Applied to Document Recognition,” LeCun, Bottou, Bengio, Haffner, Proc. IEEE 86(11), 1998.

Note: the “5” in LeNet‑5 is a *version number*, not the number of layers.

### Core problem

Before LeNet, pattern recognition pipelines typically used a hand‑crafted feature extractor (edges, strokes, templates) followed by a trainable classifier (e.g., MLP, SVM). Feature engineering constrained performance and generalisation. LeCun et al. asked whether a single, differentiable system could learn both features and classifier jointly from pixels via gradient descent.

They answered “yes” by deploying convolutional networks at NCR to read large volumes of handwritten ZIP codes and bank cheques in production, demonstrating industrial‑scale viability.

### Architecture (LeNet‑5)

LeNet‑5 takes **32×32 grayscale** images (MNIST digits are 28×28, zero‑padded to allow features near borders). The canonical stack:

- C1: 6 feature maps, **5×5** conv kernels.  
- S2: 6 feature maps, 2×2 *subsampling* (learned average pooling with trainable scale and bias per map).  
- C3: 16 feature maps, 5×5 kernels with **sparse connections** to S2 feature maps (not all‑to‑all).  
- S4: 16 feature maps, 2×2 subsampling.  
- C5: 120 “convolutional” maps with 5×5 kernels; since S4 is 5×5, C5 is effectively fully connected to S4.  
- F6: 84 fully‑connected units.  
- Output: 10 units using a radial basis function (RBF) style formulation in the original paper.

Total parameters are on the order of **60,000**, with about 340K connections, carefully matched to the training data size (~60k images) as a form of capacity control.

**Activation:** a **scaled tanh** `f(a) = 1.7159 * tanh(2a/3)` chosen so that neuron outputs roughly lie in [−1,1] and derivatives are well‑scaled for gradient descent.

**Output layer:** the original system uses Euclidean RBF‑like units rather than softmax; F6 had 84 units because they matched 7×12 bitmap prototypes for each digit class.

### Misunderstood design details

1. **S2/S4 are not simple average pooling.** Each subsampling layer applies local averaging plus a trainable scale and bias per feature map, then the nonlinearity; this is closer to a learned pooling than to today’s fixed average pooling.  
2. **C3 uses a sparse connection table.** Its 16 output maps each connect to specific subsets of S2 maps, deliberately breaking symmetry and encouraging diverse feature sets.  
3. **C5 is effectively fully connected.** Because S4 is 5×5 and C5’s kernels are 5×5, each unit in C5 sees the entire S4 map; the “conv” nomenclature reflects implementation, not receptive field size.

### Key results

On MNIST (60k train, 10k test), LeNet‑5 achieved:

- ≈0.95% error without elastic distortions.  
- ≈0.8% error with distortions (data augmentation).  
- A boosted ensemble of LeNet‑4 models reached ≈0.7% error.

The paper compares several methods on the same benchmark, including SVMs and MLPs; convolutional nets achieved the best tradeoff between error and rejection rates.

### What was superseded

Most specific design choices have since been replaced:

- Scaled tanh → **ReLU/GELU** activations.  
- Learned average subsampling → **max or average pooling** (without per‑map parameters).  
- RBF output and MSE loss → **softmax + cross‑entropy**.  
- Sparse connectivity → mostly **dense convolutional connectivity**.  
- Second‑order/diagonal‑Hessian SGD → **modern optimisers** (SGD with momentum, Adam/AdamW).

MNIST itself is now nearly saturated; state‑of‑the‑art models can reach ≤0.2% error with modern architectures and training.

### Why it matters

LeNet established the classic **conv → subsample → conv → subsample → fully connected** blueprint that influenced AlexNet, VGG, and many early CNNs. The full paper also introduced ideas like Graph Transformer Networks and Space Displacement Neural Networks, which anticipated later work on sequence models and fully convolutional architectures.

---

## VGGNet: depth wins, and 3×3 is all you need

**Paper:** “Very Deep Convolutional Networks for Large‑Scale Image Recognition,” Simonyan & Zisserman, arXiv:1409.1556, ICLR 2015.

### Core problem

After AlexNet’s ImageNet success, the open question was which architectural ingredients mattered most: filter sizes, width, depth, or local response normalisation, etc. VGG’s hypothesis was minimalist: **fix almost everything and increase depth using only 3×3 convolutions**, to isolate the effect of depth.

### Architecture (VGG‑16/19)

VGG‑16 (configuration D) uses:

- Stacks of 3×3 conv layers with channels: 64 → 128 → 256 → 512 → 512.  
- Conv layers per block: 2‑2‑3‑3‑3 (13 conv layers total).  
- Each block ends with 2×2 max pooling, stride 2.  
- Three fully connected layers: 4096‑4096‑1000, followed by softmax.  
- ReLU activations everywhere; no BatchNorm in the original VGG.  
- About **138M parameters**.

VGG‑19 (config E) adds one extra conv in the last three blocks (2‑2‑4‑4‑4), increasing depth to 16 conv layers + 3 FC (19 weight layers) with ≈144M parameters.

They explicitly tested local response normalisation (LRN) and found no benefit, so they discarded it.

### 3×3 stacks vs large kernels

Three stacked 3×3 layers have an effective receptive field equivalent to a 7×7 conv, but with:

- Three nonlinearities (ReLUs) instead of one (increased expressivity).  
- Fewer parameters: 3×(3×3 C²) = 27C² vs 7×7 C² = 49C² (≈45% fewer) for same channel count.

VGG empirically validates that deeper stacks of small kernels outperform shallower networks with larger kernels: shallower comparisons with 5×5 or 7×7 kernels yield significantly higher top‑1 error on ImageNet.

### Training specifics

Key training hyperparameters for ImageNet:

- SGD with momentum 0.9, batch size 256, weight decay 5×10⁻⁴.  
- Dropout 0.5 on the first two FC layers.  
- Initial learning rate 10⁻², reduced by factor 10 when validation error plateaus.  
- Training for 74 epochs on 4 GPUs (Titan-class at the time).  
- Multi‑scale training and testing (shorter side rescaled in [256, 512]) boosts performance.

### Key results

Representative ImageNet results (single models, multi‑scale evaluation):

- VGG‑16: ~24.4% top‑1, 7.5–7.2% top‑5 error on validation with dense + multi‑crop evaluation.  
- A 2‑model ensemble (VGG‑16 + VGG‑19) achieved ≈23.7% top‑1 and 6.8% top‑5, close to GoogLeNet’s multi‑model ensemble.  
- VGG features transferred very well: linear SVMs on frozen conv features set new state‑of‑the‑art results on VOC and other benchmarks.

Exact percentages vary by evaluation protocol, but the key finding is that 16–19‑layer VGG nets significantly improved upon prior CNNs in both classification and transfer.

### What was superseded

VGG’s simplicity comes at substantial cost:

- **Parameter inefficiency:** ≈138M parameters, the majority in FC layers (e.g., 7×7×512×4096 ≈102M params for FC1). Later architectures replaced these with global average pooling.  
- **Compute:** VGG‑16 requires approximately 15–16 GFLOPs for 224×224 inputs, whereas ResNet‑50 achieves comparable or better accuracy with ≈3–4 GFLOPs and ~25M parameters.  
- **Depth limits:** VGG‑19 provides little improvement over VGG‑16, and further depth increases were difficult to train without residual connections.

### Insights

VGG solidified several norms:

- 3×3 convolutions as the **default kernel size** for nearly a decade.  
- A simple, repeated block structure and channel‑doubling after pooling (64→128→256→512…) as a standard pattern.  
- The utility of using a strong classification backbone as a **generic feature extractor** for other tasks (detection, style transfer, perceptual losses).  

Its work on dense evaluation (converting FC layers to 7×7 and 1×1 convs to make the net fully convolutional) also anticipates techniques used in FCNs and dense prediction models.

---

## DeiT: training ViTs without 300M images

**Paper:** “Training data‑efficient image transformers & distillation through attention,” Touvron et al., ICML 2021, arXiv:2012.12877.

### Core problem

Dosovitskiy et al.’s ViT achieved state‑of‑the‑art results, but only when pre‑trained on extremely large proprietary datasets (e.g., JFT‑300M). When trained from scratch on ImageNet‑1k, ViT‑B/16 underperformed strong CNN baselines, with top‑1 accuracy around the high‑70s range. DeiT asked whether careful training schedules and distillation could make a convolution‑free ViT competitive on ImageNet‑1k alone, using modest compute (one 8‑GPU node).

### Architecture

DeiT‑B adopts the **same architecture as ViT‑B/16**:

- 12 transformer encoder layers.  
- Embedding dimension 768, 12 attention heads, head dimension 64.  
- MLP hidden size 3072 (4× expansion).  
- Patch size 16×16 on 224×224 images → 196 patch tokens + 1 CLS token.  
- ≈86M parameters.

Variants:

- DeiT‑Ti (tiny): 192‑D embeddings, 3 heads, ≈5M parameters.  
- DeiT‑S (small): 384‑D embeddings, 6 heads, ≈22M parameters.

**Distillation token.** DeiT introduces a learnable “distillation token” in addition to the CLS token, resulting in a sequence [dist] + [CLS] + patches. The distillation token is supervised by teacher predictions; the CLS token is supervised by ground‑truth labels. Separate heads read each token; their outputs are combined at inference (e.g., averaged).

### Distillation findings

Contrary to conventional distillation (Hinton et al.), DeiT finds:

- **Hard‑label distillation** (using argmax teacher labels) performs better than soft‑label distillation for DeiT‑B on ImageNet‑1k.  
- CNN teachers (e.g., RegNetY) transfer inductive biases (locality, translation equivariance) particularly well to ViT students.  
- A moderately accurate CNN teacher can outperform a higher‑accuracy transformer teacher in terms of student performance.

The paper suggests that distillation confers CNN‑like priors to the ViT student, improving stability and sample efficiency.

### Training recipe (the core innovation)

DeiT’s performance gains come primarily from an aggressive training recipe rather than architectural changes:

- Optimiser: AdamW, base LR ≈ 5×10⁻⁴ scaled with batch size/512, weight decay 0.05.  
- LR schedule: linear warmup (e.g., 5 epochs), then cosine decay over 300 epochs.  
- No standard dropout; instead, stochastic depth (≈0.1) and strong data augmentation.  
- Augmentations: RandAugment, Mixup (α~0.8), CutMix (α~1.0), random erasing, repeated augmentation.  
- Regularisation: label smoothing (ε=0.1), stochastic depth.

These combined techniques raise ViT‑B/16’s ImageNet‑1k performance into the low‑80s without external data.

### Key results

Representative results from DeiT:

- DeiT‑B/16 (no distillation): **≈81.8%** top‑1 at 224×224, ≈83%+ at 384×384 after fine‑tuning.  
- DeiT‑B/16 with distillation token and hard‑label distillation: up to ≈83–85% top‑1 with 384×384 fine‑tuning, comparable to or surpassing original ViT‑B trained with JFT pretraining (depending on exact configuration).  
- DeiT‑S and DeiT‑Ti provide competitive trade‑offs at smaller sizes; e.g., DeiT‑S exceeds 80% ImageNet top‑1 with distillation.

Exact numbers depend on evaluation (single‑crop vs multi‑crop, resolution), but the central result is that a **convolution‑free transformer** can be trained on ImageNet‑1k alone to match or exceed CNN baselines using a single node and a few days of training.

### What was superseded

Later work refined or replaced pieces of DeiT:

- **DeiT‑III** and scaling work show that simple cross‑entropy and streamlined augmentations can suffice when scaling to larger ViT models; the distillation token can be removed for large‑scale training.  
- Self‑supervised pretraining (e.g., BEiT, MAE, DINO, iBOT) became more prominent for ViT at scale.  
- Architectures like Swin Transformer (using windowed attention) and ConvNeXt (CNNs with modernised design) reached competitive accuracy with better efficiency.

### Insights

- DeiT demonstrates that **training recipe and distillation can be as important as architecture** for ViTs on standard‑size datasets.  
- Distillation from CNN teachers effectively injects convolution‑like inductive biases into a pure transformer.  
- The distillation token and CLS token behave differently; their representations are not redundant, and combining them improves performance over duplicating CLS‑style tokens.

---

## DINO: self‑supervised ViTs learn to segment without being told

**Paper:** “Emerging Properties in Self‑Supervised Vision Transformers,” Caron et al., ICCV 2021, arXiv:2104.14294.

### Core problem

Transformers in NLP benefited enormously from self‑supervised pretraining. Early ViT work, however, focused mostly on supervised training, and the benefits over CNNs were mixed on standard data. DINO (self‑**DI**stillation with **NO** labels) asks whether self‑supervised learning unlocks qualitatively new behaviours in ViTs compared to convnets.

The answer: self‑supervised ViTs trained with DINO show emergent properties, including:

- Attention maps that align closely with object regions, enabling **semantic segmentation from attention** without any segmentation labels.  
- Representations so structured that **k‑NN classifiers** can reach ImageNet accuracies close to linear probes, indicating very good instance‑space geometry.

### Method

DINO uses a student–teacher framework:

- Student and teacher share the same backbone architecture (ViT or ResNet) plus a 3‑layer MLP **projection head** with 2048 hidden units and a high‑dimensional output (e.g., 65,536).  
- The student receives gradients; the teacher is an **exponential moving average** (EMA) of the student:  
  `theta_teacher <- lambda theta_teacher + (1-lambda) theta_student` with momentum λ cosine‑scheduled from ~0.996 to near 1.0.  
- The student is trained with cross‑entropy to match the teacher’s output distributions.

### Multi‑crop augmentation

DINO relies heavily on multi‑crop training:

- Teacher sees **two global crops** (e.g. 224×224 covering >50% of image area).  
- Student sees both global crops and **multiple local crops** (e.g. eight 96×96 crops covering smaller regions).  

The student is trained so that its predictions for any crop are consistent with the teacher’s predictions on global views, encouraging strong local‑to‑global alignment.

### Preventing collapse without contrastive pairs

Unlike contrastive methods (SimCLR, MoCo), DINO does not explicitly use negatives; it avoids trivial collapse by combining:

- **Centering:** subtracting a moving average of teacher outputs (EMA with momentum ~0.9) to discourage all outputs collapsing to a constant vector.  
- **Sharpening:** using a low teacher temperature (e.g., 0.04→0.07 over a warm‑up period) and a higher student temperature (e.g., 0.1), which makes teacher outputs peaked while student outputs are smoother, balancing entropy.

This combination yields a non‑collapsed equilibrium without requiring contrastive logits or batch‑normalisation hacks.

### Backbones and training

DINO evaluates:

- ViT‑S/16 (small, 16×16 patches), ViT‑S/8 (same parameters, double resolution of tokens), ViT‑B/16, ViT‑B/8.  
- ResNet‑50 and other CNN backbones for comparison.

Training details include AdamW, batch size ≈1024, 300 epochs, cosine LR schedule with warm‑up, and cosine‑scheduled weight decay.

### Key results

On ImageNet linear probing:

- DINO with ViT‑S/16: ~77% top‑1.  
- ViT‑B/16: ~78% top‑1.  
- ViT‑S/8: ~79–80% top‑1.  
- ViT‑B/8: ~80%+ top‑1.

Using a **k‑NN classifier** on frozen DINO features achieves results close to linear probing, e.g. ViT‑S/8 reaching high‑70s top‑1, narrowing the typical gap observed in CNN‑based SSL.

For segmentation‑like tasks, attention maps from the last layers of DINO‑trained ViTs can be thresholded to obtain high‑quality object masks without segmentation labels, outperforming supervised ViTs on certain metrics like VOC segmentation from attention.

### What was superseded

Later work extended DINO:

- **DINOv2** scales data (≈100M+ curated images) and model size (ViT‑L, ViT‑g), combining DINO‑style global objectives with patch‑level tasks (as in iBOT) and adding register tokens to stabilise attention in low‑information regions.  
- **MAE** and similar masked image modelling approaches showed superior scalability for very large models and long training schedules, though DINO features often remain competitive for frozen‑backbone transfer.

Limitations of original DINO include training instability beyond ImageNet scale, lack of explicit patch‑level prediction (only global), and the computational cost of small patch sizes (e.g., 8×8) which increase sequence length.

### Why it matters

DINO demonstrates that **self‑supervised learning and ViTs are synergistic**:

- ViTs trained with DINO exhibit stronger emergent spatial organisation than analogous CNNs.  
- Attention heads naturally align with object regions and parts, even under occlusion, without segmentation labels.  
- DINO’s EMA student–teacher framework and multi‑crop training influenced subsequent SSL methods and the design of vision foundation models.

---

## Conclusion

These four papers chart a path of gradually **removing hand‑crafted priors** while scaling data and models:

- **LeNet‑5** replaces hand‑engineered features with learned convolutions but still uses hand‑designed connection patterns, activation scaling, and RBF outputs.  
- **VGG** strips away filter‑size heterogeneity, showing that repeated 3×3 conv blocks and depth alone can reach top ImageNet performance, with a simple and generalisable architecture.  
- **DeiT** removes convolutions completely in the backbone, proving that with the right training recipe and distillation, pure transformers can be data‑efficient on standard ImageNet‑1k.  
- **DINO** removes labels, showing that a self‑distilled ViT trained with SSL learns rich semantics and even segmentation‑like behaviour emergently, surpassing what supervised ViTs alone displayed.

For a practitioner, the design lessons are:

- Use **simple, repeated motifs** (LeNet/VGG) and let learning do the heavy lifting.  
- Exploit **residual attention architectures** where global context can be learned (ViT/DeiT).  
- Combine architecture with **strong training recipes and distillation** when data is limited (DeiT).  
- Leverage **self‑supervision plus ViTs** for general‑purpose, label‑efficient representation learning (DINO and successors).
