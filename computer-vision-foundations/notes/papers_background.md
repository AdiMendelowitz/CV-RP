# Four papers that shaped visual recognition

These reading notes trace a **28-year arc** from the first practical CNN to self-supervised vision transformers that learn to segment without labels. LeNet (1998) proved end-to-end learning from pixels could work at industrial scale. VGG (2014) showed that depth with small filters beats width with large ones. DeiT (2021) democratized vision transformers by ditching the need for 300M-image pretraining. DINO (2021) revealed that self-supervised ViTs spontaneously develop spatial understanding no supervised model achieves. Together, these papers encode the key design lessons a practitioner needs to understand how modern vision architectures got here — and why.

---

## LeNet-5: the blueprint every CNN still follows

**Paper**: "Gradient-Based Learning Applied to Document Recognition" — LeCun, Bottou, Bengio, Haffner (1998). Proceedings of the IEEE, 86(11).

Note: The "5" in LeNet-5 is the version number, not the layer count.

**Core problem.** Before LeNet, pattern recognition required a hand-designed feature extractor bolted onto a trainable classifier. Feature engineering was the bottleneck; performance was capped by the designer's intuition. LeCun et al. asked: can a single gradient-trained system learn features and classify in one pass? The answer was a convolutional network deployed at NCR to read **millions of bank checks daily** across US banks.

**Architecture (LeNet-5).** The network takes **32×32 grayscale** input (28×28 MNIST digits zero-padded to allow corner features to sit at receptive field centers). The layer stack: C1 (6 feature maps, **5×5** kernels) → S2 (2×2 subsampling) → C3 (16 maps, 5×5) → S4 (2×2 subsampling) → C5 (120 maps, 5×5) → F6 (84 units) → RBF output (10 classes). Total: **~60,000 trainable parameters** and ~340K connections. The activation is a **scaled tanh**: f(a) = 1.7159 · tanh(2a/3), deliberately chosen so that f(±1) = ±1 and the second derivative peaks at ±1. The output layer uses **Euclidean RBF units** measuring L2 distance to fixed 7×12 bitmap prototypes of each digit — not softmax. This is why F6 has exactly **84 = 7 × 12** units.

Three design details are routinely misunderstood. First, the S2/S4 "pooling" layers are **not** simple average pooling — each has a trainable scale and bias per feature map, passed through the activation function. Second, C3 connects to S2 via a **sparse connection table**: the 16 output maps connect to subsets of 3–6 input maps in a specific pattern to break symmetry and force complementary feature learning. Third, C5 is labeled "convolutional" but is **effectively fully connected** because S4 outputs are exactly 5×5 and the kernel is 5×5. The parameter budget was intentionally matched to the 60K training samples — a deliberate capacity control rooted in VC theory.

**Key results.** LeNet-5 achieved **0.95% error** on MNIST without augmentation, **0.8%** with distortions, and a boosted LeNet-4 ensemble hit **0.7%** — the best in the paper. For context, SVMs with polynomial kernels scored 1.1%, and a 2-layer MLP with 300 hidden units scored 1.6%. The paper's rejection analysis showed CNNs needed the fewest rejected samples to reach a target 0.5% error rate.

**What was superseded.** Almost every specific choice: scaled tanh → **ReLU**, average subsampling → **max pooling**, RBF output → **softmax + cross-entropy**, MSE loss → cross-entropy, sparse connections → full connectivity, diagonal Hessian SGD → **Adam/AdamW**. Modern reimplementations silently replace most of these. MNIST itself became trivially solved (best error now ~0.17%).

**Why it matters.** LeNet established the **conv → pool → conv → pool → FC** paradigm that served as the blueprint for AlexNet, VGG, and every major CNN until transformers arrived. More importantly, the full 46-page paper introduced Graph Transformer Networks (a precursor to differentiable programming), Space Displacement Neural Networks (a precursor to fully convolutional networks), and word-level training — ideas that took a decade to rediscover.


---

## VGGNet: depth wins, and 3×3 is all you need

**Paper**: "Very Deep Convolutional Networks for Large-Scale Image Recognition" — Simonyan & Zisserman (2014). ICLR 2015. arXiv:1409.1556.

**Core problem.** After AlexNet (2012) proved deep learning worked on ImageNet, the question was what mattered most for accuracy. AlexNet used 11×11 and 5×5 kernels. ZFNet used 7×7. VGG's hypothesis was very simple: **fix everything else and only increase depth, using exclusively 3×3 convolutions**. The paper systematically tested networks from 11 to 19 weight layers.

**Architecture.** VGG-16 (Config D) stacks five blocks of 3×3 convolutions with channel counts **64 → 128 → 256 → 512 → 512**, each block ending in 2×2 max pooling with stride 2. Block structure: 2-2-3-3-3 conv layers, followed by three FC layers (4096-4096-1000) and softmax, overall has **138M parameters**. VGG-19 (Config E) adds one conv per block in the last three blocks (2-2-4-4-4). All convolutions use stride 1 with 1-pixel padding to preserve spatial dimensions. ReLU activations everywhere (Batch Normalization did't exist yet), overall **has 144M parameter.** The paper explicitly tested and rejected Local Response Normalization — "does not improve performance but leads to increased memory consumption."

The central insight: 3 stacked 3×3 conv layers have the same effective receptive field as one 7×7 layer, but with **three ReLU nonlinearities** instead of one and **27C² parameters** vs **49C²** — a **45% parameter reduction** with more representational power. VGG directly tested this: a shallow network with 5×5 convs (equivalent receptive field to VGG-13) had **7% higher top-1 error**, confirming deep-and-small beats shallow-and-large.

**Training specifics.** SGD with momentum **0.9**, batch size **256**, weight decay **5×10⁻⁴**, dropout **0.5** on the first two FC layers. Initial LR **10⁻²**, decreased by 10× three times. Training took **2–3 weeks on 4 Titan Black GPUs**, 74 epochs. Critical trick: Config A was trained from scratch, then its weights initialized all deeper configs — this was essential pre-batch-normalization (the authors later noted Xavier init works too). Multi-scale training with S ∈ [256, 512] was a significant win: VGG-16 top-5 dropped from 8.8% to **8.1%** single-scale and **7.5%** multi-scale.

**Key results.** Single-model VGG-16 with multi-scale dense+multi-crop evaluation: **24.4% top-1, 7.2% top-5** on ImageNet validation. A 2-net ensemble (D+E): **23.7% top-1, 6.8% top-5** — within 0.1% of GoogLeNet's 7-net winning submission (**6.7%**). VGG placed **2nd in classification, 1st in localization** (25.3% vs GoogLeNet's 26.7%). As a single model, VGG beat GoogLeNet: **7.0% vs 7.9% top-5**. Transfer learning with frozen VGG features + linear SVM (no fine-tuning) achieved **89.7 mAP** on VOC-2007 and **92.7% recall** on Caltech-101.

**What was superseded.** VGG's **138M parameters** are grotesquely inefficient — GoogLeNet achieved comparable accuracy with **6.8M** (20× fewer), ResNet-50 with **25.6M** (5.4× fewer). About **89% of VGG-16's parameters sit in the FC layers** (FC1 alone: 7×7×512×4096 = 102M params). Later architectures replaced FC layers with global average pooling. VGG-16 costs **~15.3B FLOPs** vs ResNet-50's **3.8B**. The paper hit a depth wall at 19 layers — VGG-19 showed no improvement over VGG-16.

**Insights:** **3×3 convolutions were the universal default** for nearly a decade (2014–2022), established the channel-doubling-after-pooling pattern, and proved that simple, uniform, modular block design could be competitive with complex multi-branch architectures (Inception). Its clean hierarchical features remain the standard backbone for neural style transfer and perceptual loss functions.
 VGG's dense evaluation trick — converting FC layers to 7×7 and 1×1 convolutions to apply the network fully convolutionally — requires only 2 forward passes (original + flip) vs multi-crop's 150, with comparable accuracy. Combining both is complementary (+0.3% top-5) because they have different boundary conditions (zero-padding vs natural neighbor padding). The LRN result killed that technique's adoption across the field.

---

## DeiT: training ViTs without 300M images

**Paper**: "Training data-efficient image transformers & distillation through attention" — Touvron, Cord, Douze, Massa, Sablayrolles, Jégou (2021). ICML 2021. arXiv:2012.12877.

**Core problem:** ViT (Dosovitskiy et al., 2020) achieved strong results but only when pretrained on JFT-300M, a private dataset of 300 million labeled images. Trained on ImageNet-1k alone, ViT-B/16 managed just **77.9% top-1** at 384 resolution. DeiT asked whether careful training recipes and knowledge distillation could close this gap using only ImageNet-1k on a single 8-GPU node.

**Architecture:** DeiT-B is architecturally **identical to ViT-B** — 12 layers, **768** embedding dim, **12** heads, 64 dim/head, MLP ratio 4× (768→3072→768), **86M parameters**. The family includes DeiT-Ti (192 dim, 3 heads, **5M** params) and DeiT-S (384 dim, 6 heads, **22M** params). Patch size **16×16** on **224×224** input yields 196 tokens. The architectural novelty is a **distillation token**: a learnable embedding appended alongside the class token, making the sequence length N+2. The class token is supervised by ground-truth labels; the distillation token is supervised by the teacher's prediction. At inference, softmax outputs from separate linear heads on each token are averaged.

**Distillation Contribution:** Counter to Hinton et al.'s conventional wisdom, **hard-label distillation significantly outperforms soft-label distillation** (+1.2% for DeiT-B at 224). Hard distillation simply uses argmax of the teacher's output as a pseudo-label, meaning no temperature tuning, no KL divergence. The explanation: hard labels interact better with label smoothing and data augmentation, since the teacher's decision adapts to augmented images where the ground truth may not match visible content. Label smoothing (ε=0.1) is applied to true labels but **not** to teacher pseudo-labels.

**The teacher matters, and its architecture matters more than its accuracy:** The default teacher is **RegNetY-16GF** (84M params, 82.9% top-1). A CNN teacher consistently outperforms a transformer teacher, even when the transformer is stronger. RegNetY-4GF at **80.0%** accuracy produces a better student than DeiT-B at **81.8%** as teacher. The hypothesis: distillation transfers the CNN's inductive biases (locality, translation equivariance) that the transformer lacks, effectively soft-coding convolution-like priors through attention.

**Training recipe (the real innovation):** The same ViT-B architecture improved from 77.9% to **81.8%** purely through training changes and zero architectural modification. Key ingredients: AdamW with LR **5×10⁻⁴** × batchsize/512, cosine decay, weight decay **0.05**, **no dropout** (stochastic depth 0.1 instead), batch size **1024**, 300 epochs with 5-epoch warmup, RandAugment 9/0.5, Mixup 0.8, CutMix 1.0, random erasing 0.25, **repeated augmentation** (3 repetitions), label smoothing 0.1. Removing stochastic depth or random erasing caused accuracy to crater to ~4%. Training DeiT-B takes **53 hours on 8 V100s**.

**Key results:** DeiT-B without distillation: **81.8%** top-1 at 224, **83.1%** at 384 (fine-tuned 25 epochs). With hard distillation + distillation token at 1000 epochs: **85.2%** at 384, surpassing ViT-B pretrained on JFT-300M (**84.15%**) by over a point, using **234× less data**. DeiT-S: **79.8%** without distillation, **81.2%** with. Transfer results: **99.1%** CIFAR-10, **90.8%** CIFAR-100. Throughput: DeiT-B at 224 processes **292 images/sec** on a V100.

**What was superseded:** DeiT-III (2022) showed the training recipe doesn't scale to ViT-L/H. It replaced cross-entropy with **binary cross-entropy**, simplified augmentation to just 3 operations (grayscale, solarize, Gaussian blur), and dropped the distillation token entirely, achieving 85.2% with ViT-H at 224 through pure supervised training. Self-supervised methods (BEiT, MAE) showed that masked image modeling became the dominant pretraining paradigm for large ViTs. Swin Transformer and ConvNeXt demonstrated that architectural inductive biases or modernized convolutions could match or beat vanilla ViT without distillation.

**Insights:** 
 - DeiT proved vision transformers are data-efficient when trained correctly, and that **training recipe matters more than architecture**.
 - The distillation token and class token converge to cosine similarity of only **0.06** at input, rising to **0.93** at the last layer. Adding a second class token (same target) yields cos=0.999 and no performance gain. The two tokens provide genuinely complementary signals: the distillation token's attention correlates more with the CNN teacher, while the class token retains transformer-native global patterns.

---

## DINO: self-supervised ViTs learn to segment without being told

**Paper**: "Emerging Properties in Self-Supervised Vision Transformers" — Caron, Touvron, Misra, Jégou, Mairal, Bojanowski, Joulin (2021). ICCV 2021. arXiv:2104.14294.

**Core problem:** Transformers succeeded in NLP via self-supervised pretraining (BERT, GPT), yet vision transformers were trained supervised and showed no clear advantage over CNNs. DINO (self-**DI**stillation with **NO** labels) asked: does SSL unlock new properties in ViTs that supervised training does not? The answer was two striking emergent behaviors: attention maps that perform **semantic segmentation without any segmentation supervision**, and features so well-structured that a **k-NN classifier (k=20) hits 78.3% on ImageNet**.

**Method:** Student and teacher share the **exact same architecture** (backbone + 3-layer MLP projection head with 2048 hidden dim and **K=65,536** output dim, L2-normalized, weight-normalized final layer). The student minimizes cross-entropy against the teacher's output distribution. The teacher receives **no gradients** but is updated via exponential moving average: θ_t ← λθ_t + (1−λ)θ_s, with **λ cosine-scheduled from 0.996 to 1.0**. The teacher consistently outperforms the student throughout training.

**Multi-crop augmentation:** feeds the teacher **2 global crops** at **224×224** (>50% image area) and the student **all views** including **8 local crops** at **96×96** (<50% area). This forces local-to-global correspondence: the student must infer global semantics from small patches.

**Collapse avoidance:** uses only two mechanisms, no contrastive pairs, no clustering, no batch norm dependence. **Centering** subtracts an EMA-updated mean (momentum **0.9**) from teacher outputs, preventing dimension collapse. **Sharpening** uses a low teacher temperature (**τ_t warm-up from 0.04 to 0.07** over 30 epochs; student τ_s fixed at **0.1**), making outputs peaked and opposing the uniform-distribution tendency of centering. These opposing forces balance each other.

**Backbone variants tested:** ViT-S/16 (384 dim, 6 heads, **21M** params), ViT-S/8 (same params, 4× more tokens), ViT-B/16 (**85M**), ViT-B/8, and ResNet-50 (23M). Training: AdamW, batch size 1024, LR **0.0005 × batchsize/256** with 10-epoch warmup then cosine decay, weight decay cosine-scheduled **0.04 → 0.4**, 300 epochs. ViT-S/16 trains in **2.6 days on 16 GPUs**.

**Key results.** Linear evaluation on ImageNet: ViT-S/16 **77.0%**, ViT-B/16 **78.2%**, ViT-S/8 **79.7%**, ViT-B/8 **80.1%**. ViT-B/8 set SOTA with 10× fewer parameters than prior leaders (BYOL RN200w2: 79.6% with 250M params). The k-NN results are the headline: ViT-S/16 **74.5%**, ViT-S/8 **78.3%**, ViT-B/16 **76.1%**. The gap between k-NN and linear probe narrows to **2.5 points** for DINO+ViT, vs **7.8 points** for DINO+ResNet-50 (67.5% vs 75.3%) and 8–10 points for MoCo/BYOL. On DAVIS 2017 video segmentation (frozen features, nearest-neighbor frame matching, no fine-tuning): ViT-B/8 scores **71.4 (J&F)m**, beating specialized methods trained on Kinetics. PASCAL VOC12 segmentation from thresholded attention maps: **45.9 Jaccard** for DINO ViT-S/16 vs **27.3** for supervised ViT-S/16.

**What was superseded:** DINOv2 (2023) scaled to **142M curated images** and **ViT-g (1.1B params)**, combined DINO's global objective with iBOT's patch-level masked image modelling, added register tokens to fix attention artifacts, and achieved k-NN **82.0%** and linear **84.5%** with ViT-L/14. MAE (2021) showed generative SSL scales better to very large models and long training schedules (1600 epochs without saturation), though its frozen features are weaker than DINO's since MAE requires fine-tuning. DINO's limitations: training instability beyond ImageNet scale, no explicit patch-level objective (patch tokens learn indirectly), and the /8 variants that give the best features are **5–15× slower** than /16.

**Why it matters:** DINO proved that **SSL + ViTs is qualitatively more powerful than either alone**, producing representations with emergent spatial understanding that supervised training cannot achieve, and laid the architectural and methodological foundation for the DINOv2 family of universal vision foundation models.

**Insights:** 
- Use the **teacher checkpoint** for inference as it consistently beats the student. For classification, use the [CLS] token from the backbone (not the projection head). For retrieval, concatenate [CLS] + GeM-pooled patch tokens. For segmentation or video tracking, use patch token features with nearest-neighbour matching. 
- Patch size has an outsized effect: going from /16 to /8 **adds zero parameters** but improves k-NN by **3.8 points** and DAVIS by **9.1 points** at the cost of 5× throughput. 
- The attention maps reveal that different heads in the last layer specialize in different objects or object parts, even under occlusion. 
- Later work ("Vision Transformers Need Registers," Darcet et al., 2023) discovered that DINO-trained ViTs develop artifact tokens in low-information background patches; adding explicit register tokens fixes this.

---

## Conclusion

The papers demonstrate progressive **removal of hand-crafted priors**. LeNet replaced hand-designed features with learned convolutions but still hand-designed the connection topology, activation scaling, and output encoding. VGG stripped away even the variety in kernel sizes, showing a single repeated motif (3×3 conv + ReLU) could dominate. DeiT removed convolutions entirely, showing a pure attention architecture could match CNNs, but only by distilling a CNN teacher's inductive biases back in. DINO completed the circle by removing labels altogether, revealing that the richest visual representations emerge when architecture and learning objective are both unconstrained.

