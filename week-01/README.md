# Week 1: Computer Vision Foundations
**Dates:** 01/02/2026 to 08/02/2026  
**Status:** 🔄 Not Started | 🔄 In Progress | ✅ Complete  
**Time Commitment:** 25-30 hours (4-5 hours/day)

---

## Week Overview

**Learning Goals:**
- Understand classical computer vision operations
- Implement CNNs from scratch (NumPy)
- Build modern architectures (ResNet, ViT)
- Explore self-supervised learning

**Deliverables:**
- [] GitHub repo: `computer-vision-foundations`
- [ ] 5 implementations (classical CV, CNN, ResNet, ViT, SimCLR)
- [ ] Blog draft: "Vision Transformers vs CNNs"
- [ ] 8-10 papers read and summarized

---

## 📚 Papers to Read This Week

Download and save to `../papers/week-01/`:

### Must Read (Deep Understanding)
- [ ] **AlexNet** (2012) - ImageNet Classification with Deep CNNs
  - Link: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
- [ ] **ResNet** (2015) - Deep Residual Learning
  - Link: https://arxiv.org/abs/1512.03385
- [ ] **ViT** (2020) - An Image is Worth 16x16 Words  
  - Link: https://arxiv.org/abs/2010.11929
- [ ] **SimCLR** (2020) - A Simple Framework for Contrastive Learning
  - Link: https://arxiv.org/abs/2002.05709

### Skim (Background)
- [ ] LeNet (1998) - Gradient-Based Learning
- [ ] VGG (2014) - Very Deep Networks
- [ ] DeiT (2021) - Data-Efficient Image Transformers
- [ ] DINO (2021) - Self-Distillation with No Labels

**Reading Notes:** Create `notes/papers-week1.md`

---

## Monday: Classical Computer Vision

**Goal:** Understand fundamentals before deep learning

### Morning Session (3 hours)
- [x] Watch Stanford CS231n Lecture 1 (1 hour)
  - Link: http://cs231n.stanford.edu/
- [x] Watch Stanford CS231n Lecture 2 (1 hour)
- [x] Read Szeliski Chapter 3: Image Processing (30 min)
  - Link: http://szeliski.org/Book/
- [x] **Notes:** Create `notes/day1-classical-cv.md`

### Afternoon Session (3-4 hours)
- [x] Create folder: `code/classical_cv/`
- [x] **Implement** `filters.py`:
  - [x] Gaussian blur (without OpenCV)
  - [x] Sobel edge detection
  - [x] Canny edge detector
- [x] **Implement** `transforms.py`:
  - [x] Affine transformations
  - [x] Perspective transforms
- [x] Compare your implementations with OpenCV
- [ ] Write unit tests

### Evening Session (1 hour)
- [x] Create Jupyter notebook: `notebooks/01_classical_cv.ipynb`
- [x] Visualize: Original → Gaussian → Sobel → Canny pipeline
- [x] Add mathematical explanations
- [x] **Commit to GitHub**

**Daily Reflection:**
```
What I learned:


Challenges:


Questions for tomorrow:
```

---

## Tuesday: Convolutional Neural Networks from Scratch

**Goal:** Implement CNN in NumPy to understand internals

### Morning Session (3 hours)
- [x] Read Stanford CS231n notes on CNNs
- [x] Watch CS231n Lecture 5: CNNs (1 hour)
- [x] Study backpropagation through conv layers
- [x] **Derive** forward and backward pass math (pen & paper)
- [x] **Notes:** `notes/day2-cnn-math.md`

### Afternoon Session (4-5 hours)
- [x] Create folder: `code/cnn_scratch/`
- [x] **Implement** `layers.py`:
  - [x] `Conv2D` layer with forward/backward
  - [x] `MaxPool2D` layer
  - [x] `Dense` layer
  - [x] `ReLU`, `Softmax` activations
- [x] **Implement** `network.py`:
  - [x] CNN class (stacks layers)
  - [x] Forward pass
  - [x] Backward pass (backpropagation)
- [x] **Implement** `train.py`:
  - [x] SGD optimizer
  - [x] Training loop
  - [x] Loss tracking

### Evening Session (1-2 hours)
- [x] Test on MNIST dataset
- [x] Target: >95% accuracy after 5 epochs
- [x] Create notebook: `notebooks/02_cnn_from_scratch.ipynb`
- [x] Document: architecture, derivations, results
- [x] Visualize: learned filters from first conv layer
- [x] **Commit to GitHub**

**Code Quality Check:**
- [x] Add docstrings to all functions
- [x] Add type hints
- [x] Run: `black code/cnn_scratch/`

---

## Wednesday: PyTorch & ResNet

**Goal:** Modern tools and residual connections

### Morning Session (3 hours)
- [ ] Complete PyTorch tutorials:
  - [ ] "Learn the Basics" section
  - [ ] Link: https://pytorch.org/tutorials/
- [ ] Read ResNet paper (focus on architecture section)
- [ ] **Study**: Why residual connections solve vanishing gradients
- [ ] **Notes:** `notes/day3-resnet-theory.md`

### Afternoon Session (4 hours)
- [x] Create folder: `code/pytorch_cnn/`
- [x] **Rebuild** yesterday's CNN in PyTorch
  - [x] Verify it matches NumPy version results
- [x] **Implement** `resnet.py`:
  - [x] `BasicBlock` (residual block)
  - [x] `ResNet` class (full architecture)
  - [x] Build ResNet-18 from scratch
- [x] **Implement** `train_cifar.py`:
  - [x] CIFAR-10 data loading
  - [x] Data augmentation (RandomCrop, RandomHorizontalFlip)
  - [x] Training loop with logging

### Evening Session (2 hours)
- [ ] Set up Weights & Biases (W&B)
  - [ ] Create account: https://wandb.ai/
  - [ ] Initialize: `wandb init`
- [ ] Start ResNet-18 training on CIFAR-10
  - [ ] Target: >85% test accuracy
  - [ ] Log to W&B: loss, accuracy, learning rate
- [ ] Create notebook: `notebooks/03_resnet_training.ipynb`
- [ ] **Commit to GitHub**

**Training Monitoring:**
- [ ] Monitor training overnight
- [ ] Save checkpoints every 10 epochs
- [ ] Track: train loss, val loss, train acc, val acc

---

## Thursday: Vision Transformers

**Goal:** Understand attention mechanism for vision

### Morning Session (3 hours)
- [ ] Read ViT paper thoroughly
  - [ ] Focus: patch embeddings, position embeddings
  - [ ] Focus: multi-head self-attention
- [ ] Review attention mechanism from "Attention Is All You Need"
- [ ] **Study**: Why ViT needs more data than CNNs
- [ ] **Notes:** `notes/day4-vit-architecture.md`

### Afternoon Session (4-5 hours)
- [ ] Create folder: `code/vision_transformers/`
- [ ] **Implement** `vit.py`:
  - [ ] `PatchEmbedding` (image → patches → embeddings)
  - [ ] `MultiHeadAttention` (self-attention mechanism)
  - [ ] `TransformerBlock` (attention + MLP + residuals)
  - [ ] `VisionTransformer` (full ViT)
- [ ] **Configure** ViT-Tiny for CIFAR-10:
  - Patch size: 4x4
  - Embedding dim: 192
  - Depth: 12 layers
  - Heads: 3
- [ ] Start training ViT-Tiny

### Evening Session (2 hours)
- [ ] **Compare** ViT vs ResNet:
  - [ ] Training curves
  - [ ] Parameter count
  - [ ] Final accuracy
  - [ ] Training time
- [ ] Visualize attention maps
- [ ] Create notebook: `notebooks/04_vit_analysis.ipynb`
- [ ] **Start blog draft:** `../blog-drafts/vit-vs-cnn.md`
- [ ] **Commit to GitHub**

**Analysis Questions:**
- When does ViT outperform CNNs?
- When should you use CNNs vs ViT?
- What's the data efficiency difference?

---

## Friday: Self-Supervised Learning

**Goal:** Learn representations without labels

### Morning Session (3 hours)
- [ ] Read SimCLR v2 paper
- [ ] Study contrastive learning framework
- [ ] Understand: positive pairs, negative pairs, NT-Xent loss
- [ ] Watch explanation video (optional)
  - AI Coffee Break with Letitia: SimCLR
- [ ] **Notes:** `notes/day5-contrastive-learning.md`

### Afternoon Session (4-5 hours)
- [ ] Create folder: `code/self_supervised/`
- [ ] **Implement** `simclr.py`:
  - [ ] `SimCLR` class (encoder + projection head)
  - [ ] Data augmentation pipeline (critical!)
    - RandomResizedCrop
    - ColorJitter
    - GaussianBlur
    - RandomHorizontalFlip
  - [ ] `nt_xent_loss` (temperature-scaled contrastive loss)
- [ ] **Implement** `train_simclr.py`:
  - [ ] Pretraining loop
  - [ ] Generate positive pairs
  - [ ] Compute contrastive loss
- [ ] **Implement** `linear_eval.py`:
  - [ ] Freeze encoder
  - [ ] Train linear classifier
  - [ ] Evaluate on test set

### Evening Session (2 hours)
- [ ] Start SimCLR pretraining (100 epochs)
  - [ ] Let it run overnight
- [ ] Create notebook: `notebooks/05_simclr.ipynb`
- [ ] Document: augmentation strategies, temperature effects
- [ ] **Commit to GitHub**

**Experiments to Track:**
- Pretrain loss over time
- Impact of different augmentations
- Temperature parameter effects
- Linear eval accuracy vs random init

---

## Saturday: Code Cleanup & Documentation

**Goal:** Professional-quality repository

### Morning Session (3 hours)
- [ ] **Code Formatting:**
  - [ ] Run `black .` on all Python files
  - [ ] Run `flake8 .` and fix warnings
  - [ ] Add type hints to function signatures
- [ ] **Documentation:**
  - [ ] Add docstrings (Google style) to all classes/functions
  - [ ] Create `code/README.md` explaining each module
- [ ] **Testing:**
  - [ ] Write unit tests for core functions
  - [ ] Test: `pytest code/tests/`

### Afternoon Session (3 hours)
- [ ] **Create Project README:**
  - [ ] Project overview
  - [ ] Installation instructions
  - [ ] Usage examples
  - [ ] Results table (ResNet vs ViT vs SimCLR)
  - [ ] Links to notebooks
- [ ] **Organize Notebooks:**
  - [ ] Clean output cells
  - [ ] Add markdown explanations
  - [ ] Ensure reproducibility
- [ ] **Create Results Summary:**
  - [ ] Table: Model | Params | Accuracy | Training Time
  - [ ] Plots: Training curves comparison

### Evening Session (2 hours)
- [ ] **GitHub Repository:**
  - [ ] Create repo: `computer-vision-foundations`
  - [ ] Write comprehensive README
  - [ ] Add LICENSE (MIT recommended)
  - [ ] Push all code
  - [ ] Create releases/tags
- [ ] **Blog Post:**
  - [ ] Finish "Vision Transformers vs CNNs" draft
  - [ ] Add code snippets
  - [ ] Add visualizations
  - [ ] Save in `../blog-drafts/`

**Repository Checklist:**
- [ ] Clear README with installation
- [ ] requirements.txt
- [ ] .gitignore configured
- [ ] All code committed
- [ ] Notebooks with outputs
- [ ] Results documented

---

## Sunday: Reflection & Week 2 Prep

**Goal:** Review progress and prepare for next week

### Morning Session (2 hours)
- [ ] **Week 1 Reflection:**
  - [ ] What worked well?
  - [ ] What was challenging?
  - [ ] What would I do differently?
  - [ ] Key learnings?
- [ ] **Update** `../PROGRESS.md`:
  - [ ] Check off Week 1
  - [ ] Add metrics (commits, papers, hours)
- [ ] **Review** all Week 1 notebooks
  - [ ] Ensure they run top-to-bottom
  - [ ] Add missing explanations

### Afternoon Session (2 hours)
- [ ] **Read Week 2 Overview:**
  - [ ] Efficient models (EfficientNet, ConvNeXt)
  - [ ] Model compression
  - [ ] Deployment optimization
- [ ] **Download Week 2 Papers:**
  - EfficientNet, ConvNeXt, quantization papers
  - Save to `../papers/week-02/`
- [ ] **Set up Week 2 folder:**
  - [ ] Create `week-02/` structure
  - [ ] Copy successful templates from Week 1

### Evening Session (1 hour)
- [ ] **Share Progress:**
  - [ ] Tweet/LinkedIn: "Finished Week 1 of 12-week ML plan"
  - [ ] Share GitHub repo
  - [ ] Optional: Publish blog post draft
- [ ] **Prepare Monday:**
  - [ ] Review Monday's tasks
  - [ ] Download any needed data
  - [ ] Set up W&B experiments

**Week 1 Completion Checklist:**
- [ ] 5 implementations complete
- [ ] ResNet >85% CIFAR-10 accuracy
- [ ] GitHub repo public
- [ ] Blog draft written
- [ ] 8+ papers read
- [ ] Notes organized
- [ ] Ready for Week 2

---

## Resources for Week 1

### Online Courses
- **Stanford CS231n**: http://cs231n.stanford.edu/
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **Fast.ai Part 1**: https://course.fast.ai/

### Documentation
- **PyTorch Docs**: https://pytorch.org/docs/
- **NumPy Docs**: https://numpy.org/doc/
- **Weights & Biases**: https://docs.wandb.ai/

### Datasets
- **MNIST**: `torchvision.datasets.MNIST`
- **CIFAR-10**: `torchvision.datasets.CIFAR10`

### Tools
- **W&B**: Experiment tracking
- **TensorBoard**: Alternative tracking
- **GitHub**: Code hosting
- **Medium/Substack**: Blog platforms

---

## Success Metrics

Track these in `experiments/metrics.md`:

| Metric | Target | Actual |
|--------|--------|--------|
| ResNet-18 CIFAR-10 Accuracy | >85% | ___ |
| ViT-Tiny CIFAR-10 Accuracy | >75% | ___ |
| SimCLR Linear Eval Accuracy | >80% | ___ |
| GitHub Commits | 30+ | ___ |
| Papers Read (Deep) | 4 | ___ |
| Papers Skimmed | 4 | ___ |
| Hours Invested | 25-30 | ___ |
| Blog Words Written | 1500+ | ___ |

---

## Daily Log Template

Copy this to `notes/daily-log.md`:

```markdown
## Day X: [Date]

### What I Did
- 
-
-

### What I Learned
-
-

### Challenges
-
-

### Tomorrow's Focus
-
-

### Time Spent: ___ hours
```

---

## Need Help?

**Stuck on implementation?**
- Check PyTorch documentation
- Search GitHub for similar implementations
- Ask on Stack Overflow
- Use Claude/ChatGPT for debugging

**Concepts unclear?**
- Re-read the paper
- Watch lecture videos
- Find alternative explanations (blog posts, YouTube)

**Falling behind?**
- Focus on ResNet + ViT (core models)
- Skip SimCLR if needed (do in Week 2)
- Aim for understanding > completion

---

**Remember:** Week 1 is about building momentum and establishing good habits. Progress > Perfection.

**You've got this!** 🚀

---

*Last updated: [Date]*  
*Status: [ ] Not Started | [x] In Progress | [ ] Complete*
