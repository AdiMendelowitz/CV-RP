# 12-Week Computer Vision, Time Series & Adversarial ML Study Plan
## Optimized for EU Visa Sponsorship

**Background Assumed:** 6 years Data Science experience, Math degree, Python proficiency  
**Goal:** Become sponsorship-worthy for Western European ML roles in CV/Time Series/Adversarial ML  
**Focus Areas:** Computer Vision, Time Series Forecasting, Adversarial ML, Image Processing  
**Avoid:** NLP/LLMs (except adversarial attacks on multimodal models)

---

## Table of Contents
1. [Overview & Strategy](#overview--strategy)
2. [Setup & Tools](#setup--tools)
3. [Weekly Breakdown](#weekly-breakdown)
4. [Resource Library](#resource-library)
5. [Portfolio Requirements](#portfolio-requirements)
6. [Application Strategy](#application-strategy)

---

## Overview & Strategy

### Why This Plan Works for Sponsorship

**Your Competitive Advantages:**
- Math degree + 6 years production experience
- Israeli tech/cybersecurity background (highly valued in Europe)
- Specialization in CV + Adversarial ML (less saturated than LLMs)
- Safety-critical applications (mandatory for EU AI Act compliance)

**Target Roles:**
- Computer Vision Engineer (Autonomous Vehicles, Medical Imaging)
- ML Security Engineer (Adversarial ML, Red-teaming)
- Time Series ML Engineer (Energy, Industrial IoT)
- Research Engineer (DeepMind, Meta AI, etc.)

**Target Countries (Priority Order):**
1. **Netherlands** - Easiest visa (ASML, Philips, Shell)
2. **Germany** - Automotive/Industrial (Mercedes, BMW, Siemens)
3. **UK** - Strong AI scene (DeepMind, Wayve, Stability AI)
4. **Switzerland** - Highest salaries (Google, Meta, ABB)

### Success Metrics by Week 12

**Portfolio:**
- ✅ 5 production-quality GitHub repos (1000+ lines each)
- ✅ 1 arXiv preprint or workshop paper submission
- ✅ 8-10 technical blog posts (aim for 10k+ total views)
- ✅ Active presence in ML community (Twitter/LinkedIn)

**Technical Depth:**
- ✅ 40+ papers read (150+ skimmed)
- ✅ 3+ models implemented from scratch
- ✅ 2+ paper reproductions with novel insights
- ✅ 1 domain-specific application (medical/industrial/autonomous)

**Network:**
- ✅ 10+ connections at target companies
- ✅ 1-2 informational interviews completed
- ✅ Applications sent to 20-30 companies

---

## Setup & Tools

### Development Environment

**Python Stack (Week 1 Setup):**
```bash
# Create dedicated environment
conda create -n cv_research python=3.13
conda activate cv_research

# Core ML/DL
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy pandas scikit-learn matplotlib seaborn

# Computer Vision
pip install opencv-python albumentations timm segmentation-models-pytorch
pip install ultralytics  # YOLOv8

# Time Series
pip install statsmodels prophet neuralprophet tslearn

# Adversarial ML
pip install foolbox adversarial-robustness-toolbox cleverhans

# Experiment Tracking & Deployment
pip install wandb tensorboard mlflow
pip install onnx onnxruntime tensorrt  # Model optimization

# Utilities
pip install jupyter jupyterlab ipywidgets tqdm pillow
pip install gradio streamlit  # For demos

# Save environment
pip freeze > requirements.txt
```

**GPU Setup:**
- Verify CUDA installation: `nvidia-smi`
- Test PyTorch GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- Recommended: Local GPU (RTX 3060+) or cloud (Lambda Labs, Vast.ai)

### Essential Tools

**Experiment Tracking:**
- **Weights & Biases** (primary): https://wandb.ai
  - Sign up for free academic account
  - Use for all experiments starting Week 1
- **TensorBoard** (backup): Built into PyTorch

**Paper Management:**
- **Zotero**: https://www.zotero.org/download/
- Install Better BibTeX plugin for LaTeX
- Create collections: "CV Foundations", "Adversarial ML", "Time Series", "To Read"

**Writing:**
- **Overleaf**: https://www.overleaf.com (LaTeX for papers)
- **Notion** or **Obsidian**: For daily notes and paper summaries
- **Grammarly**: For blog posts

**Code:**
- **VS Code** with extensions:
  - Python, Pylance
  - Jupyter
  - GitLens
  - GitHub Copilot (optional, paid)

**Visualization:**
- **Matplotlib** + **Seaborn** for plots
- **TensorBoard** for training curves
- **Netron** for model architecture visualization

**Datasets:**
- **Hugging Face Datasets**: https://huggingface.co/datasets
- **Kaggle**: https://www.kaggle.com/datasets
- **Papers with Code Datasets**: https://paperswithcode.com/datasets

### GitHub Setup (Week 1, Day 1)

**Repository Structure:**
```
your-github-username/
├── cv-from-scratch/          # Week 1-2: CNN, ViT implementations
├── time-series-forecasting/  # Week 5-6: Transformer for TS
├── adversarial-ml-toolkit/   # Week 7-8: Attack/defense library
├── domain-application/       # Week 10: Medical/Industrial/Autonomous
└── research-project/         # Week 11-12: Novel contribution
```

**Each repo must have:**
- Comprehensive README with installation, usage, results
- `requirements.txt`
- Clear code structure (`src/`, `notebooks/`, `data/`, `models/`)
- Example usage notebooks
- Documentation (docstrings)
- MIT or Apache 2.0 license

---

## Weekly Breakdown

---

### **WEEK 1: Computer Vision Foundations**

**Goal:** Build CNNs from scratch, understand modern architectures

#### **Monday: Classical Computer Vision + CNN Basics**

**Morning (3-4 hours):**
1. **Review Classical CV** (you may know this, go fast):
   - OpenCV tutorial: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
   - Focus: Filters, edge detection, morphology, feature extraction
   - Implement: Sobel edge detection, Harris corner detection from scratch

2. **CNN Theory Deep Dive:**
   - Read: Stanford CS231n Notes on CNNs
     - http://cs231n.github.io/convolutional-networks/
     - http://cs231n.github.io/understanding-cnn/
   - Understand: Convolution math, receptive fields, pooling, normalization

**Afternoon (3-4 hours):**
3. **Implementation: CNN from Scratch in NumPy**
   - Build: Conv2D layer, MaxPool2D, fully connected
   - No PyTorch/TensorFlow - pure NumPy
   - Train on MNIST to verify correctness
   - Reference: https://github.com/wiseodd/hipsternet

**Evening (1-2 hours):**
4. **Paper Reading:**
   - **AlexNet** (2012): "ImageNet Classification with Deep CNNs" - Krizhevsky et al.
   - **VGGNet** (2014): "Very Deep Convolutional Networks" - Simonyan & Zisserman
   - Links: https://paperswithcode.com/method/alexnet

**Output:**
- NumPy CNN implementation in `cv-from-scratch/numpy_cnn/`
- Notes on convolution mathematics
- Training curves for MNIST

---

#### **Tuesday: Modern CNN Architectures**

**Morning (3-4 hours):**
1. **Study ResNet Architecture:**
   - **Paper**: "Deep Residual Learning for Image Recognition" (He et al., 2015)
   - Link: https://arxiv.org/abs/1512.03385
   - Understand: Skip connections, identity mappings, degradation problem
   - Watch: Yannic Kilcher explanation: https://www.youtube.com/watch?v=GWt6Fu05voI

2. **Study EfficientNet:**
   - **Paper**: "EfficientNet: Rethinking Model Scaling for CNNs" (Tan & Le, 2019)
   - Link: https://arxiv.org/abs/1905.11946
   - Understand: Compound scaling, network architecture search

**Afternoon (3-4 hours):**
3. **Implement ResNet in PyTorch:**
   - Build ResNet-18 from scratch (no torchvision.models)
   - Implement: BasicBlock, Bottleneck, ResNet class
   - Reference: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
   - Train on CIFAR-10

**Evening (2 hours):**
4. **Compare Architectures:**
   - Load pretrained: ResNet, EfficientNet, ConvNeXt (timm library)
   - Benchmark: Accuracy, inference time, parameters, FLOPs
   - Use: `torchinfo` for model summaries

**Output:**
- PyTorch ResNet implementation
- Comparison table of architectures
- Blog post outline: "CNN Architecture Evolution: AlexNet to ConvNeXt"

---

#### **Wednesday: Vision Transformers (ViT)**

**Morning (3-4 hours):**
1. **Study Vision Transformer Theory:**
   - **Paper**: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020)
   - Link: https://arxiv.org/abs/2010.11929
   - **Paper**: "Training data-efficient image transformers & distillation" (DeiT) - Touvron et al.
   - Link: https://arxiv.org/abs/2012.12877
   - Understand: Patch embeddings, positional encodings, attention for vision

2. **Watch Tutorials:**
   - Yannic Kilcher ViT: https://www.youtube.com/watch?v=TrdevFK_am4
   - Official implementation walkthrough

**Afternoon (4 hours):**
3. **Implement ViT from Scratch:**
   - Build: PatchEmbedding, TransformerEncoder, ViT class
   - No libraries except torch.nn basic layers
   - Reference: https://github.com/lucidrains/vit-pytorch (for structure, don't copy)
   - Train on CIFAR-10 or CIFAR-100

**Evening (1-2 hours):**
4. **Fine-tuning Experiment:**
   - Load pretrained ViT from `timm`
   - Fine-tune on custom dataset (e.g., Food-101, Oxford Pets)
   - Compare: Training from scratch vs. fine-tuning

**Output:**
- ViT implementation from scratch
- Fine-tuning notebook with results
- Notes on when ViT > CNN and vice versa

---

#### **Thursday: Self-Supervised Learning for Vision**

**Morning (3 hours):**
1. **Study Self-Supervised Methods:**
   - **Paper**: "Momentum Contrast for Unsupervised Visual Representation Learning" (MoCo) - He et al.
   - Link: https://arxiv.org/abs/1911.05722
   - **Paper**: "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR) - Chen et al.
   - Link: https://arxiv.org/abs/2002.05709
   - **Paper**: "Emerging Properties in Self-Supervised Vision Transformers" (DINO) - Caron et al.
   - Link: https://arxiv.org/abs/2104.14294

2. **Study Masked Autoencoders:**
   - **Paper**: "Masked Autoencoders Are Scalable Vision Learners" (MAE) - He et al.
   - Link: https://arxiv.org/abs/2111.06377

**Afternoon (3-4 hours):**
3. **Implement SimCLR (Simplified):**
   - Build contrastive learning pipeline
   - Data augmentations: random crop, color jitter, blur
   - NT-Xent loss implementation
   - Reference: https://github.com/sthalles/SimCLR

**Evening (2 hours):**
4. **Experiment:**
   - Pretrain on unlabeled CIFAR-10
   - Fine-tune on small labeled subset (1%, 10%)
   - Compare to supervised baseline

**Output:**
- SimCLR implementation
- Results: Self-supervised vs. supervised with limited labels
- Blog post: "Self-Supervised Learning for Computer Vision"

---

#### **Friday: Data Augmentation & Regularization**

**Morning (3 hours):**
1. **Study Augmentation Techniques:**
   - Classical: Flip, crop, rotation, color jitter
   - Advanced: Cutout, Mixup, CutMix, RandAugment, AutoAugment
   - **Paper**: "mixup: Beyond Empirical Risk Minimization" - Zhang et al.
   - Link: https://arxiv.org/abs/1710.09412
   - **Paper**: "CutMix: Regularization Strategy to Train Strong Classifiers" - Yun et al.
   - Link: https://arxiv.org/abs/1905.04899

2. **Albumentations Library:**
   - Tutorial: https://albumentations.ai/docs/
   - Build custom augmentation pipeline

**Afternoon (3 hours):**
3. **Implementation & Experiments:**
   - Implement: Mixup, CutMix from scratch
   - Build augmentation pipeline with Albumentations
   - Run ablations: baseline, Mixup, CutMix, combined

**Evening (2 hours):**
4. **Regularization Study:**
   - Implement: Dropout, DropBlock, Stochastic Depth
   - Compare: Different regularization strategies
   - **Paper**: "Deep Networks with Stochastic Depth" - Huang et al.
   - Link: https://arxiv.org/abs/1603.09382

**Output:**
- Augmentation pipeline code
- Ablation study results (table + plots)
- Best practices document

---

#### **Saturday: Week 1 Wrap-up & Blog Post**

**Morning (3 hours):**
1. **Clean Code:**
   - Refactor all Week 1 code
   - Add docstrings and README files
   - Push to GitHub: `cv-from-scratch` repository

2. **Organize Notes:**
   - Summarize all papers read (7-8 papers)
   - Create comparison table of architectures

**Afternoon (3-4 hours):**
3. **Write Blog Post:**
   - Title: "Building Modern Computer Vision Models from Scratch: A Deep Dive"
   - Sections:
     - CNNs vs. Vision Transformers
     - When to use each architecture
     - Self-supervised learning practical guide
     - Augmentation strategies
   - Include: Code snippets, visualizations, results
   - Publish on: Medium, Dev.to, or personal blog

**Evening (1-2 hours):**
4. **Week 1 Review:**
   - What worked well?
   - What needs more time?
   - Adjust Week 2 plan if needed

**Output:**
- Clean GitHub repo with all Week 1 code
- Published blog post
- Paper summary notes in Zotero

---

### **WEEK 2: Advanced Computer Vision & Object Detection**

**Goal:** Master object detection, segmentation, and instance-level tasks

#### **Monday: Object Detection Fundamentals**

**Morning (3-4 hours):**
1. **Study Two-Stage Detectors:**
   - **Paper**: "Rich feature hierarchies for accurate object detection" (R-CNN) - Girshick et al.
   - Link: https://arxiv.org/abs/1311.2524
   - **Paper**: "Fast R-CNN" - Girshick
   - Link: https://arxiv.org/abs/1504.08083
   - **Paper**: "Faster R-CNN: Towards Real-Time Object Detection" - Ren et al.
   - Link: https://arxiv.org/abs/1506.01497
   - Understand: Region proposals, RoI pooling, anchor boxes

2. **Study One-Stage Detectors:**
   - **Paper**: "You Only Look Once: Unified, Real-Time Object Detection" (YOLO) - Redmon et al.
   - Link: https://arxiv.org/abs/1506.02640
   - **Paper**: "SSD: Single Shot MultiBox Detector" - Liu et al.
   - Link: https://arxiv.org/abs/1512.02325

**Afternoon (3-4 hours):**
3. **Implement Simple Object Detector:**
   - Build basic sliding window detector
   - Implement non-maximum suppression (NMS)
   - Understand: IoU, mAP metrics

**Evening (2 hours):**
4. **Study Modern Architectures:**
   - **Paper**: "YOLOv4: Optimal Speed and Accuracy" - Bochkovskiy et al.
   - Link: https://arxiv.org/abs/2004.10934
   - Read YOLOv8 documentation: https://docs.ultralytics.com/

**Output:**
- Notes on detection architectures evolution
- NMS implementation from scratch
- mAP calculation code

---

#### **Tuesday: YOLOv8 Implementation & Fine-tuning**

**Morning (2-3 hours):**
1. **Setup YOLOv8:**
   ```bash
   pip install ultralytics
   ```
   - Tutorial: https://docs.ultralytics.com/quickstart/
   - Understand: Architecture, loss function, training process

2. **Prepare Custom Dataset:**
   - Choose domain: Medical (e.g., blood cell detection), Industrial (e.g., PCB defects), or Autonomous (e.g., traffic signs)
   - Format data in YOLO format
   - Data sources:
     - Roboflow Universe: https://universe.roboflow.com/
     - Kaggle datasets
     - Public medical datasets: https://www.kaggle.com/datasets/paultimothymooney/blood-cells

**Afternoon (4 hours):**
3. **Train YOLOv8:**
   ```python
   from ultralytics import YOLO
   
   # Load pretrained model
   model = YOLO('yolov8n.pt')
   
   # Train on custom data
   results = model.train(data='custom.yaml', epochs=100, imgsz=640)
   ```
   - Train with data augmentation
   - Monitor training with W&B
   - Run validation

**Evening (2 hours):**
4. **Evaluation & Analysis:**
   - Calculate: mAP@0.5, mAP@0.5:0.95
   - Error analysis: Which classes fail? Why?
   - Visualize: Confusion matrix, failure cases

**Output:**
- Fine-tuned YOLOv8 model
- Training logs and metrics
- Error analysis document

---

#### **Wednesday: Transformer-Based Detection (DETR)**

**Morning (3-4 hours):**
1. **Study DETR:**
   - **Paper**: "End-to-End Object Detection with Transformers" - Carion et al.
   - Link: https://arxiv.org/abs/2005.12872
   - Understand: Set prediction, bipartite matching, no anchors/NMS
   - **Paper**: "Deformable DETR: Deformable Transformers for End-to-End Object Detection" - Zhu et al.
   - Link: https://arxiv.org/abs/2010.04159

2. **Watch Tutorial:**
   - Yannic Kilcher DETR: https://www.youtube.com/watch?v=T35ba_VXkMY

**Afternoon (4 hours):**
3. **Fine-tune DETR:**
   - Use Hugging Face implementation:
     ```python
     from transformers import DetrImageProcessor, DetrForObjectDetection
     ```
   - Fine-tune on your custom dataset
   - Compare to YOLOv8 results

**Evening (1-2 hours):**
4. **Analysis:**
   - DETR vs. YOLO: Speed, accuracy, training time
   - When to use each approach?

**Output:**
- Fine-tuned DETR model
- Comparison table: DETR vs. YOLO
- Notes on transformer-based detection

---

#### **Thursday: Segmentation (Semantic & Instance)**

**Morning (3 hours):**
1. **Study Segmentation Architectures:**
   - **Paper**: "Fully Convolutional Networks for Semantic Segmentation" (FCN) - Long et al.
   - Link: https://arxiv.org/abs/1411.4038
   - **Paper**: "U-Net: Convolutional Networks for Biomedical Image Segmentation" - Ronneberger et al.
   - Link: https://arxiv.org/abs/1505.04597
   - **Paper**: "Mask R-CNN" - He et al.
   - Link: https://arxiv.org/abs/1703.06870
   - **Paper**: "Segment Anything" (SAM) - Kirillov et al.
   - Link: https://arxiv.org/abs/2304.02643

**Afternoon (4 hours):**
2. **Implement U-Net:**
   - Build U-Net from scratch in PyTorch
   - Train on medical segmentation dataset:
     - Carvana Image Masking: https://www.kaggle.com/c/carvana-image-masking-challenge
     - Or medical: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

3. **Use Segmentation Models PyTorch:**
   ```bash
   pip install segmentation-models-pytorch
   ```
   - Try different encoders: ResNet, EfficientNet, etc.
   - Compare architectures

**Evening (2 hours):**
4. **Evaluation:**
   - Metrics: IoU, Dice coefficient, pixel accuracy
   - Visualize: Predictions, errors, attention maps

**Output:**
- U-Net implementation
- Trained segmentation model
- Metrics and visualizations

---

#### **Friday: Medical/Industrial Imaging Application**

**Morning (3 hours):**
1. **Choose Domain-Specific Task:**
   
   **Option A - Medical Imaging:**
   - Dataset: ChestX-ray14, ISIC (skin lesions), or retinal images
   - Task: Disease classification or lesion segmentation
   - Study domain-specific augmentations
   
   **Option B - Industrial Inspection:**
   - Dataset: MVTec AD (anomaly detection)
   - Link: https://www.mvtec.com/company/research/datasets/mvtec-ad
   - Task: Defect detection and segmentation
   
   **Option C - Autonomous Systems:**
   - Dataset: KITTI, Cityscapes
   - Task: Road segmentation, obstacle detection

2. **Study Domain Challenges:**
   - Class imbalance handling
   - Limited labeled data
   - Domain shift/distribution shift

**Afternoon (4 hours):**
3. **Build Complete Pipeline:**
   - Data preprocessing for chosen domain
   - Model architecture selection
   - Training with domain-specific techniques:
     - Focal loss for imbalance
     - Uncertainty estimation