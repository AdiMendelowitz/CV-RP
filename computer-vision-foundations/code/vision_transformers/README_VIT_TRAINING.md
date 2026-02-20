# Training ViT-Tiny on CIFAR-10

Complete setup for training Vision Transformer on CIFAR-10 dataset.

## 📁 Files

- **`vit.py`** - Vision Transformer implementation
- **`train_vit_cifar10.py`** - Training script
- **`config_vit_cifar10.py`** - Configuration and hyperparameters
- **`evaluate_vit.py`** - Evaluation and visualization
- **`README_VIT_TRAINING.md`** - This file

## 🚀 Quick Start

### 1. Train the Model (CPU-optimized)

```bash
python train_vit_cifar10.py
```

This will:
- Download CIFAR-10 automatically (if not present)
- Train ViT-Tiny for 100 epochs (CPU-optimized)
- Save checkpoints to `./checkpoints/vit_tiny_cifar10/`
- Log training progress

**Expected training time:**
- CPU: ~12-24 hours for 100 epochs
- For quick test: Edit script to use 50 epochs (~6-12 hours)

**💡 Quick experimentation tip:**
```python
# In train_vit_cifar10.py, change:
num_epochs=50,   # Faster test (~70-75% accuracy)
batch_size=32,   # If your CPU is struggling
```

### 2. Evaluate the Model

After training completes:

```bash
python evaluate_vit.py
```

This will:
- Load the best checkpoint
- Compute per-class accuracy
- Plot training curves
- Show confusion matrix
- Visualize predictions

## ⚙️ Configuration

### Model Configuration (ViT-Tiny for CIFAR-10)

```python
img_size: 32          # CIFAR-10 image size
patch_size: 4         # 4×4 patches → 64 patches
embed_dim: 192        # Embedding dimension
depth: 12             # Transformer blocks
num_heads: 3          # Attention heads
dropout: 0.1          # Regularization
```

### Training Configuration (CPU-optimized)

```python
num_epochs: 100       # Reduced for CPU (50 for quick test)
batch_size: 64        # CPU-efficient size
learning_rate: 0.001  # Peak LR after warmup
weight_decay: 0.05    # AdamW regularization
warmup_epochs: 10     # Linear warmup
num_workers: 0        # CPU-optimized
```

Modify these in `train_vit_cifar10.py` or pass as arguments.

## 📊 Expected Results

### Performance Benchmarks (CPU Training)

| Setup | Expected Accuracy | Training Time |
|-------|------------------|---------------|
| ViT-Tiny (50 epochs) | **70-75%** | ~6-12 hours |
| ViT-Tiny (100 epochs) | **75-80%** | ~12-24 hours |
| ResNet-18 from scratch | **94-95%** | ~2-3 hours |
| ViT-Base (ImageNet pretrained) | **98-99%** | Transfer learning |

**Why the gap?**

1. **Data scale:** CIFAR-10 has only 50K training images
2. **Inductive bias:** CNNs have built-in spatial assumptions
3. **Model capacity:** ViT needs large datasets to learn spatial relationships
4. **CPU vs GPU:** Fewer epochs due to CPU limitations

### Training Curves to Expect (100 epochs)

**Loss:**
- Initial: ~2.3 (random guessing)
- After 50 epochs: ~1.0-1.2
- After 100 epochs: ~0.7-0.9

**Accuracy:**
- After 10 epochs: ~40-50%
- After 50 epochs: ~65-70%
- After 100 epochs: **75-80%**

## 🔧 Troubleshooting

### Slow Training

**This is expected on CPU!** Transformers are computationally intensive.

**Speed up options:**
1. Reduce epochs:
```python
num_epochs=50   # ~70-75% accuracy in half the time
```

2. Reduce batch size (trades speed for stability):
```python
batch_size=32   # Smaller batches = faster per epoch
```

3. Use fewer data augmentation:
```python
# In get_cifar10_dataloaders(), comment out AutoAugment
# Slightly faster, but lower accuracy
```

### Memory Issues (Rare on CPU)

If you see memory errors:

```python
batch_size=32   # Or even 16
```

### Training Appears Stuck

If accuracy doesn't improve after many epochs:

1. **Check learning rate:** Warmup might be too long
```python
warmup_epochs=5   # Instead of 10
```

2. **Verify data loading:**
```python
# Print a batch to ensure data is correct
for images, labels in train_loader:
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels[:10]}")
    break
```

## 📈 Monitoring Training

### During Training

Watch the console output:

```
Epoch [1/200]
----------------------------------------------------------------------
  Batch [100/391] | Loss: 2.1234 | Acc: 25.34% | LR: 0.000100
  Batch [200/391] | Loss: 1.9876 | Acc: 32.15% | LR: 0.000200

  Epoch Summary:
    Train Loss: 1.8543 | Train Acc: 35.67%
    Test Loss:  1.7234 | Test Acc:  38.92%
    Learning Rate: 0.000500
    Time: 87.3s
    ✓ New best accuracy! Saved checkpoint.
```

### After Training

View saved checkpoints:

```
checkpoints/vit_tiny_cifar10/
├── best_model.pth              # Best test accuracy
├── checkpoint_epoch_50.pth     # Intermediate checkpoint
├── checkpoint_epoch_100.pth
├── checkpoint_epoch_150.pth
├── final_model.pth             # Final epoch
└── training_history.json       # Loss/accuracy history
```

## 🔬 Experiments to Try

### 1. Compare Patch Sizes

```python
# In train_vit_cifar10.py, modify create_vit_tiny_cifar10():
patch_size=2   # 256 patches (slower, more detail)
patch_size=4   # 64 patches (baseline)
patch_size=8   # 16 patches (faster, less detail)
```

### 2. Compare with ResNet

Train ResNet-18 for comparison:

```bash
python train_resnet_cifar10.py  # You already have this!
```

Expected: ResNet should achieve ~94-95% (much better than ViT).

### 3. Ablation Studies

**Remove augmentation:**
```python
# In get_cifar10_dataloaders(), comment out AutoAugment
# Expected: Accuracy drops ~5-10%
```

**Remove warmup:**
```python
warmup_epochs=0
# Expected: Training less stable, slightly lower accuracy
```

**Smaller model:**
```python
embed_dim=96   # Half the dimensions
depth=6        # Half the layers
# Expected: Faster training, lower accuracy (~70-75%)
```

## 📚 Understanding the Results

### Why ViT Underperforms Here

**ViT was designed for large-scale datasets:**
- Original paper trained on JFT-300M (300 million images)
- ImageNet-21K (14 million images) also works well
- CIFAR-10 (50,000 images) is too small

**What ViT needs to learn:**
- Spatial relationships (CNNs get this "for free" via convolutions)
- Local patterns → global structure (must learn from data)
- Translation invariance (CNNs have this built-in)

**With 50K images, there's not enough data for ViT to learn these effectively.**

### When ViT Shines

1. **Large datasets:** ImageNet (1.3M) → ViT matches CNNs
2. **Huge datasets:** JFT-300M (300M) → ViT beats CNNs
3. **Transfer learning:** Pretrain on large data, finetune on CIFAR-10 → ViT reaches 98-99%

### The Lesson

This experiment demonstrates a fundamental principle in deep learning:

**Inductive bias vs data scale is a tradeoff:**
- Strong inductive bias (CNNs) → good on small data
- Weak inductive bias (ViT) → needs large data, but more flexible

## 📊 Comparing to Your ResNet Results

You should have already trained ResNet-18 on CIFAR-10. Compare:

| Metric | ResNet-18 | ViT-Tiny (100 epochs) | Winner |
|--------|-----------|----------------------|--------|
| **Accuracy** | ~94-95% | ~75-80% | ResNet ✓ |
| **Parameters** | ~11M | ~5.7M | ViT ✓ |
| **Training time (CPU)** | ~2-3 hours | ~12-24 hours | ResNet ✓ |
| **Data efficiency** | High | Low | ResNet ✓ |

**Conclusion:** For CIFAR-10 on CPU, CNNs are clearly superior. But this demonstrates *why* data scale and architecture choice matter for Transformers!

## 📝 Citation

If using this for research:

```bibtex
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

---

**Good luck with training!** 🚀


---
## 👤 Author

**Adi Mendelowitz**  
Machine Learning Engineer  
Specialization: Computer Vision & Image Processing