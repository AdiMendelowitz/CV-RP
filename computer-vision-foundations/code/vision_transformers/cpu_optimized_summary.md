# ViT CPU Optimization Summary

All ViT-related files have been optimized for CPU-only training for a local machine.

## 🔧 Changes Made

### 1. `train_vit_cifar10.py`

**Device:**
- ✅ Forced CPU usage: `device = torch.device('cpu')`
- ✅ Removed CUDA-specific code
- ✅ Added CPU training time warnings

**Performance:**
- ✅ Reduced epochs: 200 → 100 (faster convergence)
- ✅ Reduced batch size: 128 → 64 (CPU-efficient)
- ✅ Removed `pin_memory=True` (GPU optimization)
- ✅ Set `num_workers=0` (avoid multiprocessing overhead on CPU)

### 2. `evaluate_vit.py`

**Device:**
- ✅ Forced CPU: `device = torch.device('cpu')`
- ✅ Removed CUDA checks
- ✅ Set `num_workers=0`

### 3. `config_vit_cifar10.py`

**Updated defaults:**
- ✅ `num_epochs: 100` (was 200)
- ✅ `batch_size: 64` (was 128)
- ✅ `num_workers: 0` (was 4)
- ✅ `pin_memory: False` (was True)
- ✅ Added CPU-specific training notes

### 4. `README_VIT_TRAINING.md`

**Documentation:**
- ✅ CPU training time estimates
- ✅ Expected accuracy with fewer epochs
- ✅ Quick experimentation tips
- ✅ Updated troubleshooting for CPU

---

## 📊 Expected Results (CPU Training)

| Configuration | Epochs | Time | Expected Accuracy |
|--------------|--------|------|------------------|
| Quick test | 50 | ~6-12 hours | 70-75% |
| Full training | 100 | ~12-24 hours | 75-80% |
| GPU equivalent | 200 | ~2-3 hours | 75-85% |

---

## 🚀 Quick Start

```bash
# Navigate to directory
cd computer-vision-foundations/code/vision_transformers

# Start training (default: 100 epochs, batch_size=64)
python train_vit_cifar10.py
```

**For faster experimentation (50 epochs):**

Edit `train_vit_cifar10.py`:
```python
# Change this line in the main block:
num_epochs=50,   # Instead of 100
```

---

## 💡 Recommended Workflow

### Phase 1: Quick Test (50 epochs)
```python
# In train_vit_cifar10.py, modify:
history, model = train_vit_tiny(
    num_epochs=50,      # Quick test
    batch_size=64,
    ...
)
```

**Expected:**
- Time: ~6-12 hours
- Accuracy: ~70-75%
- Purpose: Verify everything works

### Phase 2: Full Training (100 epochs)
```python
# Use default settings:
history, model = train_vit_tiny(
    num_epochs=100,     # Full run
    batch_size=64,
    ...
)
```

**Expected:**
- Time: ~12-24 hours
- Accuracy: ~75-80%
- Purpose: Final results for comparison

---

## ⚙️ Further Optimizations (If Needed)

If training is too slow or your laptop struggles:

### Option 1: Even Smaller Batch Size
```python
batch_size=32   # Halves memory usage, slightly faster per epoch
```

### Option 2: Reduce Model Size
```python
# In create_vit_tiny_cifar10():
model = VisionTransformer(
    embed_dim=96,    # Half the size (was 192)
    depth=6,         # Fewer layers (was 12)
    num_heads=2,     # Fewer heads (was 3)
    ...
)
# Trains 4× faster, accuracy ~65-70%
```

### Option 3: Minimal Augmentation
```python
# In get_cifar10_dataloaders(), comment out AutoAugment:
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.AutoAugment(...),  # Disabled for speed
    transforms.ToTensor(),
    transforms.Normalize(...)
])
# ~10-15% faster, slightly lower accuracy
```

---

## 📈 Monitoring Progress

The script will print updates every 100 batches:

```
Epoch [1/100]
----------------------------------------------------------------------
  Batch [100/391] | Loss: 2.1234 | Acc: 25.34% | LR: 0.000100
  Batch [200/391] | Loss: 1.9876 | Acc: 32.15% | LR: 0.000200
  ...

  Epoch Summary:
    Train Loss: 1.8543 | Train Acc: 35.67%
    Test Loss:  1.7234 | Test Acc:  38.92%
    Time: 254.3s (~4.2 minutes per epoch)
```

**Estimated total time:**
- 4.2 min/epoch × 100 epochs ≈ 420 minutes ≈ **7 hours**
- Can vary based on your laptop specs

---

## 🎯 The Key Insight

**ViT will underperform ResNet on CIFAR-10 - this is expected and valuable!**

| Model | Accuracy | Why? |
|-------|----------|------|
| ResNet-18 | ~94-95% | Strong spatial inductive bias from convolutions |
| ViT-Tiny (CPU) | ~75-80% | Needs large datasets (10M+ images) to excel |

**This comparison demonstrates:**
1. ✅ Architecture selection matters
2. ✅ Data scale affects transformer performance
3. ✅ Inductive bias helps on small datasets
4. ✅ Understanding tradeoffs > chasing high accuracy

---

## ✅ Files Ready to Use

All files are now CPU-optimized and ready:
- ✅ `vit.py` - Model implementation (unchanged)
- ✅ `train_vit_cifar10.py` - CPU-optimized training
- ✅ `evaluate_vit.py` - CPU-optimized evaluation
- ✅ `config_vit_cifar10.py` - CPU-friendly defaults
- ✅ `README_VIT_TRAINING.md` - Updated documentation

---

**Ready to train!** Just run:
```bash
python train_vit_cifar10.py
```

And let it run overnight (or over a weekend for 100 epochs). The script handles everything automatically! 🚀