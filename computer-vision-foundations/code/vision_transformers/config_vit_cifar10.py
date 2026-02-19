"""
Configuration file for ViT training on CIFAR-10

Modify these parameters to experiment with different settings.
"""

# Model Configuration
MODEL_CONFIG = {
    'img_size': 32,  # CIFAR-10 image size
    'patch_size': 4,  # 4×4 patches → 64 patches for 32×32 image
    'in_channels': 3,  # RGB
    'num_classes': 10,  # CIFAR-10 classes
    'embed_dim': 192,  # ViT-Tiny: 192
    'depth': 12,  # Number of transformer blocks
    'num_heads': 3,  # Attention heads (embed_dim must be divisible by num_heads)
    'mlp_ratio': 4.0,  # MLP expansion ratio
    'dropout': 0.1,  # Dropout rate
}

# Training Configuration (CPU-optimized)
TRAINING_CONFIG = {
    'num_epochs': 50,  # Reduced for CPU (was 200)
    'batch_size': 32,  # Reduced for CPU efficiency (was 128)
    'learning_rate': 0.001,  # Peak LR after warmup
    'weight_decay': 0.05,  # AdamW weight decay
    'warmup_epochs': 10,  # Linear warmup epochs
    'label_smoothing': 0.1,  # Label smoothing for regularization
    'grad_clip': 1.0,  # Gradient clipping norm
}

# Data Configuration (CPU-optimized)
DATA_CONFIG = {
    'data_dir': './data',
    'num_workers': 0,  # CPU-optimized (was 4)
    'pin_memory': False,  # GPU optimization - disabled for CPU
}

# Paths
PATHS = {
    'save_dir': './checkpoints/vit_tiny_cifar10',
    'log_dir': './logs/vit_tiny_cifar10',
}

# Expected Performance Benchmarks
# (Based on "An Image is Worth 16x16 Words" paper and follow-up work)
EXPECTED_PERFORMANCE = {
    'vit_tiny_cifar10_scratch': {
        'test_acc': '~75-80%',
        'note': 'Training from scratch on CIFAR-10 (50K images) - ViT underfits without large data'
    },
    'vit_tiny_cifar10_with_augmentation': {
        'test_acc': '~80-85%',
        'note': 'Heavy augmentation helps, but still below CNN performance'
    },
    'resnet18_cifar10': {
        'test_acc': '~94-95%',
        'note': 'CNNs perform better on small datasets due to inductive bias'
    },
    'vit_base_imagenet_then_cifar10': {
        'test_acc': '~98-99%',
        'note': 'ViT pretrained on ImageNet, then finetuned - matches/beats CNNs'
    }
}

# Key Insights for Training ViT
TRAINING_NOTES = """
ViT Training on CIFAR-10 - What to Expect (CPU Version):

⚠️  CPU TRAINING CONSIDERATIONS:
   - Training time: ~12-24 hours for 100 epochs (vs 2-3 hours on GPU)
   - For quick experimentation: Reduce to 50 epochs (~6-12 hours)
   - Batch size reduced to 64 for CPU efficiency
   - Expected accuracy slightly lower than GPU version due to fewer epochs

1. CONVERGENCE:
   - Slower than CNNs initially (first 50 epochs)
   - Needs full 100 epochs to converge on CPU
   - Learning rate warmup is critical

2. PERFORMANCE (100 epochs on CPU):
   - From scratch: 70-80% (vs ResNet's 94-95%)
   - With 200 epochs (GPU): 75-85%
   - ViT needs large datasets (>100K images) to compete with CNNs
   - CIFAR-10 (50K images) is too small for ViT to excel

3. WHY VIT UNDERPERFORMS ON SMALL DATA:
   - No spatial inductive bias (CNNs have convolution locality)
   - Higher model capacity → needs more data
   - Must learn spatial relationships from scratch

4. WHAT HELPS:
   - Heavy data augmentation (AutoAugment, MixUp, CutMix)
   - Label smoothing
   - Strong regularization (dropout, weight decay)
   - Longer training (100+ epochs)

5. WHEN VIT SHINES:
   - Large datasets (ImageNet: 1.3M images → ViT matches CNNs)
   - Huge datasets (JFT-300M: 300M images → ViT beats CNNs)
   - Transfer learning (pretrain on large data, finetune on CIFAR-10)

💡 TIP: Start with 50 epochs for faster experimentation (~70-75% accuracy)
      Then run full 100 epochs if needed (~75-80% accuracy)

This experiment demonstrates why data scale matters for Transformers!
"""

if __name__ == '__main__':
    print("ViT-Tiny CIFAR-10 Configuration")
    print("=" * 70)
    print("\nModel Configuration:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")

    print("\nTraining Configuration:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n" + TRAINING_NOTES)
