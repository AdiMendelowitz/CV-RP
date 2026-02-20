"""
ViT-Tiny CIFAR-10 — Evaluation & Metrics
=========================================
Loads a trained checkpoint and produces:
    - Per-class accuracy breakdown
    - Confusion matrix
    - Training history curves
    - Sample predictions with confidence scores
    - Misclassified examples

Runs on CPU. No GPU required.
For attention analysis, use visualize_attention.py.

Prerequisite:
    checkpoints/best_vit_cifar10.pth  (downloaded from Kaggle output tab)
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ── Constants ─────────────────────────────────────────────────────────────────

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

CHECKPOINT_PATH = Path('checkpoints/best_vit_cifar10.pth')
OUTPUT_DIR = Path('outputs/evaluation')

CONFIG = {
    'image_size':   (32 // 4) ** 2,
    'patch_size':   4,
    'num_classes':  10,
    'embed_dim':    192,
    'depth':        12,
    'num_heads':    3,
    'mlp_ratio':    4.0,
    'dropout':      0.0,
    'attn_dropout': 0.0,
}


# ── Architecture ──────────────────────────────────────────────────────────────
# Must match the Kaggle training checkpoint state_dict keys exactly.

class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int = 3, embed_dim: int = 192) -> None:
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x).flatten(2).transpose(1, 2))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f'embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})')
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden  = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class ViTTiny(nn.Module):
    """
    Config matches DeiT-Tiny: embed_dim=192, depth=12, heads=3
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(cfg['image_size'], cfg['patch_size'], embed_dim=cfg['embed_dim'])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg['embed_dim']))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg['num_patches'] + 1, cfg['embed_dim']))
        self.pos_drop = nn.Dropout(cfg['dropout'])
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg['embed_dim'], cfg['num_heads'], cfg['mlp_ratio'],
                             cfg['dropout'], cfg['attn_dropout'])
            for _ in range(cfg['depth'])
        ])
        self.norm = nn.LayerNorm(cfg['embed_dim'])
        self.head = nn.Linear(cfg['embed_dim'], cfg['num_classes'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])


# ── Model & Data Loading ──────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ViTTiny, dict]:
    """Load trained ViT-Tiny from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f'Checkpoint not found at {checkpoint_path}.\n'
            'Download best_vit_cifar10.pth from Kaggle output tab and place it in checkpoints/'
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ViTTiny(CONFIG).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'✓ Checkpoint loaded — epoch {ckpt["epoch"]}, best acc {ckpt["best_acc"]:.2f}%')
    return model, ckpt


def load_data(batch_size: int = 256) -> tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10, DataLoader]:
    """Return normalised dataset, raw dataset, and test DataLoader."""
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    raw_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return test_dataset, raw_dataset, test_loader


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_all_predictions(model: ViTTiny, loader: DataLoader, device: torch.device) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run full test set inference.

    Returns:
        all_preds:  (N,) predicted class indices
        all_labels: (N,) true class indices
        all_confs:  (N,) softmax confidence of the predicted class
    """
    preds, labels, confs = [], [], []
    for images, lbls in loader:
        probs = torch.softmax(model(images.to(device)), dim=1).cpu()
        conf, pred = probs.max(dim=1)
        preds.append(pred)
        labels.append(lbls)
        confs.append(conf)
    return torch.cat(preds), torch.cat(labels), torch.cat(confs)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_training_history(history: dict, best_acc: float, out_dir: Path) -> None:
    """Four-panel training history: loss, accuracy, LR schedule, zoomed test acc."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['test_loss'],  label='Test',  linewidth=2)
    axes[0, 0].set_title('Loss'); axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, history['train_acc'], label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['test_acc'],  label='Test',  linewidth=2)
    axes[0, 1].axhline(best_acc, linestyle='--', color='green', alpha=0.7, label=f'Best {best_acc:.2f}%')
    axes[0, 1].set_title('Accuracy (%)'); axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epochs, history['lr'], linewidth=2, color='purple')
    axes[1, 0].set_title('Learning Rate Schedule'); axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_yscale('log'); axes[1, 0].grid(alpha=0.3)

    test_acc = history['test_acc']
    axes[1, 1].plot(epochs, test_acc, linewidth=2, color='coral')
    axes[1, 1].set_title('Test Accuracy (Zoomed)'); axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylim(min(test_acc) - 3, max(test_acc) + 2)
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle('ViT-Tiny CIFAR-10 — Training History', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print('✓ training_history.png saved')


def plot_per_class_accuracy(all_preds: torch.Tensor, all_labels: torch.Tensor,
                            out_dir: Path) -> list[float]:
    """Bar chart of per-class accuracy. Also prints a text breakdown. Returns class_acc list."""
    overall_acc = (all_preds == all_labels).float().mean().item() * 100

    class_correct = torch.zeros(10, dtype=torch.long)
    class_total = torch.zeros(10, dtype=torch.long)
    for i in range(10):
        mask = all_labels == i
        class_total[i] = mask.sum()
        class_correct[i] = (all_preds[mask] == i).sum()

    class_acc = (100.0 * class_correct / class_total).tolist()

    print(f'\n{"Class":>12}  {"Acc":>6}  {"Correct":>8}  {"Total":>6}')
    print('-' * 40)
    for cls, acc, correct, total in zip(CIFAR_CLASSES, class_acc, class_correct.tolist(), class_total.tolist()):
        print(f'{cls:>12}  {acc:6.2f}%  {correct:8d}  {total:6d}')
    print('-' * 40)
    print(f'{"Overall":>12}  {overall_acc:6.2f}%')

    colors = ['#2ecc71' if a >= 90 else '#f39c12' if a >= 80 else '#e74c3c' for a in class_acc]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(CIFAR_CLASSES, class_acc, color=colors, edgecolor='white', linewidth=0.8)
    ax.axhline(overall_acc, linestyle='--', color='black', alpha=0.5, label=f'Overall {overall_acc:.1f}%')
    ax.set_title('Per-Class Test Accuracy — ViT-Tiny', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 105)
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    for bar, acc in zip(bars, class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{acc:.1f}%', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / 'per_class_accuracy.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print('✓ per_class_accuracy.png saved')
    return class_acc


def plot_confusion_matrix(all_preds: torch.Tensor, all_labels: torch.Tensor, out_dir: Path) -> None:
    """Row-normalised confusion matrix with raw counts and proportions annotated."""
    # Vectorised: encode (true, pred) pairs as a single index, count with bincount
    conf = torch.bincount(all_labels * 10 + all_preds, minlength=100).reshape(10, 10)

    conf_norm = conf.float() / conf.sum(dim=1, keepdim=True)
    conf_np = conf.numpy()
    norm_np = conf_norm.numpy()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(norm_np, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax.set_xticks(range(10)); ax.set_xticklabels(CIFAR_CLASSES, rotation=45, ha='right')
    ax.set_yticks(range(10)); ax.set_yticklabels(CIFAR_CLASSES)
    ax.set_xlabel('Predicted Label', fontsize=12); ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix (counts and proportions) — ViT-Tiny', fontsize=13, fontweight='bold', pad=20)

    for i in range(10):
        for j in range(10):
            ax.text(j, i, f'{conf_np[i, j]}\n({norm_np[i, j]:.2f})', ha='center', va='center', fontsize=8,
                    color='white' if norm_np[i, j] > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print('\nTop misclassifications:')
    off_diag = [(conf[i, j].item(), CIFAR_CLASSES[i], CIFAR_CLASSES[j]) for i in range(10) for j in range(10) if i != j]
    for count, true_cls, pred_cls in sorted(off_diag, reverse=True)[:5]:
        print(f'  {true_cls:<12} → {pred_cls:<12}  ({count} samples)')
    print('✓ confusion_matrix.png saved')


def plot_sample_predictions(raw_dataset: torchvision.datasets.CIFAR10, all_preds: torch.Tensor,
                            all_labels: torch.Tensor, all_confs: torch.Tensor, out_dir: Path,
                            num_samples: int = 16) -> None:
    """
    Random grid of predictions with confidence scores.
    Green title = correct, red = wrong.
    Uses the raw (unnormalised) dataset directly — images are already in [0, 1].
    """
    indices = np.random.choice(len(raw_dataset), num_samples, replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()

    for ax, idx in zip(axes, indices):
        img, _ = raw_dataset[idx]                      # already [0,1], no normalisation
        true_cls = CIFAR_CLASSES[all_labels[idx].item()]
        pred_cls = CIFAR_CLASSES[all_preds[idx].item()]
        conf = all_confs[idx].item()
        correct = all_preds[idx] == all_labels[idx]

        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axis('off')
        ax.set_title(f'True: {true_cls}\nPred: {pred_cls}\nConf: {conf:.2f}', fontsize=9, color='green' if correct else 'red')

    plt.suptitle('Sample Predictions — ViT-Tiny', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print('✓ sample_predictions.png saved')


def plot_misclassified(all_preds: torch.Tensor, all_labels: torch.Tensor, raw_dataset: torchvision.datasets.CIFAR10,
                       out_dir: Path, n_samples: int = 16) -> None:
    """Grid of misclassified examples. Shows true vs predicted label."""
    wrong_idx = (all_preds != all_labels).nonzero(as_tuple=True)[0].tolist()
    print(f'\nMisclassified: {len(wrong_idx):,} / {len(raw_dataset):,} '
          f'({100 * len(wrong_idx) / len(raw_dataset):.1f}%)')

    sample = wrong_idx[:n_samples]
    n_cols = 4
    n_rows = math.ceil(len(sample) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))
    axes = axes.flatten()

    for ax, idx in zip(axes, sample):
        img, _ = raw_dataset[idx]
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(f'True: {CIFAR_CLASSES[all_labels[idx].item()]}\n'
                     f'Pred: {CIFAR_CLASSES[all_preds[idx].item()]}',
                     fontsize=8, color='red')
        ax.axis('off')
    for ax in axes[len(sample):]:
        ax.axis('off')

    plt.suptitle('Misclassified Samples — ViT-Tiny', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'misclassified_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print('✓ misclassified_samples.png saved')



if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cpu')
    print(f'Device: {device}')

    print('\n' + '=' * 70)
    print('ViT-Tiny CIFAR-10 — Evaluation')
    print('=' * 70)

    model, ckpt = load_model(CHECKPOINT_PATH, device)
    test_dataset, raw_dataset, test_loader = load_data()
    print(f'✓ Test data loaded ({len(test_dataset):,} samples)')

    print('\nRunning inference...')
    all_preds, all_labels, all_confs = get_all_predictions(model, test_loader, device)

    print('\n── Training History ──')
    plot_training_history(ckpt['history'], ckpt['best_acc'], OUTPUT_DIR)

    print('\n── Per-Class Accuracy ──')
    plot_per_class_accuracy(all_preds, all_labels, OUTPUT_DIR)

    print('\n── Confusion Matrix ──')
    plot_confusion_matrix(all_preds, all_labels, OUTPUT_DIR)

    print('\n── Sample Predictions ──')
    plot_sample_predictions(raw_dataset, all_preds, all_labels, all_confs, OUTPUT_DIR)

    print('\n── Misclassified Samples ──')
    plot_misclassified(all_preds, all_labels, raw_dataset, OUTPUT_DIR)

    total_params = sum(p.numel() for p in model.parameters())
    overall_acc = (all_preds == all_labels).float().mean().item() * 100
    print('\n' + '=' * 70)
    print('Evaluation complete')
    print('=' * 70)
    print(f'Parameters:     {total_params:,}')
    print(f'Epochs trained: {ckpt["epoch"]}')
    print(f'Best acc:       {ckpt["best_acc"]:.2f}%')
    print(f'Eval acc:       {overall_acc:.2f}%')
    print(f'Outputs:        {OUTPUT_DIR.resolve()}')
    print('=' * 70)
