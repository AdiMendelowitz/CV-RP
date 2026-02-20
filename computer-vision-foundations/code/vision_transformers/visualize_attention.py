"""
Attention Map Visualisation for ViT-Tiny
=========================================
Loads a trained checkpoint and visualizes which patches each attention head
focuses on, for a given CIFAR-10 test image.

Usage:
    python visualize_attention.py
    python visualize_attention.py --checkpoint ./checkpoints/vit_tiny_cifar10/best_model.pth
    python visualize_attention.py --checkpoint ./path/to/model.pth --image_idx 42 --layer 11

What it produces:
    1. Attention maps for all 3 heads of a chosen layer (default: last layer)
    2. CLS token attention — which patches the model focused on per layer
    3. Attention rollout — propagated attention through the full 12-layer stack
"""


from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# ── Constants ────────────────────────────────────────────────────────────────

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

# Precomputed as numpy arrays — used in denorm() on every call, no allocation per call
_CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

# Used in DataLoader transforms (torchvision expects plain lists)
CIFAR10_MEAN = _CIFAR10_MEAN.tolist()
CIFAR10_STD  = _CIFAR10_STD.tolist()


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int=32, patch_size: int=4, in_channels: int=3, embed_dim: int=192) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.projection(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, x], dim=1) + self.positional_embedding


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention.
    When return_attn=True, forward returns (output, attn_weights) instead of output alone.
    attn_weights shape: (B, num_heads, N, N)
    """
    def __init__(self, embed_dim: int=192, num_heads: int=3, dropout: float = 0.0) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool=False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)  # each (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).flatten(2)
        out = self.proj_dropout(self.projection(out))

        if return_attn:
            return out, attn.detach()   # caller gets weights; no side effect state
        return out


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int]=None,
                 out_features: Optional[int]=None, dropout: float=0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int=192, num_heads: int=3, mlp_ratio: float=4.0, dropout: float=0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor, return_attn: bool=False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attn_out = self.attn(self.norm1(x), return_attn=return_attn)
        if return_attn:
            attn_out, weights = attn_out
            return x + attn_out + self.mlp(self.norm2(x + attn_out)), weights
        return x + attn_out + self.mlp(self.norm2(x + attn_out))


class VisionTransformerViz(nn.Module):
    """ViT with attention extraction. Only one forward interface needed."""

    def __init__(self, img_size: int=32, patch_size: int=4, in_channels: int=3, num_classes: int=10,
                 embed_dim: int=192, depth: int=12, num_heads: int=3, mlp_ratio: float=4.0,
                 dropout: float=0.0) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Always captures attention weights from every layer.

        Returns:
            logits:       (B, num_classes)
            attn_weights: list of L tensors, each (B, num_heads, N, N), on CPU
        """
        x = self.patch_embed(x)
        attn_weights: list[torch.Tensor] = []
        for block in self.blocks:
            x, w = block(x, return_attn=True)
            attn_weights.append(w.cpu())
        x = self.norm(x)
        return self.head(x[:, 0]), attn_weights



def load_model(checkpoint_path: str, device: str = 'cpu') -> VisionTransformerViz:
    model = VisionTransformerViz()

    # weights_only=True prevents arbitrary code execution from malicious .pth files
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint — epoch {checkpoint['epoch']}, test acc {checkpoint['test_acc']:.2f}%")
    return model.to(device)


def denorm(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse CIFAR-10 normalisation for display.
    Args:
        tensor: (C, H, W) torch tensor
    Returns:
        (H, W, C) float32 numpy array clipped to [0, 1]
    """
    img = tensor.cpu().numpy().transpose(1, 2, 0)   # (C,H,W) → (H,W,C)
    img = img * _CIFAR10_STD + _CIFAR10_MEAN         # no allocation — uses precomputed arrays
    return np.clip(img, 0.0, 1.0, out=img)           # in-place clip


def _normalise_map(attn_map: np.ndarray) -> np.ndarray:
    """Min-max normalise a 2D attention map to [0, 1]."""
    lo, hi = attn_map.min(), attn_map.max()
    return (attn_map - lo) / (hi - lo + 1e-8)


def _n_patches_per_side(attn_weights: torch.Tensor) -> int:
    """Derive grid size from attention weight tensor shape. N = seq_len, patches = N-1."""
    n_patches = attn_weights.shape[-1] - 1   # subtract CLS token
    return int(n_patches ** 0.5)


def attention_rollout(attn_weights_list: list[torch.Tensor]) -> np.ndarray:
    """
    Attention Rollout (Abnar & Zuidema, 2020): Propagates attention through all layers to estimate which input patches
    the CLS token is attending to after the full forward pass.

    Formula: R = A_hat_1 @ A_hat_2 @ ... @ A_hat_L
    where A_hat_i = 0.5 * mean_heads(A_i) + 0.5 * I  (residual correction)

    Args:
        attn_weights_list: list of (1, num_heads, N, N) tensors, one per layer
    Returns:
        (n_patches,) numpy array of rollout scores for each patch token (CLS excluded)
    """
    # Convert to numpy once upfront — all subsequent ops are pure numpy
    N = attn_weights_list[0].shape[-1]
    eye = np.eye(N, dtype=np.float32)               # built once, reused every iteration
    rollout = eye.copy()

    for attn in attn_weights_list:
        attn_avg = attn[0].numpy().mean(axis=0)     # (num_heads, N, N) → (N, N)
        attn_hat = 0.5 * attn_avg + 0.5 * eye
        attn_hat /= attn_hat.sum(axis=-1, keepdims=True)   # row-normalise in-place
        rollout = attn_hat @ rollout

    # Row 0 = CLS token. Columns 1: = patch tokens (drop CLS→CLS self-attention weight)
    return rollout[0, 1:]   # (n_patches,)



def _title(true_label: int, pred_label: int) -> str:
    t = CIFAR_CLASSES[true_label]
    p = CIFAR_CLASSES[pred_label]
    mark = "✓" if true_label == pred_label else "✗"
    return f"True: {t}  |  Predicted: {p} {mark}"


def plot_single_layer_heads(img_tensor: torch.Tensor, attn_weights: torch.Tensor, layer_idx: int,
                            true_label: int, pred_label: int, save_path: Optional[Path] = None,) -> None:
    """
    For a single layer, plot each attention head's CLS→patch attention map.

    Args:
        attn_weights: (1, num_heads, N, N)
    """
    num_heads = attn_weights.shape[1]
    grid = _n_patches_per_side(attn_weights)
    img = denorm(img_tensor[0])

    fig, axes = plt.subplots(1, num_heads + 1, figsize=(4 * (num_heads + 1), 4))
    fig.suptitle(f"Layer {layer_idx + 1} — Per-Head CLS Attention\n{_title(true_label, pred_label)}",
                 fontsize=12, fontweight='bold')

    axes[0].imshow(img)
    axes[0].set_title("Input", fontsize=10)
    axes[0].axis('off')

    for h in range(num_heads):
        # CLS row (position 0), drop CLS self-attention (column 0), reshape to grid
        cls_attn = attn_weights[0, h, 0, 1:].numpy()
        attn_map = _normalise_map(cls_attn.reshape(grid, grid))

        ax = axes[h + 1]
        ax.imshow(img)
        heatmap = ax.imshow(attn_map, cmap='hot', alpha=0.6, extent=[0, img.shape[1], img.shape[0], 0],
                            interpolation='bilinear')
        ax.set_title(f"Head {h + 1}", fontsize=10)
        ax.axis('off')
        plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)  # stored return value

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


def plot_all_layers_cls_attention( img_tensor: torch.Tensor, all_attn_weights: list[torch.Tensor],
                                   true_label: int, pred_label: int, save_path: Optional[Path] = None,) -> None:
    """
    Grid: CLS attention (mean over heads) for every layer.
    Shows how attention evolves from local (early) to global (late).
    """
    num_layers = len(all_attn_weights)
    grid = _n_patches_per_side(all_attn_weights[0])
    img = denorm(img_tensor[0])

    cols = 4
    rows = (num_layers + 1 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.ravel()

    fig.suptitle(f"CLS Attention — All {num_layers} Layers (mean over heads)\n"
                 f"{_title(true_label, pred_label)}",
                 fontsize=12, fontweight='bold')

    axes[0].imshow(img)
    axes[0].set_title("Input", fontsize=9)
    axes[0].axis('off')

    for i, attn_weights in enumerate(all_attn_weights):
        cls_attn = attn_weights[0].numpy().mean(axis=0)[0, 1:]  # (N-1,)
        attn_map = _normalise_map(cls_attn.reshape(grid, grid))

        ax = axes[i + 1]
        ax.imshow(img)
        ax.imshow(attn_map, cmap='hot', alpha=0.6, extent=[0, img.shape[1], img.shape[0], 0],
                  interpolation='bilinear')
        ax.set_title(f"Layer {i + 1}", fontsize=9)
        ax.axis('off')

    for i in range(num_layers + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


def plot_attention_rollout(img_tensor: torch.Tensor, all_attn_weights: list[torch.Tensor], true_label: int,
                           pred_label: int, save_path: Optional[Path] = None,) -> None:
    """
    Attention rollout — traces information flow through all layers.
    Most reliable indicator of which patches drove the final classification.
    """
    grid = _n_patches_per_side(all_attn_weights[0])
    img = denorm(img_tensor[0])

    rollout = attention_rollout(all_attn_weights)
    attn_map = _normalise_map(rollout.reshape(grid, grid))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Attention Rollout\n{_title(true_label, pred_label)}", fontsize=12, fontweight='bold')

    axes[0].imshow(img)
    axes[0].set_title("Input Image", fontsize=10)
    axes[0].axis('off')

    heatmap = axes[1].imshow(attn_map, cmap='hot', interpolation='bilinear')
    axes[1].set_title(f"Rollout Map ({grid}×{grid} patches)", fontsize=10)
    axes[1].axis('off')
    plt.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)  # stored return value, not axes[1].images[0]

    axes[2].imshow(img)
    axes[2].imshow(attn_map, cmap='hot', alpha=0.6, extent=[0, img.shape[1], img.shape[0], 0],
                   interpolation='bilinear')
    axes[2].set_title("Overlay", fontsize=10)
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    checkpoint_path='./checkpoints/vit_tiny_cifar10/best_model.pth'
    image_idx=0    # 0-9999
    layer=11       # 0-indexed; 11 = last layer
    device = 'cpu'
    save_dir = './attention_maps'

    print("Loading model...")
    model = load_model(checkpoint_path, device)

    num_layers = len(model.blocks)
    if not 0 <= layer < num_layers:
        raise ValueError(f"--layer must be in [0, {num_layers - 1}], got {layer}")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    n_test = len(test_dataset)
    if not 0 <= image_idx < n_test:
        raise ValueError(f"--image_idx must be in [0, {n_test - 1}], got {image_idx}")

    img_tensor, true_label = test_dataset[image_idx]
    img_tensor = img_tensor.unsqueeze(0).to(device)

    print(f"\nImage #{image_idx} — True class: {CIFAR_CLASSES[true_label]}")

    with torch.no_grad():
        logits, all_attn_weights = model(img_tensor)

    # Compute softmax once — derive both pred and confidence from it
    probs = logits.softmax(dim=1)[0]
    pred_label = probs.argmax().item()
    confidence = probs[pred_label].item()
    print(f"Predicted: {CIFAR_CLASSES[pred_label]} ({confidence:.1%}) {'✓' if pred_label == true_label else '✗'}")

    # Build save paths using Path — cross-platform
    save_paths: list[Optional[Path]] = [None, None, None]
    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_paths = [
            out / f"attn_layer{layer + 1}_heads.png",
            out / "attn_all_layers.png",
            out / "attn_rollout.png",
        ]

    print(f"\nPlotting layer {layer + 1} per-head attention...")
    plot_single_layer_heads(img_tensor, all_attn_weights[layer], layer, true_label, pred_label, save_path=save_paths[0])

    print("Plotting CLS attention across all layers...")
    plot_all_layers_cls_attention(img_tensor, all_attn_weights, true_label, pred_label, save_path=save_paths[1])

    print("Computing attention rollout...")
    plot_attention_rollout(img_tensor, all_attn_weights, true_label, pred_label, save_path=save_paths[2])

    print("\nDone.")













