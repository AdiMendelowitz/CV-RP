"""
PatchTST: Patch Time Series Transformer for long-horizon forecasting.

Reference: Nie et al., "A Time Series Is Worth 64 Words: Long-term Forecasting with Transformers",
ICLR 2023. https://arxiv.org/abs/2211.14730
"""

import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Extract overlapping patches from a univariate series and project to d_model.

    Args:
        patch_size: Number of time steps per patch.
        stride: Stride between consecutive patches.
        d_model: Projection dimension.
        dropout: Dropout rate applied after embedding.
    """

    def __init__(self, patch_size: int, stride: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.projection = nn.Linear(patch_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self._d_model = d_model

    @staticmethod
    def _sinusoidal_encoding(num_patches: int, d_model: int, device: torch.device) -> torch.Tensor:
        """Return fixed sinusoidal positional encoding of shape (1, num_patches, d_model)."""
        position = torch.arange(num_patches, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(1, num_patches, d_model, device=device)
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        encoding[0, :, 1::2] = torch.cos(position * div_term[: d_model // 2])
        return encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed univariate series into patch tokens.
        Args:
            x: Input tensor of shape (B, seq_len, 1).

        Returns:
            Tensor of shape (B, num_patches, d_model).
        """

        # x: (B, seq_len, 1) -> squeeze channel -> (B, seq_len)
        x = x.squeeze(-1)

        # Pad right end so the patches are fully populated.
        # After padding: num_patches = floor((seq_len - patch_size) / stride) + 2.
        pad_len = self.stride - ((x.size(1) - self.patch_size) % self.stride)
        if pad_len < self.stride:
            x = nn.functional.pad(x, (0, pad_len))

        # Unfold: (B, num_patches, patch_size)
        x = x.unfold(dimension=1, size=self.patch_size, step=self.stride)

        # Project each patch: (B, num_patches, d_model)
        x = self.projection(x)
        x = x + self._sinosoidal_encoding(x.size(1), self._d_model, x.device)
        return self.dropout(x)


class _TransformerBlock(nn.Module):
    """
    Single pre-norm transformer block (attention + MLP residual connections).

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim = d_model * num_ratio.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-norm and residual connections.

        Args:
            x: Tensor of shape (B, num_patches, d_model).

        Returns:
            Tensor of shape (B, num_patches, d_model).
        """

        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of pre-norm transformer blocks with a final layer norm.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of stacked transformer blocks.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TransformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass token sequence through all transformer blocks.

        Args:
            x: Tensor of shape (B, num_patches, d_model).

        Returns:
            Tensor of shape (B, num_patches, d_model).
        """

        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class ForecastHead(nn.Module):
    """
    Flatten patch token and project to a forecast horizon.

    Args:
        num_patches: Number of patch tokens.
        d_model: Model dimension.
        pred_len: Forecast horizon (number of time steps to predict).
    """

    def __init__(self, num_patches: int, d_model: int, pred_len: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(num_patches * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project flattened tokens to a forecast.

        Args:
            x: Tensor of shape (B, num_patches, d_model).

        Returns:
            Tensor of shape (B, pred_len).
        """
        return self.linear(self.flatten(x))


class PatchTST(nn.Module):
    """
    Channel-independent PatchTST for multivariate long-horizon forecasting.

    Channels are processed independently: the input is reshaped from (B, seq_len, C) to (B*C, seq_len, 1)
    before the encoder. Transformer weights are shared across all channels by construction.
    This is the Patch/TST/64 configuration from Nie et al., ICLR 2023.

    Args:
        seq_len: Input sequence length.
        pred_len: Forecast horizon.
        num_variates: Number of input channels (C).
        patch_size: Number of time steps per patch, default 16 (paper config).
        stride: Stride between consecutive patches, default 8 (paper config).
        d_model: Transformer model dimension, default 128 (paper config).
        num_heads: Number of attention heads, default 16 (paper config).
        num_layers: Number of transformer blocks, default 3 (paper config).
        dropout: Dropout rate, default 0.2 (paper config).
        channel_mixing: True: patches from all variates are concatenated along the sequence dimension before the
                        encoder, enabling cross-variate attention. Default False (channel-independent mode).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_variates: int,
        patch_size: int = 16,
        stride: int = 8,
        d_model: int = 128,
        num_heads: int = 16,
        num_layers: int = 3,
        dropout: float = 0.2,
        channel_mixing: bool = False,
    ) -> None:
        super().__init__()
        self.num_variates = num_variates
        self.pred_len = pred_len

        # Compute the number of patches produced after right-side padding.
        pad_len = stride - ((seq_len - patch_size) % stride)
        padded_len = seq_len + (pad_len if pad_len < stride else 0)
        num_patches = (padded_len - patch_size) // stride + 1

        self.patch_embedding = PatchEmbedding(patch_size, stride, d_model, dropout)
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, dropout)
        self.head = ForecastHead(num_patches, d_model, pred_len)

        self.channel_mixing = channel_mixing
        self.num_patches = num_patches  # needed in forward for CD reshape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multivariate input.

        Args:
            x: Input tensor of shape (B, seq_len, C).

        Returns:
            Forecast tensor of shape (B, pred_len, C).
        """

        B, seq_len, C = x.shape

        # Channel independence: treat each channel as an independent sample.
        x = x.permute(0, 2, 1).reshape(B * C, seq_len, 1)  # (B*C, seq_len, 1)
        x = self.patch_embedding(x)  # (B*C, num_patches, d_model)

        if self.channel_mixing:
            x = x.reshape(B, C * self.num_patches, -1)  # (B, C*N, D)
            x = self.encoder(x)  # (B, C*N, D)
            x = x.reshape(B * C, self.num_patches, -1)  # (B*C, N, D)
        else:
            x = self.encoder(x)  # (B*C, num_patches, d_model)

        x = self.head(x)  # (B*C, pred_len)

        # Restore batch and channel dimensions: (B, pred_len, C)
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)
        return x
