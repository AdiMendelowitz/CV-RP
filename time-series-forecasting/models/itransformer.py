"""
iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.

Reference: Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series
           Forecasting", ICLR 2024 Spotlight. https://arxiv.org/abs/2310.06625

Standard transformers tokenize across the time axis where each timestep is a token, so attention
captures temporal correlations while variates are features within each token.
iTransformer inverts this by turning each variate's full history into a single token, so attention captures
cross-variate correlations instead.
This design benefits high-dimensional datasets (e.g., ECL: 321 variates) where cross-variate structure dominates, but
underperforms on low-dimensional datasets (e.g., ETTh1: 7 variates) where strong local temporal patterns are the
primary predictive signal.

The TransformerEncoder is reused verbatim from patchtst.py. The architecture is identical, only the
tokenization strategy differs.
"""

import torch
import torch.nn as nn

from models.patchtst import TransformerEncoder


class VariateEmbedding(nn.Module):
    """
    Embed each variate's full history as a single token.
    Transposes the input so that the variate dimension becomes the sequence dimension, then projects each
    variate's seq_len-length history to d_model via a shared linear layer.

    Args:
        seq_len: Input sequence length (number of timesteps).
        d_model: Embedding dimension for each variate token.
    """

    def __init__(self, seq_len: int, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project variate histories to token embeddings.

        Args:
            x: Input tensor of shape (B, seq_len, C).

        Returns:
            Tensor of shape (B, C, d_model), one token per variate.
        """
        x = x.transpose(1, 2)  # (B, seq_len, C) -> (B, C, seq_len)
        return self.projection(x)  # (B, C, d_model)


class ForecastHead(nn.Module):
    """
    Per-variate linear projection from d_model to pred_len.
    A shared linear layer is applied identically to each variate token, projecting from d_model to pred_len and
    then transposing to match the standard (B, pred_len, C) output convention.

    Args:
        d_model: Embedding dimension.
        pred_len: Number of future timesteps to forecast.
    """

    def __init__(self, d_model: int, pred_len: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project variate tokens to forecasts.

        Args:
            x: Tensor of shape (B, C, d_model).

        Returns:
            Tensor of shape (B, pred_len, C).
        """
        x = self.projection(x)  # (B, C, d_model) -> (B, C, pred_len)
        return x.transpose(1, 2)  # (B, pred_len, C)


class iTransformer(nn.Module):
    """
    iTransformer for multivariate long-horizon time series forecasting.

    Architecture:
        1. VariateEmbedding: each variate's full history -> one token (B, C, d_model).
        2. TransformerEncoder: attention runs over C variate tokens, capturing cross-variate correlations.
        3. ForecastHead: per-variate linear projection -> (B, pred_len, C).

    The variate dimension == sequence dimension after the embedding step, so the encoder operates on C tokens directly.

    Args:
        seq_len: Input sequence length.
        pred_len: Forecast horizon.
        d_model: Embedding dimension.
        num_heads: Number of attention heads. Must divide d_model evenly.
        num_layers: Number of transformer encoder layers.
        dropout: Dropout probability applied in the encoder.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        self.embedding = VariateEmbedding(seq_len=seq_len, d_model=d_model)
        self.encoder = TransformerEncoder(d_model=d_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        self.head = ForecastHead(d_model=d_model, pred_len=pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the full iTransformer forward pass.

        Args:
            x: Input tensor of shape (B, seq_len, C).

        Returns:
            Forecast tensor of shape (B, pred_len, C).
        """

        tokens = self.embedding(x)  # (B, seq_len, C) -> (B, C, d_model)
        encoded = self.encoder(tokens)  # (B, C, d_model)  [attention over C variate tokens]
        return self.head(encoded)  # (B, pred_len, C)
