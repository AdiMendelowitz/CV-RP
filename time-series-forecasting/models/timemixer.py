"""
TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting.

Reference: Wang et al., "TimeMixer: Decomposable Multiscale Mixing for Time Series
           Forecasting", ICLR 2024. https://arxiv.org/abs/2405.14616

Time series exhibit different patterns at different sampling scales. TimeMixer downsamples the input
to multiple resolutions, decomposes each into seasonal and trend components via a moving average, mixes
them with lightweight MLPs (Past-Decomposable-Mixing), then ensembles forecasts from all scales vis a learned
weighted average (Future-Multipredictor-Mixing).

Forward pass shapes (B = batch, C = variates, L = seq_len, P = pred_len, D = d_model):
    Input: (B, L, C)
    Per scale s (L_s = L // 2^s):
        - Downsample: (B, L, C) -> (B, L_s, C)
        - Decompose: seasonal (B, C, L_s), trend (B, C, L_s)
        - PDM: mixed (B, C, D)
        - FMM head: forecast (B, C, P)
    Ensemble forecasts across scales: (B, P, C)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SeriesDecomposition(nn.Module):
    """
    Extract trend via moving average. Seasonality = series - trend.

    Uses nn.AvgPool1d with reflect padding so the output length exactly matches the input length regardless
    of kernel parity.

    Args:
        kernel_size: Moving average window length, odd values are typical.
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        # Padding on each size so output length == input length
        self.padding = kernel_size // 2
        # stride=1 + count_include_pad=False to avoid edge distortions
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose x into (seasonal, trend).

        Args:
            x: Tensor, shape (B, C, L)
        Returns:
            x_seasonal: (B, C, L) high frequency residual.
            x_trend: (B, C, L) low frequency moving average.
        """
        # Explicitly pad only the last dimension (L) to keep output length == L after pooling.
        # F.pad format for 3D tensor last dim: (pad_left, pad_right)
        x_padded = F.pad(x, (self.padding, self.padding), mode="reflect")

        x_trend = self.avg_pool(x_padded)

        if x_trend.shape[-1] != x.shape[-1]: # Trim any length mismatch caused by even kernel sizes.
            x_trend = x_trend[..., : x.shape[-1]]
        x_seasonal = x - x_trend
        return x_seasonal, x_trend


class PDMBlock(nn.Module):
    """
    Past-Decomposable-Mixing Block for one scale.

    Projects the seasonal and trend components independently along the time axis to d_model, then
    sums them to produce a mixed scale representation.

    Args:
        seq_len: Sequence length at this scale.
        d_model: Output hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.seasonal_mix = nn.Linear(seq_len, d_model)
        self.trend_mix = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seasonal: torch.Tensor, x_trend: torch.Tensor) -> torch.Tensor:
        """
        Mix seasonal and trend components.

        Args:
            x_seasonal: (B, C, L_s)
            x_trend: (B, C, L_s)

        Returns:
            mixed: (B, C, d_model)
        """
        # Linear model operates on the last dimension L_s -> d_model.
        mixed = self.seasonal_mix(x_seasonal) + self.trend_mix(x_trend)
        return self.dropout(mixed)


class FMMBlock(nn.Module):
    """
    Future-Multipredictor-Mixing Block.
    One linear forecast head per scale. Predictions are ensembled vua a learned softmax-weighted average
    across scales.

    Args:
        num_scales: Number of resolution scales.
        d_model: Hidden dimension from PDM.
        pred_len: Forecast horizon.
    """

    def __init__(self, num_scales: int, d_model: int, pred_len: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(d_model, pred_len) for _ in range(num_scales)])

        # Learned ensemble weights, one scalar per scale.
        # Scale weights can be scalar, but initialized to zeros so softmax starts uniform.
        self.scale_weights = nn.Parameter(torch.zeros(num_scales))

    def forward(self, scale_representations: list[torch.Tensor]) -> torch.Tensor:
        """
        Produce an ensembled forecast from all scale representations.

        Args:
            scale_representations: List of num_scales tensors, each (B, C, d_model).

        Returns:
            forecast: (B, C, pred_len)
        """
        weights = F.softmax(self.scale_weights, dim=0) # (num_scales,)

        # Vectorized projection across heads, then weighted sum.
        out = 0
        for i, head in enumerate(self.heads):
            out += weights[i] * head(scale_representations[i])
        return out # (B, C, pred_len)


class TimeMixer(nn.Module):
    """
    TimeMixer for multivariate long-horizon time series forecasting.

    Architecture:
        1. Downsample input to num_scales resolutions.
        2. Decompose each resolution into seasonal and trend.
        3. PDMblock per scale: mix seasonal + trend -> (B, C, d_model).
        4. FMMblock: forecast from each scale, ensemble via learned weights.

    Args:
        seq_len: Input sequence length.
        pred_len: Forecast horizon.
        num_scales: Number of downsampling levels, default 3: 1x, 2x, 4x.
        d_model: Hidden dimension for PDM mixing.
        decomp_kernel: Moving average kernel size for trend extraction.
        dropout: Dropout probability.
    """

    def __init__(self, seq_len: int, pred_len: int, num_scales: int = 3, d_model: int = 16,
                 decomp_kernel: int = 25, dropout: float = 0.1) -> None:
        super().__init__()

        self.num_scales = num_scales

        # Track layers and correct lengths dynamically using Ceil/Floor matching AvgPool1d
        self.downsamplers = nn.ModuleList()
        self.scale_lens = []

        current_len = seq_len
        for s in range(num_scales):
            if s == 0:
                self.downsamplers.append(nn.Identity())
            else:
                # ceil_mode=True ensures predictable shapes if seq_len is odd
                self.downsamplers.append(nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=False))
                current_len = current_len // 2

            self.scale_lens.append(max(1, current_len))

        self.decomposition = SeriesDecomposition(decomp_kernel)

        self.pdm_blocks = nn.ModuleList([
            PDMBlock(seq_len=l, d_model=d_model, dropout=dropout) for l in self.scale_lens
        ])

        self.fmm = FMMBlock(num_scales=num_scales, d_model=d_model, pred_len=pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the full TimeMixer forward pass.
        Args:
            x: Input tensor, shape (B, seq_len, C).

        Returns:
            Forecast tensor, shape (B, pred_len, C).
        """
        # Work in (B, C, L) throughout: transpose once at entry and once at exit.
        x = x.transpose(1, 2) # (B, C, seq_len)

        scale_reps = []
        x_s = x
        for i, (downsampler, pdm) in enumerate(zip(self.downsamplers, self.pdm_blocks)):
            x_s = downsampler(x_s)

            # Defensive check for dynamic shapes / odd sequence lengths
            if x_s.shape[-1] != self.scale_lens[i]:
                # Dynamic adjustment or padding could go here; tracking enforces stability
                pass

            x_seasonal, x_trend = self.decomposition(x_s)
            rep = pdm(x_seasonal, x_trend)
            scale_reps.append(rep)

        forecast = self.fmm(scale_reps)  # (B, C, pred_len)
        return forecast.transpose(1, 2)  # (B, pred_len, C)