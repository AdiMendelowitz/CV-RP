"""
RoI Align: Region Of Interest (ROI) feature extraction with bilinear interpolation.

Implements the quantization-free pooling operation from Mask R-CNN He et al.,
ICCV 2017. https://arxiv.org/abs/1703.06870

All box coordinates are in absolute feature-map space (pre scaled).
"""

import torch

def bilinear_interpolation(feature_map: torch.Tensor, row_c: torch.Tensor, col_c: torch.Tensor) -> torch.Tensor:
    """
    Sample a feature map at continuous (x, y) coordinates using bilinear interpolation.
    Coordinates outside the feature map boundaries are clamped to the nearest valid pixel before interpolation,
    matching the boundary behaviour used in the Mask R-CNN reference implementation.

    Args:
        feature_map: Feature map of shape (C, H, W).
        row_c: Floating point row coordinates, shape (N,).
        col_c: Floating point column coordinates, shape (N,).

    Returns:
        Interpolated feature map, shape (C, N).
    """

    _, H, W = feature_map.shape

    row_c = row_c.clamp(min=0, max=float(H - 1))
    col_c = col_c.clamp(min=0, max=float(W - 1))

    x0 = row_c.floor().long()
    y0 = col_c.floor().long()
    x1 = (x0 + 1).clamp(max=H-1)
    y1 = (y0 + 1).clamp(max=W-1)

    # Bilinear weights: fractional distance from the upper-left neighbour.
    w_x1 = (row_c - x0.float()).unsqueeze(0)  # (1, N)
    w_x0 = 1.0 - w_x1
    w_y1 = (col_c - y0.float()).unsqueeze(0)  # (1, N)
    w_y0 = 1.0 - w_y1

    # Sample the neighbours. Each one of shape (C,N)
    n00 = feature_map[:, x0, y0]
    n01 = feature_map[:, x0, y1]
    n10 = feature_map[:, x1, y0]
    n11 = feature_map[:, x1, y1]

    return w_x0*w_y0*n00 + w_x0*w_y1*n01 + w_x1*w_y0*n10 + w_x1*w_y1*n11

def roi_align(feature_map: torch.Tensor, boxes: torch.Tensor, output_size: int,
              sampling_ratio: int = 2) -> torch.Tensor:
    """
    Extract fixed-sized features for a set of RoI boxes using RoI Align.

    Each box is divided into a grid of (output_size x output_size) bins.
    In each bin sampling_ratio x sampling_ratio points are placed on a regular sub-grid and averaged.
    No coordinate quantization is applied, matching He et al., ICCV 2017.

    Args:
        feature_map: Feature map of shape (C, H, W).
        boxes: RoI boxes in absolute feature-map coordinates, shape (N, 4), in (x1, y1, x2, y2) format.
        output_size: Spatial size of the output feature grid.
        sampling_ratio: Number of sampling points per bin side, total samples per bin: sampling_ratio^2.

    Returns:
        Pooled RoI features, shape (R, C, output_size, output_size).
    """

    C, H, W = feature_map.shape
    R = boxes.shape[0]
    device = feature_map.device

    output = torch.zeros(R, C, output_size, output_size, device=device, dtype=feature_map.dtype)

    for r in range(R):
        x1, y1, x2, y2 = boxes[r].unbind(0)
        roi_w = (x2 - x1).clamp(min=1e-7)
        roi_h = (y2 - y1).clamp(min=1e-7)

        bin_w = roi_w / output_size
        bin_h = roi_h / output_size

        # Build all sample point coordinates
        bin_i = torch.arange(output_size, device=device, dtype=torch.float32) # row
        bin_j = torch.arange(output_size, device=device, dtype=torch.float32) # column
        s = torch.arange(sampling_ratio, device=device, dtype=torch.float32)

        # Even spacing within the bin
        offset = (s + 0.5) / sampling_ratio
        sample_y = y1 + (bin_i.unsqueeze(1) + offset.unsqueeze(0)) * bin_h   # (output_size, sampling_ratio)
        sample_x = x1 + (bin_j.unsqueeze(1) + offset.unsqueeze(0)) * bin_w   # (output_size, sampling_ratio)

        # Expand to all (bin_i, bin_j, si, sj) combinations.
        # y depends on row bin and si, x on col bin and sj.
        # Flatten to 1D of (output_size^2)*(sampling_ratio^2) length for a single batched interpolation call
        y_coords = sample_y.unsqueeze(1).unsqueeze(3).expand(
            output_size, output_size, sampling_ratio, sampling_ratio
        ).reshape(-1)
        x_coords = sample_x.unsqueeze(0).unsqueeze(2).expand(
            output_size, output_size, sampling_ratio, sampling_ratio
        ).reshape(-1)

        # Interpolate all sample points at once, (C, total_samples)
        samples = bilinear_interpolation(feature_map, y_coords, x_coords)

        samples = samples.reshape(C, output_size, output_size, sampling_ratio*sampling_ratio)

        output[r] = samples.mean(dim=-1)

    return output
