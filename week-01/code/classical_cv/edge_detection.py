from typing import Any

import numpy as np
from filters import gaussian_blur, sobel_edges
from numpy import dtype, ndarray


def hysteresis(image: np.ndarray, strong, weak) -> np.ndarray:
    is_strong = (image==strong)
    is_weak = (image==weak)
    connected = is_strong.copy()
    neighbor_mask = np.ones(shape=(3, 3), dtype=bool)
    neighbor_mask[1, 1] = False  # Exclude center pixel

    while True:
        prev = connected.copy()
        padded = np.pad(array=connected, pad_width=1, mode='constant', constant_values=False)
        windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=(3, 3))
        has_strong_neighbor = np.any(windows & neighbor_mask, axis=(2, 3))
        connected = is_strong | (is_weak & has_strong_neighbor)
        if np.array_equal(connected, prev):
            break

    result = np.zeros_like(image, dtype='uint8')
    result[connected] = image.max()
    return result

def double_threshold(image: np.ndarray, low_ratio=0.05, high_ratio=0.15) -> tuple[ndarray[Any, dtype[Any]], int, int]:
    """Double thresholding for edge tracking by hysteresis"""
    high_threshold = image.max()*high_ratio
    low_threshold = image.max()*low_ratio

    strong, weak = 255, 75
    conditions = [image>=high_threshold, (high_threshold>image) & (image>=low_threshold)]
    choices = [strong, weak]
    result = np.select(conditions, choices, default=0).astype('uint8')
    return result, strong, weak

def non_max_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    suppressed = np.zeros_like(magnitude)
    angle = np.rad2deg(direction) % 180
    directions = [
        ((angle<22.5) | (angle>=157.5), [(0, 1), (0,-1)]),      # 0 degrees, neighbors right and left
        ((angle>=22.5) & (angle<67.5), [(-1, 1), (1, -1)]),     # 45 degrees, neighbors top-right and bottom-left
        ((angle>=67.5) & (angle<112.5), [(-1, 0), (1, 0)]),     # 90 degrees, neighbors top and bottom
        ((angle>=112.5) & (angle<157.5), [(-1, -1), (1, 1)]),   # 135 degrees, neighbors top-left and bottom-right
    ]

    for directed_pixel, [(dx1, dy1), (dx2, dy2)] in directions:
        neighbor1 = magnitude[1+dx1:-1+dx1 or None, 1+dy1:-1+dy1 or None]
        neighbor2 = magnitude[1+dx2:-1+dx2 or None, 1+dy2:-1+dy2 or None]

        interior_directed_pixel = directed_pixel[1:-1, 1:-1]
        local_max = (magnitude[1:-1, 1:-1] >= neighbor1) & (magnitude[1:-1, 1:-1] >= neighbor2)
        suppressed[1:-1, 1:-1][interior_directed_pixel & local_max] = magnitude[1:-1, 1:-1][interior_directed_pixel & local_max]

    return suppressed

def canny_edge_detector(image: np.ndarray, low_ratio=0.05, high_ratio=0.15, sigma=1.4) -> np.ndarray:
    """Args:
        Image: gray scale input, float, range [0, 1]
        low_threshold: low threshold for hysteresis
        high_threshold: high threshold for hysteresis
        sigma: Gaussian smoothing sigma
    Returns:
        Binary edge map (0 or 255)
    """

    blurred = gaussian_blur(image=image, kernel_dim=5, sigma=sigma)
    Gx, Gy, magnitude = sobel_edges(image=blurred)
    direction = np.arctan2(Gy, Gx)  # Edge direction
    nms_image = non_max_suppression(magnitude=magnitude, direction=direction)
    thresholded, strong, weak = double_threshold(image=nms_image, low_ratio=low_ratio, high_ratio=high_ratio)
    edges = hysteresis(thresholded, strong, weak)

    return edges


