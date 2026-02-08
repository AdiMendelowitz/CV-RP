"""Geomatric Transformers for Classical Computer Vision."""

import numpy as np
from typing import Tuple, List
def get_affine_transform(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Compute the affine transformation matrix from points correspondences.
    Args:
        src_points: Source points, shape (3,2) - [[x1, y1], [x2, y2], [x3, y3]]
        dst_points: Destination points, shape (3,2)
    Returns:
        Affine transformation matrix of shape (2,3)

    References: Szelinski 3.6.1
    """
    try:
        if src_points.shape != (3, 2):
            raise ValueError("src_points must be of shape (3, 2)")
        elif dst_points.shape != (3, 2):
            raise ValueError("dst_points must be of shape (3, 2)")
        else:
            print("Input points are valid.")
    except AttributeError as e:
        raise ValueError(f"Validation error {e}")

    A, b = [], []
    for i in range(3):
        x_src, y_src= src_points[i]
        x_dst, y_dst = dst_points[i]
        A.extend([[x_src, y_src, 1, 0, 0, 0], [0, 0, 0, x_src, y_src, 1]])
        b.extend([x_dst, y_dst])

    A = np.array(A)
    b =  np.array(b)

    params = np.linalg.lstsq(A, b, rcond=None)[0]
    affine_matrix = params.reshape(2, 3)

    return affine_matrix

def get_perspective_transform(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Compute the perspective transformation matrix from points correspondences.
    Args:
        src_points: Source points, shape (4,2) - [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        dst_points: Destination points, shape (4,2)
    Returns:
        Perspective transformation matrix of shape (3,3)
    References: Szelinski 3.6.2
    """
    try:
        if src_points.shape != (4, 2):
            raise ValueError("src_points must be of shape (4, 2)")
        elif dst_points.shape != (4, 2):
            raise ValueError("dst_points must be of shape (4, 2)")
        else:
            print("Input points are valid.")
    except AttributeError as e:
        raise ValueError(f"Validation error {e}")

    A = []
    for i in range(4):
        x_src, y_src= src_points[i]
        x_dst, y_dst = dst_points[i]
        A.extend([
            [-x_src, -y_src, -1, 0, 0, 0, x_src*x_dst, y_src*x_dst, x_dst],
            [0, 0, 0, -x_src, -y_src, -1, x_src*y_dst, y_src*y_dst, y_dst]
        ])

    A = np.array(A)

    #Solve SVD for  Ah=0  homogeneous system
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]  # Last row of Vt corresponds to the smallest singular value
    perspective_matrix = h.reshape(3, 3)

    return perspective_matrix/perspective_matrix[2,2] # Normalize so that bottom-right value is 1

def warp_affine(image: np.ndarray, matrix: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    """Apply affine transformation to the image.
    Args:
        image: Input image (H, W) or (H, W, C).
        matrix: Affine transformation matrix (2,3).
        output_shape: (height, width) of output image.
    Returns:
        Warped image
    """
    height, width = output_shape

    if image.ndim == 2:
        outout = np.zeros((height, width), dtype=image.dtype)
    else:
        outout = np.zeros((height, width, image.shape[2]), dtype=image.dtype)

    # Inverse transformaion (destination to source)
    # Add homogeneous coordinate
    M_full = np.vstack([matrix, [0, 0, 1]])
    M_inv = np.linalg.inv(M_full)[:2]

    # Coordinate grid of output image
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones(height*width)])  # (3, H*W)

    src_coords = M_inv @ coords  # (2, H*W)
    src_x = src_coords[0].reshape(height, width)
    src_y = src_coords[1].reshape(height, width)

    # Bilinear interpolation
    output = _bilinear_interpolate(image, src_x, src_y)

    return output

def warp_perspective(image: np.ndarray, matrix: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    """Apply perspective transformation to the image.
    Args:
        image: Input image (H, W) or (H, W, C).
        matrix: Perspective matrix (3,3).
        output_shape: (height, width) of output image.
    Returns:
        Warped image
    """
    height, width = output_shape

    if image.ndim == 2:
        outout = np.zeros((height, width), dtype=image.dtype)
    else:
        outout = np.zeros((height, width, image.shape[2]), dtype=image.dtype)

    # Inverse transformaion
    H_inv = np.linalg.inv(matrix)

    y_coords, x_coords = np.mgrid[0:height, 0:width]
    coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones(height*width)])

    src_coords_homogeneous = H_inv @ coords

    w = src_coords_homogeneous[2]
    src_x = (src_coords_homogeneous[0] / w).reshape(height, width)
    src_y = (src_coords_homogeneous[1] / w).reshape(height, width)

    output = _bilinear_interpolate(image, src_x, src_y)

    return output


def _bilinear_interpolate(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation for image sampling
    Args:
        image: Input image (H, W) or (H, W, C).
        x: x coordinates to sample (float)
        y: y coordinates to sample (float)
    :return:
        Interpolated values
    """

    h, w = image.shape[:2]
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # Clip coordinates to image boundaries
    x0 = np.clip(x0, 0, w-1)
    x1 = np.clip(x1, 0, w-1)
    y0 = np.clip(y0, 0, h-1)
    y1 = np.clip(y1, 0, h-1)

    # Fractional part
    fx = x - x0
    fy = y - y0

    # Sample at 4 corners
    if image.ndim == 2: # Grayscale
        I00 = image[y0, x0]
        I01 = image[y1, x0]
        I10 = image[y0, x1]
        I11 = image[y1, x1]

        # Bilinear weights
        w00 = (1 - fx) * (1 - fy)
        w01 = (1 - fx) * fy
        w10 = fx * (1 - fy)
        w11 = fx * fy

        result = w00*I00 + w01*I01 + w10*I10 + w11*I11

    else: # Color
        result = np.zeros_like(x)[..., None].repeat(image.shape[2], axis=-1)
        for c in range(image.shape[2]):
            I00 = image[y0, x0, c]
            I01 = image[y1, x0, c]
            I10 = image[y0, x1, c]
            I11 = image[y1, x1, c]

            w00 = (1 - fx) * (1 - fy)
            w01 = (1 - fx) * fy
            w10 = fx * (1 - fy)
            w11 = fx * fy

            result[..., c] = w00*I00 + w01*I01 + w10*I10 + w11*I11

    return result.astype(image.dtype)

def rotate(image: np.ndarray, angle_degrees: float, center: Tuple[float, float], ) -> np.ndarray:
    """Rotate image around center point.
    Args:
        image: Input image.
        angle_degrees: Rotation angle in degrees.
        center: (x, y) coordinates of the rotation center.
    Returns:
        Rotated image, same size as input.
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w/2, h/2)

    cx, cy = center
    angle_rad = np.deg2rad(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Rotation matrix around center
    # T(-c) * R(theta) * T(c)
    matrix = np.array([
        [cos_a, -sin_a, -cx*cos_a + cy*sin_a + cx],
        [sin_a,  cos_a, -cx*sin_a - cy*cos_a + cy]
    ])
    return warp_affine(image, matrix, output_shape=(h, w))

def resize(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize image by a scale factor.
    Args:
        image: Input image.
        scale: Scaling factor (>1 enlarges, <1 shrins).
    Returns:
        Resized image.
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h*scale), int(w*scale)
    matrix = np.array([[1/scale, 0, 0], [0, 1/scale, 0]])
    return warp_affine(image, matrix, output_shape=(new_h, new_w))
