import numpy as np
from typing import Tuple, Optional, Any
import matplotlib.pyplot as plt


def _pad_image(image: np.ndarray, pad_h: int, pad_w: int, mode: str='zero') -> np.ndarray:
    if mode == 'zero':
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    elif mode == 'replicate':
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    elif mode == 'mirror':
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    else:
        raise ValueError(f"Unknown padding mode: {mode}")

def correlated2d(image: np.ndarray, kernel: np.ndarray, padding: str='zero') -> np.ndarray:
    """Applies a 2D correlation operation on the input image using the given kernel.
    Args:
        image (np.ndarray): The input image (H,W) or (H,W,C).
        kernel (np.ndarray): filter kernel (kH, kW).
        padding (str): 'zero', 'replicate', 'mirror'

    Returns:
        Filtered image (H,W) or (H,W,C)
    """
    if image.ndim == 3: #handle grayscale vs color
        return np.stack([
            correlated2d(image[:,:,c], kernel, padding) for c in range(image.shape[2])
            ], axis=2)

    # Dimensions
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Padding
    padded = _pad_image(image=image, pad_h=pad_h, pad_w=pad_h, mode=padding)
    output = np.zeros_like(image, dtype='float32')

    # Sliding window correlation
    for i in range(h):
        for j in range(w):
            window = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(window * kernel)

    return output

def convolve2d(image: np.ndarray, kernel: np.ndarray, padding: str='zero') -> np.ndarray:
    """Convolution = correlation with flipped kernel."""
    flipped_kernel = np.flip(kernel)
    return correlated2d(image, flipped_kernel, padding)

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Generates a Gaussian kernel."""

    # Coordinate grid centered at zero
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

    # Normalize the kernel
    return kernel / np.sum(kernel)

def gaussian_blur(image: np.ndarray, kernel_dim: int=5, sigma: float=1.0) -> np.ndarray:
    """Apply gaussian blur"""
    kernel = gaussian_kernel(size=kernel_dim, sigma=sigma)
    return convolve2d(image, kernel, padding='replicate')

def sobel_edges(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sobel edge detection from scratch.

    Returns:
        gradient_x, gradient_y, gradient_magnitude
    """
    if image.ndim == 3: # Convert to grayscale
        gray = np.dot(image, [0.2989, 0.5870, 0.1140])
    else:
        gray = image.astype('float32')

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    sobel_y  = sobel_x.T

    gradient_x = convolve2d(gray, sobel_x, padding='replicate')
    gradient_y = convolve2d(gray, sobel_y, padding='replicate')
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    return gradient_x, gradient_y, gradient_magnitude

if __name__ == '__main__':
    test_image = np.array([
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9]
    ], dtype='float32')

    box_kernel = np.ones((3,3))/9

    result = correlated2d(test_image, box_kernel)
    print("Input:\n", test_image)
    print("\nFiltered:\n", result)


    #test gaussian
    kernel_size, sigma_value = 5, 1.0
    gauss_kernel = gaussian_kernel(size=kernel_size, sigma=sigma_value)
    print(f"\nGaussian Kernel {kernel_size}x{kernel_size}, sigma = {sigma_value}\n")
    print(gauss_kernel)
    print('Sum of kernel elements:', np.sum(gauss_kernel))

    blurred = gaussian_blur(test_image, kernel_dim=kernel_size, sigma=sigma_value)
    print("\nBlurred image:\n", blurred)

    # Create a sharper test pattern
    sharp_img = np.zeros((100, 100))
    sharp_img[30:70, 30:70] = 1.0  # White square
    sharp_img[45:55, 45:55] = 0.5  # Gray center
    blurred_img = gaussian_blur(sharp_img, kernel_size=15, sigma=3.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(sharp_img, cmap='gray')
    axes[0].set_title('Original (Sharp Edges)')
    axes[1].imshow(blurred_img, cmap='gray')
    axes[1].set_title('Gaussian Blur (σ=3)')
    plt.savefig('gaussian_test.png')
    print("Saved gaussian_test.png")

    # Test Sobel on diagonal edge
    edge_img = np.tril(np.ones(shape=(50,50)), k=-1)  # Diagonal edge
    gx, gy, mag = sobel_edges(edge_img)

    print("\nSobel test:")
    print("Gradient X range:", gx.min(), "to", gx.max())
    print("Gradient Y range:", gy.min(), "to", gy.max())
    print("Magnitude range:", mag.min(), "to", mag.max())

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(edge_img, cmap='gray')
    axes[0, 0].set_title('Original (Diagonal Edge)')

    axes[0, 1].imshow(gx, cmap='RdBu', vmin=-4, vmax=4)
    axes[0, 1].set_title(f'Gradient X\n[{gx.min():.1f}, {gx.max():.1f}]')

    axes[1, 0].imshow(gy, cmap='RdBu', vmin=-4, vmax=4)
    axes[1, 0].set_title(f'Gradient Y\n[{gy.min():.1f}, {gy.max():.1f}]')

    axes[1, 1].imshow(mag, cmap='hot')
    axes[1, 1].set_title(f'Magnitude\n[{mag.min():.1f}, {mag.max():.1f}]')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('sobel_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved: sobel_visualization.png")
    plt.show()