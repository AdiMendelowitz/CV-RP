import numpy as np
import matplotlib.pyplot as plt
from filters import gaussian_blur, sobel_edges
from pathlib import Path

output_dir = Path(__file__).parent / 'outputs'
output_dir.mkdir(parents=True, exist_ok=True)

def create_test_patterns():
    """Create synthetic test images"""

    checker = np.zeros((200, 200))
    checker[::20, ::20] = 1
    checker = np.kron(checker, np.ones((10, 10)))

    gradient = np.linspace(0, 1, 200)
    gradient = np.tile(gradient, (200, 1))

    y, x = np.ogrid[0:200, :200]
    circle = ((x-100)**2 + (y-100)**2 < 60**2).astype('float32')

    return {'checkerboard': checker, 'gradient': gradient, 'circle': circle}

def test_gaussian():
    """Test gaussian blur on patterns"""
    patterns = create_test_patterns()

    for name, img in patterns.items():
        blurred = gaussian_blur(image=img, kernel_dim=15, sigma=3.0)

        fix, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'{name.capitalize()} - Original: {name}')
        axes[0].axis('off')

        axes[1].imshow(blurred, cmap='gray')
        axes[1].set_title(f'{name.capitalize()} - Blurred')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / f'output_{name}.png')
        print(f"Saved output_{name}.png")
        plt.close()

def test_sobel():
    """Test Sobel edge detection on patterns"""
    patterns = create_test_patterns()

    for name, img in patterns.items():
        grad_x, grad_y, grad_mag = sobel_edges(image=img)

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title(f'{name.capitalize()} - Original')

        axes[0, 1].imshow(grad_mag, cmap='hot')
        axes[0, 1].set_title('Edge Magnitude')

        axes[1, 0].imshow(grad_x, cmap='RdBu')
        axes[1, 0].set_title('Gradient X')

        axes[1, 1].imshow(grad_y, cmap='RdBu')
        axes[1, 1].set_title('Gradient Y')

        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / f'output_sobel_{name}.png')
        print(f"Saved output_sobel_{name}.png")
        plt.close()

if __name__ == "__main__":
    print("Testing Gaussian Blur\n")
    test_gaussian()

    print("\nTesting Sobel Edge Detection\n")
    test_sobel()

    print("Tests completed.")
