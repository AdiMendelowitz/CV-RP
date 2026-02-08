"""
Unit tests for classical CV implementations.

Run with: pytest test_unit.py -v
"""

import numpy as np
import pytest
from filters import gaussian_blur, sobel_edges, convolve2d
from edge_detection import canny_edge_detector, non_max_suppression, double_threshold, hysteresis
from transformers import get_affine_transform, get_perspective_transform, warp_affine, warp_perspective, rotate, resize


# ============================================================================
# FILTERS TESTS
# ============================================================================

class TestGaussianBlur:
    """Test Gaussian blur implementation"""

    def test_output_shape(self):
        """Output should have same shape as input"""
        img = np.random.rand(100, 100).astype('float32')
        result = gaussian_blur(img, kernel_dim=5, sigma=1.0)
        assert result.shape == img.shape

    def test_smooths_noise(self):
        """Gaussian blur should reduce variance"""
        np.random.seed(42)
        noisy = np.random.rand(50, 50).astype('float32')
        smoothed = gaussian_blur(noisy, kernel_dim=5, sigma=1.0)
        assert smoothed.var() < noisy.var()

    def test_preserves_constant(self):
        """Constant image should stay constant"""
        img = np.ones((50, 50), dtype='float32') * 0.5
        result = gaussian_blur(img, kernel_dim=5, sigma=1.0)
        np.testing.assert_allclose(result, 0.5, rtol=1e-3)

    def test_symmetric(self):
        """Gaussian kernel should be symmetric"""
        img = np.zeros((50, 50), dtype='float32')
        img[25, 25] = 1.0  # Impulse
        result = gaussian_blur(img, kernel_dim=5, sigma=1.0)
        # Center should be symmetric
        assert np.abs(result[24, 25] - result[26, 25]) < 1e-6
        assert np.abs(result[25, 24] - result[25, 26]) < 1e-6


class TestSobelEdges:
    """Test Sobel edge detection"""

    def test_output_shape(self):
        """Output shapes should match input"""
        img = np.random.rand(100, 100).astype('float32')
        gx, gy, mag = sobel_edges(img)
        assert gx.shape == img.shape
        assert gy.shape == img.shape
        assert mag.shape == img.shape

    def test_vertical_edge(self):
        """Vertical edge should have strong Gx"""
        img = np.zeros((50, 50), dtype='float32')
        img[:, :25] = 0.0
        img[:, 25:] = 1.0

        gx, gy, mag = sobel_edges(img)

        # Gx should be strong at edge
        assert np.max(np.abs(gx[:, 23:27])) > 0.5
        # Gy should be weak (no vertical variation)
        assert np.max(np.abs(gy)) < 0.2

    def test_horizontal_edge(self):
        """Horizontal edge should have strong Gy"""
        img = np.zeros((50, 50), dtype='float32')
        img[:25, :] = 0.0
        img[25:, :] = 1.0

        gx, gy, mag = sobel_edges(img)

        # Gy should be strong at edge
        assert np.max(np.abs(gy[23:27, :])) > 0.5
        # Gx should be weak
        assert np.max(np.abs(gx)) < 0.2

    def test_magnitude_positive(self):
        """Magnitude should always be non-negative"""
        img = np.random.rand(50, 50).astype('float32')
        _, _, mag = sobel_edges(img)
        assert np.all(mag >= 0)


class TestConvolve2D:
    """Test 2D convolution"""

    def test_identity_kernel(self):
        """Identity kernel should return original"""
        img = np.random.rand(50, 50).astype('float32')
        kernel = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]], dtype='float32')
        result = convolve2d(img, kernel, padding='replicate')
        np.testing.assert_allclose(result, img, rtol=1e-5)

    def test_output_shape(self):
        """Output should preserve input shape"""
        img = np.random.rand(100, 100).astype('float32')
        kernel = np.ones((5, 5), dtype='float32')
        result = convolve2d(img, kernel, padding='replicate')
        assert result.shape == img.shape


# ============================================================================
# EDGE DETECTION TESTS
# ============================================================================

class TestNonMaxSuppression:
    """Test non-maximum suppression"""

    def test_output_shape(self):
        """Output should match input shape"""
        mag = np.random.rand(100, 100).astype('float32')
        direction = np.random.rand(100, 100).astype('float32')
        result = non_max_suppression(mag, direction)
        assert result.shape == mag.shape

    def test_suppresses_values(self):
        """Should reduce number of non-zero pixels"""
        mag = np.random.rand(50, 50).astype('float32')
        direction = np.random.rand(50, 50).astype('float32')
        result = non_max_suppression(mag, direction)
        assert np.count_nonzero(result) <= np.count_nonzero(mag)

    def test_non_negative(self):
        """Output should be non-negative"""
        mag = np.random.rand(50, 50).astype('float32')
        direction = np.random.rand(50, 50).astype('float32')
        result = non_max_suppression(mag, direction)
        assert np.all(result >= 0)


class TestDoubleThreshold:
    """Test double thresholding"""

    def test_output_shape(self):
        """Output should match input shape"""
        img = np.random.rand(100, 100).astype('float32')
        result, strong, weak = double_threshold(img, 0.05, 0.15)
        assert result.shape == img.shape

    def test_three_values(self):
        """Output should have only 3 unique values"""
        img = np.random.rand(50, 50).astype('float32')
        result, strong, weak = double_threshold(img, 0.05, 0.15)
        unique = np.unique(result)
        assert len(unique) <= 3
        assert 0 in unique or weak in unique or strong in unique

    def test_weak_strong_values(self):
        """Should return correct weak/strong values"""
        img = np.random.rand(50, 50).astype('float32')
        result, strong, weak = double_threshold(img, 0.05, 0.15)
        assert weak == 75
        assert strong == 255


class TestHysteresis:
    """Test edge tracking by hysteresis"""

    def test_output_shape(self):
        """Output should match input shape"""
        img = np.zeros((50, 50), dtype='uint8')
        result = hysteresis(img, strong=255, weak=75)
        assert result.shape == img.shape

    def test_keeps_strong(self):
        """Strong edges should be preserved"""
        img = np.zeros((50, 50), dtype='uint8')
        img[25, 25] = 255  # Strong edge
        result = hysteresis(img, strong=255, weak=75)
        assert result[25, 25] == 255

    def test_removes_isolated_weak(self):
        """Isolated weak edges should be removed"""
        img = np.zeros((50, 50), dtype='uint8')
        img[25, 25] = 75  # Isolated weak
        result = hysteresis(img, strong=255, weak=75)
        assert result[25, 25] == 0

    def test_connects_weak_to_strong(self):
        """Weak edges connected to strong should survive"""
        img = np.zeros((50, 50), dtype='uint8')
        img[25, 25] = 255  # Strong
        img[25, 26] = 75  # Weak neighbor
        result = hysteresis(img, strong=255, weak=75)
        assert result[25, 25] == 255
        assert result[25, 26] == 255


class TestCannyEdgeDetector:
    """Test full Canny pipeline"""

    def test_output_shape(self):
        """Output should match input shape"""
        img = np.random.rand(100, 100).astype('float32')
        result = canny_edge_detector(img)
        assert result.shape == img.shape

    def test_binary_output(self):
        """Output should be binary (0 or 255)"""
        img = np.random.rand(50, 50).astype('float32')
        result = canny_edge_detector(img)
        unique = np.unique(result)
        assert all(v in [0, 255] for v in unique)

    def test_circle_detection(self):
        """Should detect circle edge"""
        img = np.zeros((100, 100), dtype='float32')
        y, x = np.ogrid[:100, :100]
        circle = ((x - 50) ** 2 + (y - 50) ** 2 < 30 ** 2).astype('float32')

        edges = canny_edge_detector(circle, low_ratio=0.1, high_ratio=0.3)

        # Should find some edges
        assert np.sum(edges > 0) > 100


# ============================================================================
# TRANSFORMS TESTS
# ============================================================================

class TestAffineTransform:
    """Test affine transformation"""

    def test_identity_transform(self):
        """Identity transform should preserve points"""
        src = np.array([[0, 0], [100, 0], [0, 100]], dtype='float32')
        dst = src.copy()

        M = get_affine_transform(src, dst)

        # Should be identity matrix
        expected = np.array([[1, 0, 0],
                             [0, 1, 0]], dtype='float32')
        np.testing.assert_allclose(M, expected, atol=1e-5)

    def test_translation(self):
        """Pure translation"""
        src = np.array([[0, 0], [100, 0], [0, 100]], dtype='float32')
        dst = src + [10, 20]  # Translate by (10, 20)

        M = get_affine_transform(src, dst)

        # Should be translation matrix
        assert abs(M[0, 0] - 1) < 1e-5
        assert abs(M[1, 1] - 1) < 1e-5
        assert abs(M[0, 2] - 10) < 1e-5
        assert abs(M[1, 2] - 20) < 1e-5

    def test_needs_three_points(self):
        """Should require exactly 3 points"""
        src = np.array([[0, 0], [100, 0]], dtype='float32')  # Only 2 points
        dst = src.copy()

        with pytest.raises(ValueError):
            get_affine_transform(src, dst)


class TestPerspectiveTransform:
    """Test perspective transformation"""

    def test_identity_transform(self):
        """Identity transform should preserve points"""
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype='float32')
        dst = src.copy()

        H = get_perspective_transform(src, dst)

        # Should be close to identity
        expected = np.eye(3, dtype='float32')
        np.testing.assert_allclose(H, expected, atol=1e-4)

    def test_needs_four_points(self):
        """Should require exactly 4 points"""
        src = np.array([[0, 0], [100, 0], [100, 100]], dtype='float32')  # Only 3
        dst = src.copy()

        with pytest.raises(ValueError):
            get_perspective_transform(src, dst)

    def test_normalization(self):
        """H[2,2] should be 1 after normalization"""
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype='float32')
        dst = np.array([[10, 10], [90, 10], [95, 95], [5, 95]], dtype='float32')

        H = get_perspective_transform(src, dst)

        assert abs(H[2, 2] - 1.0) < 1e-6


class TestWarpAffine:
    """Test affine warping"""

    def test_identity_warp(self):
        """Identity matrix should preserve image"""
        img = np.random.rand(50, 50).astype('float32')
        M = np.array([[1, 0, 0],
                      [0, 1, 0]], dtype='float32')

        result = warp_affine(img, M, (50, 50))

        np.testing.assert_allclose(result, img, atol=1e-2)

    def test_output_shape(self):
        """Should produce requested output shape"""
        img = np.random.rand(50, 50).astype('float32')
        M = np.eye(3, dtype='float32')[:2]

        result = warp_affine(img, M, (100, 75))

        assert result.shape == (100, 75)


class TestRotate:
    """Test rotation function"""

    def test_zero_rotation(self):
        """0° rotation should preserve image"""
        img = np.random.rand(50, 50).astype('float32')
        result = rotate(img, angle_degrees=0, center=(100, 100))
        np.testing.assert_allclose(result, img, atol=1e-2)

    def test_output_shape(self):
        """Should preserve image shape"""
        img = np.random.rand(100, 75).astype('float32')
        result = rotate(img, 45, center=(img.shape[1]//2, img.shape[0]//2))
        assert result.shape == img.shape

    def test_90_degree_rotation(self):
        """90° rotation should work correctly"""
        img = np.zeros((50, 50), dtype='float32')
        img[10:20, 10:15] = 1.0  # Small rectangle

        rotated = rotate(img, 90, center=(25, 25))

        # Should have rotated the rectangle
        assert np.sum(rotated > 0.5) > 0


class TestResize:
    """Test resize function"""

    def test_upscale(self):
        """2x upscale should double dimensions"""
        img = np.random.rand(50, 50).astype('float32')
        result = resize(img, scale=2.0)
        assert result.shape == (100, 100)

    def test_downscale(self):
        """0.5x scale should halve dimensions"""
        img = np.random.rand(100, 100).astype('float32')
        result = resize(img, scale=0.5)
        assert result.shape == (50, 50)

    def test_identity_scale(self):
        """1x scale should preserve dimensions"""
        img = np.random.rand(50, 50).astype('float32')
        result = resize(img, scale=1.0)
        assert result.shape == img.shape


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])