"""
Unit tests for computer-vision-foundations core modules.

Covers: classical CV (filters, edge detection), ResNet, ViT, SimCLR.
All tests run on CPU with synthetic data — no downloads, no GPU required.

Run from the project root:
    pytest computer-vision-foundations/code/tests/test_core.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

# Reproducible random data across all tests
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Path setup — each module directory added so imports resolve without
# installing packages. Order matters: more specific paths first.
# ---------------------------------------------------------------------------
_CODE = Path(__file__).parent.parent
for _subdir in ("classical_cv", "pytorch_cnn", "vision_transformers", "self_supervised_learning"):
    sys.path.insert(0, str(_CODE / _subdir))


# ---------------------------------------------------------------------------
# Classical CV — filters
# ---------------------------------------------------------------------------


class TestCorrelated2d:
    def test_output_shape_preserved(self):
        from filters import correlated2d

        image = np.random.rand(64, 64).astype(np.float32)
        kernel = np.ones((3, 3), dtype=np.float32) / 9
        out = correlated2d(image, kernel)
        assert out.shape == image.shape

    def test_multichannel_shape_preserved(self):
        from filters import correlated2d

        image = np.random.rand(64, 64, 3).astype(np.float32)
        kernel = np.ones((3, 3), dtype=np.float32) / 9
        out = correlated2d(image, kernel)
        assert out.shape == image.shape

    def test_identity_kernel(self):
        from filters import correlated2d

        image = np.random.rand(32, 32).astype(np.float32)
        kernel = np.zeros((3, 3), dtype=np.float32)
        kernel[1, 1] = 1.0
        out = correlated2d(image, kernel)
        np.testing.assert_allclose(out, image, atol=1e-5)


class TestGaussianKernel:
    def test_shape(self):
        from filters import gaussian_kernel

        k = gaussian_kernel(5, sigma=1.0)
        assert k.shape == (5, 5)

    def test_sums_to_one(self):
        from filters import gaussian_kernel

        k = gaussian_kernel(5, sigma=1.0)
        assert abs(k.sum() - 1.0) < 1e-5

    def test_symmetric(self):
        from filters import gaussian_kernel

        k = gaussian_kernel(5, sigma=1.0)
        np.testing.assert_allclose(k, k.T, atol=1e-6)


class TestGaussianBlur:
    def test_spatial_dims_unchanged(self):
        from filters import gaussian_blur

        image = np.random.rand(64, 64).astype(np.float32)
        out = gaussian_blur(image, kernel_dim=5, sigma=1.0)
        assert out.shape == image.shape

    def test_output_finite(self):
        from filters import gaussian_blur

        image = np.random.rand(64, 64).astype(np.float32)
        out = gaussian_blur(image)
        assert np.isfinite(out).all()


class TestSobelEdges:
    def test_return_structure(self):
        from filters import sobel_edges

        image = np.random.rand(64, 64).astype(np.float32)
        gx, gy, mag = sobel_edges(image)
        assert gx.shape == image.shape
        assert gy.shape == image.shape
        assert mag.shape == image.shape

    def test_magnitude_nonnegative(self):
        from filters import sobel_edges

        image = np.random.rand(64, 64).astype(np.float32)
        _, _, mag = sobel_edges(image)
        assert (mag >= 0).all()

    def test_uniform_image_zero_gradient(self):
        """A zero-valued flat image has no intensity steps — magnitude must be zero."""
        from filters import sobel_edges

        # np.zeros avoids the border artefact that np.ones*128 produces when
        # zero-padded: padding zeros onto zeros creates no step.
        image = np.zeros((64, 64), dtype=np.float32)
        _, _, mag = sobel_edges(image)
        np.testing.assert_allclose(mag, 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Classical CV — edge detection
# ---------------------------------------------------------------------------


class TestCannyEdgeDetector:
    def test_output_shape(self):
        from edge_detection import canny_edge_detector

        image = np.random.rand(64, 64).astype(np.float32)
        out = canny_edge_detector(image)
        assert out.shape == image.shape

    def test_output_binary(self):
        """Canny output should contain only 0s and 255s (or 0s and 1s)."""
        from edge_detection import canny_edge_detector

        image = (np.random.rand(64, 64) * 255).astype(np.float32)
        out = canny_edge_detector(image)
        unique = np.unique(out)
        assert set(unique.tolist()).issubset({0, 1, 255}), f"Unexpected values: {unique}"

    def test_uniform_image_no_edges(self):
        from edge_detection import canny_edge_detector

        image = np.zeros((64, 64), dtype=np.float32)
        out = canny_edge_detector(image)
        assert out.sum() == 0


class TestNonMaxSuppression:
    def test_output_shape(self):
        from edge_detection import non_max_suppression

        magnitude = np.random.rand(64, 64).astype(np.float32)
        direction = np.random.rand(64, 64).astype(np.float32) * 360
        out = non_max_suppression(magnitude, direction)
        assert out.shape == magnitude.shape


class TestDoubleThreshold:
    def test_output_shape(self):
        from edge_detection import double_threshold

        image = np.random.rand(64, 64).astype(np.float32)
        out, strong, weak = double_threshold(image)
        assert out.shape == image.shape

    def test_strong_greater_than_weak(self):
        from edge_detection import double_threshold

        image = np.random.rand(64, 64).astype(np.float32)
        _, strong, weak = double_threshold(image)
        assert strong > weak


# ---------------------------------------------------------------------------
# ResNet
# ---------------------------------------------------------------------------


class TestResNet18:
    @pytest.fixture(scope="class")
    def model(self):
        from resnet import resnet18

        return resnet18(num_classes=10).eval()

    def test_output_shape(self, model):
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_output_finite(self, model):
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all().item()

    def test_has_skip_connections(self, model):
        """BasicBlock must be present — confirms residual connections are wired."""
        from resnet import BasicBlock

        assert any(isinstance(m, BasicBlock) for m in model.modules()), "ResNet18 should contain BasicBlock layers"

    def test_batch_size_invariant(self, model):
        """Output batch dimension must match input batch dimension."""
        for bs in (1, 4, 8):
            x = torch.randn(bs, 3, 32, 32)
            with torch.no_grad():
                out = model(x)
            assert out.shape[0] == bs


class TestResNet18Channels:
    def test_custom_in_channels(self):
        from resnet import resnet18

        model = resnet18(num_classes=5, in_channels=1).eval()
        x = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 5)


# ---------------------------------------------------------------------------
# Vision Transformer
# ---------------------------------------------------------------------------


class TestPatchEmbedding:
    def test_output_shape(self):
        from vit import PatchEmbedding

        embed = PatchEmbedding(img_size=32, patch_size=4, in_channels=3, embed_dim=192)
        x = torch.randn(2, 3, 32, 32)
        out = embed(x)
        # 64 patches + 1 CLS token prepended by PatchEmbedding
        expected_tokens = (32 // 4) ** 2 + 1
        assert out.shape == (2, expected_tokens, 192)

    def test_different_patch_sizes(self):
        from vit import PatchEmbedding

        for patch_size in (4, 8):
            embed = PatchEmbedding(img_size=32, patch_size=patch_size, in_channels=3, embed_dim=64)
            x = torch.randn(1, 3, 32, 32)
            out = embed(x)
            # num_patches + 1 CLS token
            expected_tokens = (32 // patch_size) ** 2 + 1
            assert out.shape[1] == expected_tokens


class TestVitTiny:
    @pytest.fixture(scope="class")
    def model(self):
        from vit import vit_tiny

        return vit_tiny(num_classes=10, img_size=32, patch_size=4).eval()

    def test_output_shape(self, model):
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_output_finite(self, model):
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all().item()

    def test_batch_size_invariant(self, model):
        for bs in (1, 4):
            x = torch.randn(bs, 3, 32, 32)
            with torch.no_grad():
                out = model(x)
            assert out.shape[0] == bs


# ---------------------------------------------------------------------------
# SimCLR
# ---------------------------------------------------------------------------


class TestNTXentLoss:
    def test_output_is_scalar(self):
        from simclr import NTXentLoss

        loss_fn = NTXentLoss(temperature=0.5)
        z_i = torch.randn(8, 128)
        z_j = torch.randn(8, 128)
        loss = loss_fn(z_i, z_j)
        assert loss.shape == ()

    def test_output_finite(self):
        from simclr import NTXentLoss

        loss_fn = NTXentLoss(temperature=0.5)
        z_i = torch.randn(16, 128)
        z_j = torch.randn(16, 128)
        loss = loss_fn(z_i, z_j)
        assert torch.isfinite(loss).item()

    def test_temperature_effect(self):
        """Lower temperature sharpens the distribution — loss must differ across temperatures."""
        from simclr import NTXentLoss

        z_i = torch.randn(16, 128)
        z_j = torch.randn(16, 128)
        loss_low = NTXentLoss(temperature=0.1)(z_i, z_j)
        loss_high = NTXentLoss(temperature=1.0)(z_i, z_j)
        assert not torch.isclose(loss_low, loss_high)


class TestProjectionHead:
    def test_output_shape(self):
        from simclr import ProjectionHead

        head = ProjectionHead(in_dim=512, hidden_dim=512, out_dim=128)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4, 128)

    def test_custom_dims(self):
        from simclr import ProjectionHead

        head = ProjectionHead(in_dim=256, hidden_dim=128, out_dim=64)
        x = torch.randn(4, 256)
        out = head(x)
        assert out.shape == (4, 64)


class TestSimCLRModel:
    @pytest.fixture(scope="class")
    def model(self):
        from simclr import SimCLR

        return SimCLR(base_encoder="resnet18", out_dim=128).eval()

    def test_projected_output_shape(self, model):
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = model(x, project=True)
        assert out.shape == (4, 128)

    def test_encoder_output_shape(self, model):
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = model(x, project=False)
        assert out.shape == (4, 512)

    def test_output_finite(self, model):
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all().item()


class TestSimCLRAugmentation:
    def _make_image(self) -> PILImage.Image:
        return PILImage.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))

    def test_returns_two_tensors(self):
        from simclr import SimCLRAugmentation

        v1, v2 = SimCLRAugmentation(image_size=32)(self._make_image())
        assert isinstance(v1, torch.Tensor)
        assert isinstance(v2, torch.Tensor)

    def test_two_views_have_correct_shape(self):
        from simclr import SimCLRAugmentation

        v1, v2 = SimCLRAugmentation(image_size=32)(self._make_image())
        assert v1.shape == (3, 32, 32)
        assert v2.shape == (3, 32, 32)

    def test_two_views_differ(self):
        """Stochastic augmentation must produce different views with high probability."""
        from simclr import SimCLRAugmentation

        aug = SimCLRAugmentation(image_size=32)
        image = self._make_image()
        # 5 independent draws — probability of all being identical is negligible
        assert any(
            not torch.equal(*aug(image)) for _ in range(5)
        ), "Augmentation produced identical views 5 times in a row"
