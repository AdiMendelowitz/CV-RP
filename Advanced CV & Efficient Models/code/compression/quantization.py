"""
Post training quantization (dynamic & static) for a pre-trained ResNet-18.

Measures model size (MB), median single-sample inference latency (ms) and top-1 acuracy on CIFAR-10 before and after
each quantization scheme.

Both schemes run on CPU. PyTorch's quantized kernels are CPU-only for static quantization, so models and inputs are
explicitly kept on CPU regardless of what device the checkpoint was trained on.


Note on dynamic quantization and Conv2d:
    PyTorch does not support dynamic quantization for nn.Conv2d,  passing it to quantize_dynamic is silently ignored.
    Only nn.Linear layers are quantized in the dynamic scheme.
    Static quantization covers all Conv2d layers via the observer/calibration pipeline.
    Reference: https://pytorch.org/docs/stable/quantization.html
"""

import argparse
import copy
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "computer-vision-foundations" / "code" / "pytorch_cnn"))
from resnet import resnet18 as custom_resnet18

_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"

# ----------------------------------------------------------------------------------------------------------------------
# Data Loaders
# ----------------------------------------------------------------------------------------------------------------------

def get_cifar10_loaders(data_dir: str = str(_DATA_ROOT), batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """
    Calibration loader uses the training split with no shuffling so that consecutive batches are representative of the
    full distribution. The test loader is used for accuracy evaluation.

    Returns:
        (calibration_loader, test_loader) for CIFAR-10.
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    calibration_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])







