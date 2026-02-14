"""
Resnet-18 implementation

Reference: "Deep Residual Learning for Image Recognition"
           He et al., 2015 (https://arxiv.org/abs/1512.03385)
"""

import torch
import torch.nn as nn
from typing import List, Optional

class BasicBlock(nn.Module):
    """
    Residual block form ResNet-18 and ResNet-34

    2 3x3 conv layers with a skip connection:
    output = F(x) + x
    where F(x) = Conv -> BN -> ReLU -> Conv BN

    Skip connection allows gradients to flow directly through the network, solving the vanishing gradients problem
    """