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

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        stride: for first conv
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super(BasicBlock, self).__init__()

        # First conv: possible downsample (stride=2) and change channel count
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False) # no bias needed, BatchNorm handles it
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second conv: no downsampling, keeps channel count
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection projection. Identity (no parameters needed) unless:
        # 1. stride>1 (spatial dims changes)
        # 2. Channel count change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), # 1x1 conv to match dims
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection
        Args:
            x: Input tensor (batch, in_channels, H, W)
        Returns:
             output: (batch, out_channels, H', W')
        """

        # Main path: F(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection: Identity or projection
        # Then add: output = F(x) + x
        out += self.shortcut(x)

        out = self.relu(out)

        return out

    def __repr__(self) -> str:
        return(
            f"BasicBlock(in={self.conv1.in_channels}, "
            f"out={self.conv1.out_channels}, "
            f"stride={self.conv1.stride[0]}, "
            f"shortcut={'projection' if len(self.shortcut) > 0 else 'Identity'}"
        )

# =============================================================================
# RESNET - Full Architecture
# =============================================================================

class ResNet(nn.Module):
    """
    ResNet architecture (Focus on ResNet-18).

    Can build ResNet-18, 34, 50, 101, 152 - depends on block config.
    Args:
        block: Block class to use (BasicBlock for ResNet-18/34)
        layers: List of blocks per layer [2, 2, 2, 2] for ResNet-18
        num_classes: Number of output classes
        in_channels: Numer of input channels (3=RGB, 1=Grayscale)
    """

    def __init__(self, block: type, layers: List[int], num_classes: int=1000, in_channels: int=3) -> None:
        super(ResNet, self).__init__()
        self.current_channels = 64  # Tracks channels as layers are built

        # Inital conv 7x7 to quickly reduce spatial dims
        # Input: (batch, 3, 224, 224) -> Output: (batch, 64, 112, 112)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # MaxPool: (batch, 64, 112, 112) -> (batch, 64, 56, 56)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers - each doubles channels (except for layer1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)  # (batch, 64, 56, 56)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # (batch, 128, 28, 28)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # (batch, 256, 14, 14)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # (batch, 512, 7, 7)

        # Global average pooling: (batch, 512, 7, 7) -> (batch, 512, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier: (batch, 512) -> (batch, num_classes)
        self.fc = nn.Linear(512, num_classes)

        # Weight initialization
        self._initialize_weights()

    def _make_layer(self, block: type, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Build one residual layer (group of blocks).
        First block may downsample (stride=2), the rest keeps the same spatial size (stride=1)

        Args:
            block: Block class (BasicBlock)
            out_channels: Output channels for this layer
            num_blocks: Number of blocks for this layer
            stride: Stride for first block

        Returns:
            Sequential container of blocks
        """

        layers = [block(self.current_channels, out_channels, stride)]

        # First block: downsampling and channel change
        self.current_channels = out_channels # Update for next block

        # Remaining blocks: same channels, no downsampling
        for _ in range(1, num_blocks):
            layers.append(block(self.current_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """
        Initializes weighting using He initializations
        - Conv layers: He normal (good for ReLU)
        - BatchNorm: weight=1, bias=0 (identity at start)
        - Linear: standard normal with small std
        """

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight,0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through full ResNet.

        Args:
            x: Input tensor (batch, channels, H, W)
        Returns:
            logits: (batch, num_classes)
        """

        # Initial feature extraction
        x = self.conv1(x) # -> (batch, 64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # -> (batch, 64, 56, 56)

        # Residual layers (batch, 64, 56, 56) -> ... -> (batch, 512, 7, 7)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling + classify
        x = self.avgpool(x)     # -> (batch, 512, 1, 1)
        x = torch.flatten(x, 1)     # -> (batch, 512)
        x = self.fc(x)          # -> (batch, num_classes)

        return x

    def count_parameters(self) -> int:
        """Return total number of trainable paramerters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================================================
# ResNet Versions
# ============================================================================================================

def resnet18(num_classes: int=1000, in_channels: int=3) -> ResNet:
    """
    ResNet-18: 18 weight layers
    config: [2, 2, 2, 2] BasicBlocks
    ~11.7m parameters
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)

def resnet34(num_classes: int=1000, in_channels: int=3) -> ResNet:
    """
    ResNet-34: 34 weight layers
    Config: [3, 4, 6, 3] BasicBlock
    ~21.8M parameters
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)



# ============================================================================================================
# Test
# ============================================================================================================

if __name__ == '__main__':
    print("="*60)
    print("ResNet-18 architecture test")
    print("="*60)

    model = resnet18(num_classes=10)
    print("\nFull architecture")
    print(model)

    total = model.count_parameters()
    print(f"\nTotal trainable parameters: {total:,}")
    print(f"Expected (ImageNet):           11,689,512")


    print("\nTesting forward pass...")
    x = torch.randn(4, 3, 224, 224) # 4 RGB 224x224 images
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == (4, 10), f"Expected (4,10), got {out.shape}"
    print("Output shape correct!")

    # Testing MNIST dimensions (grayscale, 28x28)
    print("\nTesting with MNIST")
    model_mnist = resnet18(num_classes=10, in_channels=1)
    x_mnist = torch.randn(4, 1, 28, 28)
    out_mnist = model_mnist(x_mnist)
    print(f"Input shape: {x_mnist.shape}, Output shape: {out_mnist.shape}")
    assert out_mnist.shape == (4, 10), f"Expected (4,10), got {out_mnist.shape}"
    print("MNIST forward pass correct!")

    # Test BasicBlock
    print("\nTesting BasicBlock")
    block_identity = BasicBlock(64, 64, stride=1)
    x_block = torch.randn(2, 64, 56, 56)
    out_block = block_identity(x_block)
    assert out_block.shape == x_block.shape, f"in block shape: {x_block.shape}, output shape: {out_block.shape}"
    print("Identity block correct")

    # Projection shortcut (different channels, stride=2)
    block_projction = BasicBlock(64, 128, stride=2)
    x_block_projction = torch.randn(2, 64, 56, 56)
    out_block_projction = block_projction(x_block_projction)
    assert out_block_projction.shape == (2, 128, 28, 28)
    print(f"Projection block correct: {x_block_projction.shape} -> {out_block_projction.shape}")

    # Layer-by-layer shape trace
    print("\nShape trace through ResNet-18:")
    model_trace = resnet18(num_classes=10)
    x = torch.randn(1, 3, 224, 224)

    x = model_trace.conv1(x)
    print(f"After conv1:   {x.shape}")
    x = model_trace.bn1(x)
    x = model_trace.relu(x)
    x = model_trace.maxpool(x)
    print(f"After maxpool: {x.shape}")
    x = model_trace.layer1(x)
    print(f"After layer1:  {x.shape}")
    x = model_trace.layer2(x)
    print(f"After layer2:  {x.shape}")
    x = model_trace.layer3(x)
    print(f"After layer3:  {x.shape}")
    x = model_trace.layer4(x)
    print(f"After layer4:  {x.shape}")
    x = model_trace.avgpool(x)
    print(f"After avgpool: {x.shape}")
    x = torch.flatten(x, 1)
    print(f"After flatten: {x.shape}")
    x = model_trace.fc(x)
    print(f"After fc:      {x.shape}")

    print("\n" + "=" * 60)
    print("✅ All tests passed! ResNet-18 implementation correct!")
    print("=" * 60)
