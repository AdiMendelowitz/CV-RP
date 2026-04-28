"""
Mask head for Mask R-CNN instance segmentation.

Implementation of the fully convolutional mask branch from Mask R-CNN, He et al.,
ICCV 2017. https://arxiv.org/abs/1703.06870

The head takes the fixed-size RoI feature (14x14) produced by RoI Align and output a per-class binary mask
prediction (28x28)
"""

import torch
import torch.nn as nn

class MaskHead(nn.Module):
    """
    Fully convolutional mask prediction head.

    Architecture:
            4 x (Conv 3x3, ReLU) -> ConvTransposed2d 2x upsample -> Conv 1x1 -> num_classes channels.

    All convolutional weights are initialised with Kaiming normal (fan-out mode), which is the standard
    initialisation for ReLU networks.

    Args:
        in_channels: Number of channels in the input RoI feature map.
        num_classes: Number of object classes, one binary mask per class.
        hidden_channels: Number of channels in the hidden layers, default = 256,
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 256) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=2, stride=2)
        self.upsample_relu = nn.ReLU(inplace=True)

        self.predict = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes all conv weights with Kaiming normal and zero biases."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the mask head on a batch of RoI features.
        Args:
            x: RoI feature maps, shape (B, in_channels, 14, 14)

        Returns:
            Per-class mask logits, shape (B, num_classes, 28, 28) (Raw logit, apply sigmoid for binary probabilities)
        """

        x = self.conv_layers(x)
        x = self.upsample_relu(self.upsample(x))
        return self.predict(x)

if __name__ == "__main__":
    B, in_channels, num_classes = 4, 256, 80
    head = MaskHead(in_channels=in_channels, num_classes=num_classes)
    dummy = torch.zeros(B, in_channels, 14, 14)
    out = head(dummy)
    assert out.shape == (B, num_classes, 28, 28), f"Unexpected output shape: {out.shape}"
    print(f"MaskHead output shape: {out.shape}")
