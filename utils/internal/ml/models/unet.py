try:
    from local_setup import local_setup
    local_setup()
except ModuleNotFoundError as e:
    from utils.internal.sbas.local_setup import local_setup
    local_setup()

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from utils.internal.log.logger import get_logger

log = get_logger()


class DoubleConv(nn.Module):
    """
    Double convolution block: (Conv2d -> BatchNorm -> ReLU) x 2

    This is the basic building block used in both encoder and decoder.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block: MaxPool2d -> DoubleConv
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block: TransposeConv2d -> Concatenate with skip -> DoubleConv
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Input from previous layer (decoder)
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)

        # Pad x1 to match x2 size if needed (handles odd dimensions)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AtmosphericCorrectionUNet(nn.Module):
    """
    U-Net for atmospheric phase correction.

    Architecture:
    - 5-level encoder with skip connections
    - Bottleneck layer
    - 5-level decoder with upsampling
    - Center cropping (input 256×256 → output 128×128) to accumulate context

    Input: [Batch, 14, 256, 256]
        - SEVIRI t1: 6 channels (WV_062, WV_073, IR_097, IR_087, IR_108, IR_120)
        - SEVIRI t2: 6 channels
        - Coherence: 2 channels

    Output: [Batch, 1, 128, 128]
        - Atmospheric phase correction
    """

    def __init__(
        self,
        in_channels: int = 14,
        out_channels: int = 1,
        init_features: int = 64,
        input_size: int = 256,
        output_size: int = 128
    ):
        """
        Initialize UNet model.

        Args:
            in_channels: Number of input channels (14 for SEVIRI + coherence)
            out_channels: Number of output channels (1 for phase correction)
            init_features: Number of features in first layer
            input_size: Input image size (256)
            output_size: Output image size after center crop (128)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.input_size = input_size
        self.output_size = output_size

        # Calculate features for each level
        features = init_features

        # Initial convolution (no downsampling)
        self.inc = DoubleConv(in_channels, features)

        # Encoder (downsampling path)
        self.down1 = Down(features, features * 2)      # 64 -> 128
        self.down2 = Down(features * 2, features * 4)  # 128 -> 256
        self.down3 = Down(features * 4, features * 8)  # 256 -> 512
        self.down4 = Down(features * 8, features * 16) # 512 -> 1024

        # Bottleneck
        self.bottleneck = Down(features * 16, features * 16)  # 1024 -> 1024

        # Decoder (upsampling path)
        self.up1 = Up(features * 16, features * 8)   # 1024 -> 512
        self.up2 = Up(features * 8, features * 4)    # 512 -> 256
        self.up3 = Up(features * 4, features * 2)    # 256 -> 128
        self.up4 = Up(features * 2, features)        # 128 -> 64

        # Final 1×1 convolution
        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)

        log.info(
            f"UNet initialized: {in_channels} -> {out_channels} channels, "
            f"init_features={init_features}, {input_size}×{input_size} -> {output_size}×{output_size}"
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor [Batch, 14, 256, 256]

        Returns:
            Output tensor [Batch, 1, 128, 128]
        """
        # Encoder with skip connections
        x1 = self.inc(x)      # [B, 64, 256, 256]
        x2 = self.down1(x1)   # [B, 128, 128, 128]
        x3 = self.down2(x2)   # [B, 256, 64, 64]
        x4 = self.down3(x3)   # [B, 512, 32, 32]
        x5 = self.down4(x4)   # [B, 1024, 16, 16]

        # Bottleneck
        x6 = self.bottleneck(x5)  # [B, 1024, 8, 8]

        # Decoder with skip connections
        x = self.up1(x6, x5)  # [B, 512, 16, 16]
        x = self.up2(x, x4)   # [B, 256, 32, 32]
        x = self.up3(x, x3)   # [B, 128, 64, 64]
        x = self.up4(x, x2)   # [B, 64, 128, 128]

        # Final convolution
        x = self.outc(x)      # [B, 1, 128, 128]

        # Center crop to output size if needed
        if x.size(2) != self.output_size or x.size(3) != self.output_size:
            # Calculate crop offsets
            diff_h = (x.size(2) - self.output_size) // 2
            diff_w = (x.size(3) - self.output_size) // 2
            x = x[:, :, diff_h:diff_h+self.output_size, diff_w:diff_w+self.output_size]

        return x

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_device(self):
        """Get device of model parameters."""
        return next(self.parameters()).device


def test_model():
    """
    Test model forward pass with dummy data.
    """
    print("Testing AtmosphericCorrectionUNet...")

    # Create model
    model = AtmosphericCorrectionUNet(
        in_channels=14,
        out_channels=1,
        init_features=64,
        input_size=256,
        output_size=128
    )

    # Print model info
    print(f"Total parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 14, 256, 256)

    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        y = model(x)

    print(f"Output shape: {y.shape}")

    # Verify output shape
    assert y.shape == (batch_size, 1, 128, 128), f"Expected (2, 1, 128, 128), got {y.shape}"

    print("✓ Model test passed!")

    return model


if __name__ == "__main__":
    # Run test when script is executed directly
    test_model()
