"""Lightweight Decoder for ForMa.

This module implements a lightweight decoder with pixel shuffle
upsampling for image tampering localization.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShuffleUpsample(nn.Module):
    """Pixel shuffle upsampling block.

    More efficient than transposed convolution while
    avoiding checkerboard artifacts.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
    ):
        """Initialize pixel shuffle upsample.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            scale_factor: Upsampling factor
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (scale_factor ** 2),
            kernel_size=3,
            padding=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C_in, H, W)

        Returns:
            Output tensor (B, C_out, H*scale, W*scale)
        """
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block with skip connection fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        upsample_mode: str = "pixel_shuffle",
    ):
        """Initialize decoder block.

        Args:
            in_channels: Number of input channels (from previous decoder layer)
            skip_channels: Number of skip connection channels (from encoder)
            out_channels: Number of output channels
            upsample_mode: Upsampling method ('pixel_shuffle', 'bilinear', 'nearest')
        """
        super().__init__()

        self.upsample_mode = upsample_mode

        # Upsampling
        if upsample_mode == "pixel_shuffle":
            self.upsample = PixelShuffleUpsample(in_channels, in_channels)
        elif upsample_mode == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )
        else:  # nearest
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )

        # Skip connection projection
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        # Feature projection
        self.feat_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        # Fusion convolution
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor from previous decoder layer (B, C_in, H, W)
            skip: Skip connection from encoder (B, C_skip, H', W')

        Returns:
            Output tensor (B, C_out, H', W')
        """
        # Upsample
        x = self.upsample(x)

        # Handle size mismatch
        if skip is not None and x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        # Project features
        x = self.feat_proj(x)

        # Fuse with skip connection
        if skip is not None:
            skip = self.skip_proj(skip)
            x = torch.cat([x, skip], dim=1)
            x = self.fusion(x)
        else:
            x = self.fusion(torch.cat([x, x], dim=1))

        return x


class LightweightDecoder(nn.Module):
    """Lightweight decoder for image tampering localization.

    Uses pixel shuffle upsampling and efficient feature fusion.
    """

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int] = [256, 128, 64, 32],
        upsample_mode: str = "pixel_shuffle",
        num_classes: int = 1,
    ):
        """Initialize lightweight decoder.

        Args:
            encoder_channels: Channel dimensions from encoder (coarse to fine)
            decoder_channels: Channel dimensions for decoder stages
            upsample_mode: Upsampling method
            num_classes: Number of output classes (1 for binary)
        """
        super().__init__()

        self.num_stages = len(encoder_channels)

        # Reverse encoder channels for bottom-up decoding
        encoder_channels = encoder_channels[::-1]  # Now fine to coarse

        # Initial projection from deepest encoder features
        self.init_proj = nn.Sequential(
            nn.Conv2d(encoder_channels[0], decoder_channels[0], kernel_size=1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(1, self.num_stages):
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=decoder_channels[i - 1],
                    skip_channels=encoder_channels[i],
                    out_channels=decoder_channels[i],
                    upsample_mode=upsample_mode,
                )
            )

        # Final upsampling to original resolution
        self.final_upsample = nn.Sequential(
            PixelShuffleUpsample(decoder_channels[-1], decoder_channels[-1]),
            PixelShuffleUpsample(decoder_channels[-1], decoder_channels[-1]),
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[-1] // 2, num_classes, kernel_size=1),
        )

    def forward(
        self,
        features: List[torch.Tensor],
        target_size: Optional[tuple] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Multi-scale features from encoder [f1, f2, f3, f4]
                     (from finest to coarsest)
            target_size: Target output size (H, W)

        Returns:
            Segmentation mask (B, num_classes, H, W)
        """
        # Reverse features for bottom-up decoding
        features = features[::-1]  # Now coarsest to finest

        # Initial projection
        x = self.init_proj(features[0])

        # Decode with skip connections
        for i, (block, skip) in enumerate(zip(self.decoder_blocks, features[1:])):
            x = block(x, skip)

        # Final upsampling
        x = self.final_upsample(x)

        # Output head
        x = self.output_head(x)

        # Resize to target if specified
        if target_size is not None and x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        return x


class UNetDecoder(nn.Module):
    """UNet-style decoder as alternative to lightweight decoder."""

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int] = [256, 128, 64, 32],
        num_classes: int = 1,
    ):
        """Initialize UNet decoder.

        Args:
            encoder_channels: Channel dimensions from encoder
            decoder_channels: Channel dimensions for decoder
            num_classes: Number of output classes
        """
        super().__init__()

        # Reverse for bottom-up
        encoder_channels = encoder_channels[::-1]

        self.blocks = nn.ModuleList()

        # First block
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(encoder_channels[0], decoder_channels[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(decoder_channels[0]),
                nn.ReLU(inplace=True),
            )
        )

        # Subsequent blocks
        for i in range(1, len(encoder_channels)):
            in_ch = decoder_channels[i - 1] + encoder_channels[i]
            out_ch = decoder_channels[i] if i < len(decoder_channels) else decoder_channels[-1]

            self.blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        decoder_channels[i - 1], decoder_channels[i - 1],
                        kernel_size=2, stride=2
                    ),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )

        # Output head
        self.output = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(
        self,
        features: List[torch.Tensor],
        target_size: Optional[tuple] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        features = features[::-1]

        x = self.blocks[0](features[0])

        for i, (block, skip) in enumerate(zip(self.blocks[1:], features[1:]), 1):
            # Upsample
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            # Concatenate skip
            x = torch.cat([x, skip], dim=1)
            # Process
            x = block[1:](x)  # Skip the transposed conv

        x = self.output(x)

        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        return x
