"""Hierarchical Wavelet Feature Extraction Branch (HWFEB) for WMamba.

This module implements the wavelet-based feature extraction
branch that captures multi-scale frequency information.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet_utils import DWT2D, WaveletTransform
from .dcconv import DCConv, AdaptiveContourConv


class WaveletFeatureBlock(nn.Module):
    """Feature extraction block for wavelet subbands."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_dcconv: bool = True,
    ):
        """Initialize wavelet feature block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_dcconv: Use DCConv for contour detection
        """
        super().__init__()

        if use_dcconv:
            self.conv1 = AdaptiveContourConv(in_channels, out_channels)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Skip connection
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, out_channels, H, W)
        """
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + identity


class HighFrequencyModule(nn.Module):
    """Module for processing high-frequency wavelet components.

    Designed to detect manipulation artifacts that often
    appear in high-frequency regions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """Initialize high-frequency module.

        Args:
            in_channels: Number of input channels (3x for LH, HL, HH)
            out_channels: Number of output channels
        """
        super().__init__()

        # Process each high-frequency subband
        self.lh_conv = nn.Sequential(
            nn.Conv2d(in_channels // 3, out_channels // 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 3),
            nn.ReLU(inplace=True),
        )
        self.hl_conv = nn.Sequential(
            nn.Conv2d(in_channels // 3, out_channels // 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 3),
            nn.ReLU(inplace=True),
        )
        self.hh_conv = nn.Sequential(
            nn.Conv2d(in_channels // 3, out_channels // 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 3),
            nn.ReLU(inplace=True),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid(),
        )

    def forward(
        self,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            lh: LH subband (B, C, H, W)
            hl: HL subband (B, C, H, W)
            hh: HH subband (B, C, H, W)

        Returns:
            Output tensor (B, out_channels, H, W)
        """
        lh_feat = self.lh_conv(lh)
        hl_feat = self.hl_conv(hl)
        hh_feat = self.hh_conv(hh)

        # Concatenate
        combined = torch.cat([lh_feat, hl_feat, hh_feat], dim=1)

        # Fusion
        fused = self.fusion(combined)

        # Channel attention
        attn = self.channel_attn(fused).unsqueeze(-1).unsqueeze(-1)
        output = fused * attn

        return output


class HWFEB(nn.Module):
    """Hierarchical Wavelet Feature Extraction Branch.

    Extracts multi-scale features using wavelet decomposition
    and processes them with contour-aware convolutions.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: List[int] = [64, 128, 256],
        wavelet: str = "db1",
        levels: int = 3,
        use_dcconv: bool = True,
    ):
        """Initialize HWFEB.

        Args:
            in_channels: Number of input channels
            out_channels: Output channels at each level
            wavelet: Wavelet type
            levels: Number of decomposition levels
            use_dcconv: Use DCConv for contour detection
        """
        super().__init__()

        self.levels = levels
        self.wavelet = wavelet

        # Wavelet transform
        self.dwt = DWT2D(wavelet)

        # Feature extraction blocks for each level
        self.ll_blocks = nn.ModuleList()
        self.hf_blocks = nn.ModuleList()

        current_channels = in_channels
        for i in range(levels):
            # Low-frequency (LL) processing
            self.ll_blocks.append(
                WaveletFeatureBlock(
                    current_channels,
                    out_channels[i],
                    use_dcconv=use_dcconv,
                )
            )

            # High-frequency processing
            self.hf_blocks.append(
                HighFrequencyModule(
                    current_channels * 3,  # LH, HL, HH
                    out_channels[i],
                )
            )

            current_channels = out_channels[i]

        # Multi-scale fusion
        self.scale_fusion = nn.ModuleList()
        for i in range(levels - 1):
            self.scale_fusion.append(
                nn.Sequential(
                    nn.Conv2d(out_channels[i] * 2, out_channels[i], kernel_size=1),
                    nn.BatchNorm2d(out_channels[i]),
                    nn.ReLU(inplace=True),
                )
            )

        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(sum(out_channels), out_channels[-1], kernel_size=1),
            nn.BatchNorm2d(out_channels[-1]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Tuple of:
                - Final fused features (B, out_channels[-1], H', W')
                - List of features at each level
        """
        features = []
        current = x

        for i in range(self.levels):
            # Apply DWT
            ll, lh, hl, hh = self.dwt(current)

            # Process low-frequency
            ll_feat = self.ll_blocks[i](ll)

            # Process high-frequency
            hf_feat = self.hf_blocks[i](lh, hl, hh)

            # Combine LL and HF features
            combined = ll_feat + hf_feat
            features.append(combined)

            # Use LL for next level
            current = ll

        # Multi-scale fusion (bottom-up)
        fused_features = [features[-1]]
        for i in range(self.levels - 2, -1, -1):
            # Upsample and fuse
            upsampled = F.interpolate(
                fused_features[-1],
                size=features[i].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = torch.cat([features[i], upsampled], dim=1)
            fused = self.scale_fusion[i](fused)
            fused_features.append(fused)

        fused_features = fused_features[::-1]  # Reverse to coarse-to-fine order

        # Final output
        # Resize all to same size and concatenate
        target_size = features[0].shape[2:]
        resized_features = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            resized_features.append(feat)

        concat_features = torch.cat(resized_features, dim=1)
        output = self.output_proj(concat_features)

        return output, features
