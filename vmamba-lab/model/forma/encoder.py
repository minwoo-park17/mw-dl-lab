"""VSS Encoder for ForMa.

This module implements the Visual State Space encoder
based on VMamba for image tampering localization.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import VMamba, create_vmamba


class VSSEncoder(nn.Module):
    """Visual State Space Encoder.

    Multi-scale feature extractor based on VMamba backbone.
    """

    def __init__(
        self,
        backbone_type: str = "vmamba_small",
        pretrained: bool = True,
        drop_path_rate: float = 0.3,
        out_indices: List[int] = [0, 1, 2, 3],
        freeze_stages: int = -1,
    ):
        """Initialize VSS Encoder.

        Args:
            backbone_type: VMamba variant to use
            pretrained: Use pretrained weights
            drop_path_rate: Drop path rate
            out_indices: Indices of stages to output
            freeze_stages: Number of stages to freeze (-1 for none)
        """
        super().__init__()

        self.out_indices = out_indices
        self.freeze_stages = freeze_stages

        # Create VMamba backbone
        self.backbone = create_vmamba(
            backbone_type,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate,
            out_indices=out_indices,
        )

        # Get output dimensions
        self.out_channels = self.backbone.dims

        # Freeze stages if specified
        if freeze_stages >= 0:
            self._freeze_stages(freeze_stages)

    def _freeze_stages(self, num_stages: int):
        """Freeze specified number of stages.

        Args:
            num_stages: Number of stages to freeze
        """
        # Freeze patch embedding
        if num_stages >= 0:
            self.backbone.patch_embed.eval()
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = False

        # Freeze stages
        for i in range(min(num_stages, len(self.backbone.stages))):
            stage = self.backbone.stages[i]
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            List of multi-scale features, each (B, C_i, H_i, W_i)
        """
        features = self.backbone.forward_features(x)
        return features


class MultiScaleVSSEncoder(nn.Module):
    """Multi-scale VSS Encoder with additional feature processing.

    Adds feature refinement and channel alignment for decoder.
    """

    def __init__(
        self,
        backbone_type: str = "vmamba_small",
        pretrained: bool = True,
        out_channels: List[int] = [64, 128, 256, 512],
        drop_path_rate: float = 0.3,
    ):
        """Initialize multi-scale VSS encoder.

        Args:
            backbone_type: VMamba variant
            pretrained: Use pretrained weights
            out_channels: Desired output channels at each scale
            drop_path_rate: Drop path rate
        """
        super().__init__()

        # Base encoder
        self.encoder = VSSEncoder(
            backbone_type=backbone_type,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
        )

        # Channel alignment layers
        backbone_channels = self.encoder.out_channels
        self.channel_align = nn.ModuleList()

        for i, (in_ch, out_ch) in enumerate(zip(backbone_channels, out_channels)):
            if in_ch != out_ch:
                self.channel_align.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.channel_align.append(nn.Identity())

        self.out_channels = out_channels

        # Feature refinement
        self.refine_blocks = nn.ModuleList([
            FeatureRefinementBlock(ch) for ch in out_channels
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            List of refined multi-scale features
        """
        # Get backbone features
        features = self.encoder(x)

        # Align channels and refine
        refined_features = []
        for i, (feat, align, refine) in enumerate(
            zip(features, self.channel_align, self.refine_blocks)
        ):
            feat = align(feat)
            feat = refine(feat)
            refined_features.append(feat)

        return refined_features


class FeatureRefinementBlock(nn.Module):
    """Feature refinement block with residual connection."""

    def __init__(self, channels: int):
        """Initialize refinement block.

        Args:
            channels: Number of channels
        """
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

        # Squeeze-and-excitation attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W)
        """
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        # SE attention
        attn = self.se(out)
        out = out * attn

        out = out + identity
        out = self.relu(out)

        return out
