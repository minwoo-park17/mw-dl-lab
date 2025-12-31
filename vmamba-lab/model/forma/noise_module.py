"""Noise-assisted Module for ForMa.

This module implements noise extraction and processing
to enhance manipulation detection sensitivity.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRMFilter(nn.Module):
    """Steganalysis Rich Model (SRM) filters.

    Fixed high-pass filters designed to extract noise residuals
    that reveal manipulation artifacts.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize SRM filters.

        Args:
            in_channels: Number of input channels
        """
        super().__init__()

        # SRM filter kernels (30 filters from SRM)
        filters = self._get_srm_filters()

        # Register as buffer (not trainable)
        # Shape: (30, 1, 5, 5)
        self.register_buffer("filters", torch.tensor(filters, dtype=torch.float32))

        self.in_channels = in_channels
        self.num_filters = 30

    def _get_srm_filters(self) -> np.ndarray:
        """Get SRM filter kernels.

        Returns:
            Array of shape (30, 1, 5, 5) containing SRM filters
        """
        filters = []

        # 1st order filters
        # Horizontal edge
        f1 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.float32)
        filters.append(f1)

        # Vertical edge
        f2 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.float32)
        filters.append(f2)

        # Diagonal edges
        f3 = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.float32)
        filters.append(f3)

        f4 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.float32)
        filters.append(f4)

        # 2nd order filters
        f5 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.float32)
        filters.append(f5)

        f6 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, -2, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.float32)
        filters.append(f6)

        # 3rd order filters (SQUARE)
        f7 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
            [0, -1, 4, -1, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.float32) / 4
        filters.append(f7)

        # EDGE 3x3
        f8 = np.array([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.float32) / 4
        filters.append(f8)

        # SQUARE 5x5
        f9 = np.array([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ], dtype=np.float32) / 12
        filters.append(f9)

        # Additional high-pass filters
        # Laplacian variants
        f10 = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, -2, 0, 0],
            [1, -2, 4, -2, 1],
            [0, 0, -2, 0, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.float32) / 8
        filters.append(f10)

        # Generate more filters by rotating existing ones
        base_filters = filters[:10]
        for bf in base_filters:
            # 90 degree rotation
            filters.append(np.rot90(bf))
            # 180 degree rotation
            filters.append(np.rot90(bf, 2))

        # Ensure we have exactly 30 filters
        while len(filters) < 30:
            filters.append(np.zeros((5, 5), dtype=np.float32))

        filters = np.array(filters[:30])
        return filters.reshape(30, 1, 5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SRM filters to extract noise residuals.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Noise residuals (B, 30*C, H, W)
        """
        B, C, H, W = x.shape

        # Apply filters to each channel
        outputs = []
        for c in range(C):
            channel = x[:, c : c + 1]  # (B, 1, H, W)
            filtered = F.conv2d(channel, self.filters, padding=2)  # (B, 30, H, W)
            outputs.append(filtered)

        # Concatenate all channels
        output = torch.cat(outputs, dim=1)  # (B, 30*C, H, W)

        return output


class LearnableNoiseExtractor(nn.Module):
    """Learnable noise extraction module.

    Learns to extract manipulation-relevant noise patterns.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
    ):
        """Initialize learnable noise extractor.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels per kernel size
            kernel_sizes: Kernel sizes for multi-scale extraction
        """
        super().__init__()

        self.branches = nn.ModuleList()

        for ks in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=ks,
                    padding=ks // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.branches.append(branch)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(kernel_sizes), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract learnable noise features.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Noise features (B, out_channels, H, W)
        """
        branch_outputs = [branch(x) for branch in self.branches]
        concat = torch.cat(branch_outputs, dim=1)
        output = self.fusion(concat)
        return output


class NoiseAssistedModule(nn.Module):
    """Noise-assisted module for tampering detection.

    Combines SRM filters with learnable noise extraction
    to enhance manipulation detection sensitivity.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        use_srm: bool = True,
        use_learnable: bool = True,
    ):
        """Initialize noise-assisted module.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            use_srm: Use SRM filters
            use_learnable: Use learnable noise extractor
        """
        super().__init__()

        self.use_srm = use_srm
        self.use_learnable = use_learnable

        modules_channels = 0

        if use_srm:
            self.srm = SRMFilter(in_channels)
            srm_channels = 30 * in_channels
            self.srm_proj = nn.Sequential(
                nn.Conv2d(srm_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            modules_channels += out_channels

        if use_learnable:
            self.learnable = LearnableNoiseExtractor(
                in_channels,
                out_channels,
            )
            modules_channels += out_channels

        # Final fusion if both are used
        if use_srm and use_learnable:
            self.fusion = nn.Sequential(
                nn.Conv2d(modules_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.fusion = nn.Identity()

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract noise features.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Noise features (B, out_channels, H, W)
        """
        features = []

        if self.use_srm:
            srm_feat = self.srm(x)
            srm_feat = self.srm_proj(srm_feat)
            features.append(srm_feat)

        if self.use_learnable:
            learn_feat = self.learnable(x)
            features.append(learn_feat)

        if len(features) > 1:
            combined = torch.cat(features, dim=1)
            output = self.fusion(combined)
        else:
            output = features[0]

        return output


class NoiseGuidedAttention(nn.Module):
    """Noise-guided attention for decoder features.

    Uses noise information to guide attention
    toward manipulated regions.
    """

    def __init__(
        self,
        feat_channels: int,
        noise_channels: int,
    ):
        """Initialize noise-guided attention.

        Args:
            feat_channels: Number of feature channels
            noise_channels: Number of noise channels
        """
        super().__init__()

        self.noise_proj = nn.Conv2d(noise_channels, feat_channels, kernel_size=1)

        self.attention = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feat: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Apply noise-guided attention.

        Args:
            feat: Feature tensor (B, C_f, H, W)
            noise: Noise tensor (B, C_n, H', W')

        Returns:
            Attended features (B, C_f, H, W)
        """
        # Resize noise to match features
        if noise.shape[2:] != feat.shape[2:]:
            noise = F.interpolate(noise, size=feat.shape[2:], mode="bilinear", align_corners=False)

        # Project noise
        noise_proj = self.noise_proj(noise)

        # Compute attention
        combined = torch.cat([feat, noise_proj], dim=1)
        attn = self.attention(combined)

        # Apply attention
        output = feat * attn + feat

        return output
