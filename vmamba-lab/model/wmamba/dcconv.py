"""Dynamic Contour Convolution (DCConv) for WMamba.

This module implements deformable convolution variants designed
to capture facial contours and fine-grained boundaries.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class DCConv(nn.Module):
    """Dynamic Contour Convolution.

    Uses deformable convolution with specially designed kernels
    to adaptively model slender facial contours.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        num_deform_groups: int = 1,
    ):
        """Initialize DCConv.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            dilation: Convolution dilation
            groups: Number of groups for grouped convolution
            bias: Use bias
            num_deform_groups: Number of deformable groups
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_deform_groups = num_deform_groups

        # Number of points in kernel
        self.num_points = kernel_size * kernel_size

        # Offset prediction: 2 * num_points (x, y offsets)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels // 4,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels // 4,
                2 * self.num_points * num_deform_groups,
                kernel_size=3,
                padding=1,
            ),
        )

        # Modulation mask prediction
        self.mask_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels // 4,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels // 4,
                self.num_points * num_deform_groups,
                kernel_size=3,
                padding=1,
            ),
        )

        # Main convolution weight
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # Initialize offset conv to output near-zero offsets
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)

        # Initialize mask conv to output ones
        nn.init.zeros_(self.mask_conv[-1].weight)
        nn.init.constant_(self.mask_conv[-1].bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, out_channels, H', W')
        """
        # Predict offsets
        offset = self.offset_conv(x)

        # Predict modulation mask
        mask = self.mask_conv(x)
        mask = torch.sigmoid(mask)

        # Apply deformable convolution
        output = deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )

        return output


class ContourConv(nn.Module):
    """Contour-aware Convolution.

    Designed specifically for detecting facial contour artifacts.
    Uses multiple orientations to capture edges in different directions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_orientations: int = 4,
    ):
        """Initialize ContourConv.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_orientations: Number of edge orientations
        """
        super().__init__()

        self.num_orientations = num_orientations

        # Edge detection filters for different orientations
        self.edge_convs = nn.ModuleList()
        for i in range(num_orientations):
            conv = nn.Conv2d(
                in_channels,
                out_channels // num_orientations,
                kernel_size=(1, 3) if i % 2 == 0 else (3, 1),
                padding=(0, 1) if i % 2 == 0 else (1, 0),
            )
            self.edge_convs.append(conv)

        # Diagonal edge filters
        self.diag_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // num_orientations, kernel_size=3, padding=1)
            for _ in range(num_orientations // 2)
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, out_channels, H, W)
        """
        # Apply edge convolutions
        edge_features = []
        for conv in self.edge_convs:
            edge_features.append(conv(x))

        # Apply diagonal convolutions
        for conv in self.diag_convs:
            edge_features.append(conv(x))

        # Concatenate all edge features
        edge_concat = torch.cat(edge_features, dim=1)

        # Fusion
        output = self.fusion(edge_concat)

        return output


class AdaptiveContourConv(nn.Module):
    """Adaptive Contour Convolution.

    Combines DCConv with contour-aware processing for
    detecting manipulation artifacts along facial boundaries.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_dcconv: bool = True,
    ):
        """Initialize AdaptiveContourConv.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            use_dcconv: Use deformable convolution
        """
        super().__init__()

        self.use_dcconv = use_dcconv

        if use_dcconv:
            self.main_conv = DCConv(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        else:
            self.main_conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )

        # Contour enhancement branch
        self.contour_branch = nn.Sequential(
            # Sobel-like edge detection
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1),
        )

        # Spatial attention for contours
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(out_channels + out_channels // 2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels + out_channels // 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, out_channels, H, W)
        """
        # Main branch
        main_feat = self.main_conv(x)

        # Contour branch
        contour_feat = self.contour_branch(x)

        # Concatenate features
        combined = torch.cat([main_feat, contour_feat], dim=1)

        # Spatial attention
        attn = self.spatial_attn(combined)
        combined = combined * attn

        # Fusion
        output = self.fusion(combined)

        return output
