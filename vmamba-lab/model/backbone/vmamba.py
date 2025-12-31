"""VMamba: Visual State Space Model backbone.

This module implements the VMamba architecture for visual recognition.
"""

import math
from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_

from .vss_block import VSSBlock


class PatchEmbed2D(nn.Module):
    """2D Patch Embedding with overlapping patches."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 96,
        patch_size: int = 4,
        norm_layer: Optional[Callable] = None,
    ):
        """Initialize patch embedding.

        Args:
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            patch_size: Patch size
            norm_layer: Normalization layer
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, H', W', embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.permute(0, 2, 3, 1)  # (B, H', W', embed_dim)
        x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    """Patch merging layer for downsampling."""

    def __init__(
        self,
        dim: int,
        out_dim: Optional[int] = None,
        norm_layer: Callable = nn.LayerNorm,
    ):
        """Initialize patch merging.

        Args:
            dim: Input dimension
            out_dim: Output dimension (default: 2 * dim)
            norm_layer: Normalization layer
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, H, W, C)

        Returns:
            Output tensor (B, H/2, W/2, out_dim)
        """
        B, H, W, C = x.shape

        # Pad if needed
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        # Merge 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class VSSLayer(nn.Module):
    """A layer of VSS Blocks with optional downsampling."""

    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        drop_path: List[float] = [0.0],
        norm_layer: Callable = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ):
        """Initialize VSS layer.

        Args:
            dim: Input dimension
            depth: Number of VSS blocks
            d_state: State space dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            drop_path: Drop path rates for each block
            norm_layer: Normalization layer
            downsample: Downsampling layer
            use_checkpoint: Use gradient checkpointing
        """
        super().__init__()
        self.dim = dim
        self.depth = depth

        # VSS blocks
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if i < len(drop_path) else 0.0,
                norm_layer=norm_layer,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)
        ])

        # Downsampling
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (B, H, W, C)

        Returns:
            Tuple of (output, output_before_downsample)
        """
        for block in self.blocks:
            x = block(x)

        x_before_down = x

        if self.downsample is not None:
            x = self.downsample(x)

        return x, x_before_down


class VMamba(nn.Module):
    """VMamba: Visual State Space Model.

    A hierarchical vision backbone based on state space models.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        depths: List[int] = [2, 2, 9, 2],
        dims: List[int] = [96, 192, 384, 768],
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        patch_size: int = 4,
        norm_layer: Callable = nn.LayerNorm,
        use_checkpoint: bool = False,
        out_indices: Optional[List[int]] = None,
        pretrained: Optional[str] = None,
    ):
        """Initialize VMamba.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            depths: Number of blocks at each stage
            dims: Dimensions at each stage
            d_state: State space dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            drop_rate: Dropout rate
            drop_path_rate: Drop path rate
            patch_size: Patch size for embedding
            norm_layer: Normalization layer
            use_checkpoint: Use gradient checkpointing
            out_indices: Indices of stages to output features
            pretrained: Path to pretrained weights
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.dims = dims
        self.out_indices = out_indices or [0, 1, 2, 3]

        # Patch embedding
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=dims[0],
            patch_size=patch_size,
            norm_layer=norm_layer,
        )

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(self.num_stages):
            stage = VSSLayer(
                dim=dims[i],
                depth=depths[i],
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=dpr[cur : cur + depths[i]],
                norm_layer=norm_layer,
                downsample=PatchMerging2D(dims[i], dims[i + 1], norm_layer)
                if i < self.num_stages - 1
                else None,
                use_checkpoint=use_checkpoint,
            )
            self.stages.append(stage)
            cur += depths[i]

        # Output norms for each stage
        self.out_norms = nn.ModuleList([
            norm_layer(dims[i]) for i in self.out_indices
        ])

        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity(),
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Load pretrained
        if pretrained:
            self.load_pretrained(pretrained)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_pretrained(self, path: str):
        """Load pretrained weights."""
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

        # Remove head weights if num_classes doesn't match
        if "head.3.weight" in state_dict:
            if state_dict["head.3.weight"].shape[0] != self.num_classes:
                del state_dict["head.3.weight"]
                del state_dict["head.3.bias"]

        self.load_state_dict(state_dict, strict=False)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            List of feature tensors at each stage
        """
        x = self.patch_embed(x)  # (B, H', W', C)

        features = []
        for i, stage in enumerate(self.stages):
            x, x_before_down = stage(x)
            if i in self.out_indices:
                idx = self.out_indices.index(i)
                feat = self.out_norms[idx](x_before_down)
                feat = feat.permute(0, 3, 1, 2)  # (B, C, H, W)
                features.append(feat)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Class logits (B, num_classes)
        """
        x = self.patch_embed(x)

        for stage in self.stages:
            x, _ = stage(x)

        # Global average pooling
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = x.flatten(2)  # (B, C, H*W)
        x = self.head(x)

        return x


# Model variants
def vmamba_tiny(pretrained: bool = False, **kwargs) -> VMamba:
    """VMamba-Tiny model."""
    model = VMamba(
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        **kwargs,
    )
    return model


def vmamba_small(pretrained: bool = False, **kwargs) -> VMamba:
    """VMamba-Small model."""
    model = VMamba(
        depths=[2, 2, 27, 2],
        dims=[96, 192, 384, 768],
        **kwargs,
    )
    return model


def vmamba_base(pretrained: bool = False, **kwargs) -> VMamba:
    """VMamba-Base model."""
    model = VMamba(
        depths=[2, 2, 27, 2],
        dims=[128, 256, 512, 1024],
        **kwargs,
    )
    return model


def create_vmamba(
    model_name: str = "vmamba_tiny",
    pretrained: bool = False,
    **kwargs,
) -> VMamba:
    """Create VMamba model.

    Args:
        model_name: Model variant name
        pretrained: Load pretrained weights
        **kwargs: Additional arguments

    Returns:
        VMamba model
    """
    model_dict = {
        "vmamba_tiny": vmamba_tiny,
        "vmamba_small": vmamba_small,
        "vmamba_base": vmamba_base,
    }

    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}")

    return model_dict[model_name](pretrained=pretrained, **kwargs)
