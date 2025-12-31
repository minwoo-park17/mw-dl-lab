"""WMamba: Wavelet-based Mamba for Face Forgery Detection.

This module implements the WMamba model that combines wavelet-based
feature extraction with VMamba for face forgery detection.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import VMamba, create_vmamba
from .hwfeb import HWFEB


class FeatureFusion(nn.Module):
    """Feature fusion module for combining wavelet and VMamba features."""

    def __init__(
        self,
        wavelet_dim: int,
        vmamba_dims: List[int],
        out_dim: int,
    ):
        """Initialize feature fusion.

        Args:
            wavelet_dim: Dimension of wavelet features
            vmamba_dims: Dimensions of VMamba features at each stage
            out_dim: Output dimension
        """
        super().__init__()

        # Project wavelet features
        self.wavelet_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(wavelet_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        # Project VMamba features (use last stage)
        self.vmamba_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(vmamba_dims[-1], out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        # Attention-based fusion
        self.attention = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Linear(out_dim * 2, out_dim)

    def forward(
        self,
        wavelet_feat: torch.Tensor,
        vmamba_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            wavelet_feat: Wavelet features (B, C_w, H, W)
            vmamba_feat: VMamba features (B, C_v, H', W')

        Returns:
            Fused features (B, out_dim)
        """
        # Project features
        w_feat = self.wavelet_proj(wavelet_feat)
        v_feat = self.vmamba_proj(vmamba_feat)

        # Concatenate
        concat = torch.cat([w_feat, v_feat], dim=-1)

        # Attention weights
        attn = self.attention(concat)

        # Weighted fusion
        fused = torch.cat([w_feat * attn, v_feat * (1 - attn)], dim=-1)
        output = self.out_proj(fused)

        return output


class WMamba(nn.Module):
    """WMamba: Wavelet-based Mamba for Face Forgery Detection.

    Combines:
    - HWFEB: Hierarchical Wavelet Feature Extraction Branch
    - VMamba: Visual State Space Model backbone
    - DCConv: Dynamic Contour Convolution

    for robust face forgery detection.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize WMamba.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()

        model_config = config.get("model", {})
        wavelet_config = model_config.get("wavelet", {})
        backbone_config = model_config.get("backbone", {})
        hwfeb_config = model_config.get("hwfeb", {})
        classifier_config = model_config.get("classifier", {})

        # Wavelet parameters
        self.wavelet_type = wavelet_config.get("type", "db1")
        self.wavelet_levels = wavelet_config.get("levels", 3)

        # HWFEB: Hierarchical Wavelet Feature Extraction Branch
        hwfeb_channels = hwfeb_config.get("channels", [64, 128, 256])
        self.hwfeb = HWFEB(
            in_channels=3,
            out_channels=hwfeb_channels,
            wavelet=self.wavelet_type,
            levels=self.wavelet_levels,
            use_dcconv=hwfeb_config.get("use_dcconv", True),
        )

        # VMamba backbone
        backbone_type = backbone_config.get("type", "vmamba_tiny")
        self.vmamba = create_vmamba(
            backbone_type,
            pretrained=backbone_config.get("pretrained", False),
            num_classes=0,  # Remove classifier head
            drop_path_rate=backbone_config.get("drop_path_rate", 0.2),
            out_indices=[0, 1, 2, 3],
        )

        # Get VMamba output dimensions
        vmamba_dims = self.vmamba.dims

        # Feature fusion
        hidden_dim = classifier_config.get("hidden_dim", 512)
        self.fusion = FeatureFusion(
            wavelet_dim=hwfeb_channels[-1],
            vmamba_dims=vmamba_dims,
            out_dim=hidden_dim,
        )

        # Classifier head
        num_classes = classifier_config.get("num_classes", 2)
        dropout = classifier_config.get("dropout", 0.5)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input image tensor (B, C, H, W)
            return_features: Return intermediate features

        Returns:
            Dictionary containing:
                - logits: Classification logits (B, num_classes)
                - probs: Classification probabilities (B, num_classes)
                - features: (optional) Intermediate features
        """
        # Extract wavelet features
        wavelet_feat, wavelet_levels = self.hwfeb(x)

        # Extract VMamba features
        vmamba_features = self.vmamba.forward_features(x)
        vmamba_feat = vmamba_features[-1]  # Use last stage

        # Fuse features
        fused_feat = self.fusion(wavelet_feat, vmamba_feat)

        # Classification
        logits = self.classifier(fused_feat)
        probs = F.softmax(logits, dim=-1)

        output = {
            "logits": logits,
            "probs": probs,
        }

        if return_features:
            output["wavelet_features"] = wavelet_feat
            output["vmamba_features"] = vmamba_feat
            output["fused_features"] = fused_feat

        return output

    def get_prediction(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prediction labels and scores.

        Args:
            x: Input image tensor (B, C, H, W)

        Returns:
            Tuple of (predictions, scores)
        """
        output = self.forward(x)
        probs = output["probs"]
        predictions = probs.argmax(dim=-1)
        scores = probs[:, 1]  # Probability of fake class
        return predictions, scores


def create_wmamba(config: Dict[str, Any]) -> WMamba:
    """Create WMamba model from config.

    Args:
        config: Model configuration

    Returns:
        WMamba model instance
    """
    return WMamba(config)
