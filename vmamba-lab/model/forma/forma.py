"""ForMa: Forensic Network based on Vision Mamba.

This module implements the ForMa model for image tampering localization.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import VSSEncoder, MultiScaleVSSEncoder
from .decoder import LightweightDecoder
from .noise_module import NoiseAssistedModule, NoiseGuidedAttention


class ForMa(nn.Module):
    """ForMa: Forensic Network based on Vision Mamba.

    A lightweight and effective network for image tampering localization
    using Visual State Space Model backbone.

    Components:
    - VSS Encoder: VMamba-based multi-scale feature extractor
    - Noise Module: SRM + learnable noise extraction
    - Lightweight Decoder: Pixel shuffle based upsampling
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize ForMa.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()

        model_config = config.get("model", {})
        encoder_config = model_config.get("encoder", {})
        decoder_config = model_config.get("decoder", {})
        noise_config = model_config.get("noise_module", {})

        # Encoder
        backbone_type = encoder_config.get("type", "vmamba_small")
        self.encoder = VSSEncoder(
            backbone_type=backbone_type,
            pretrained=encoder_config.get("pretrained", True),
            drop_path_rate=encoder_config.get("drop_path_rate", 0.3),
            out_indices=encoder_config.get("out_indices", [0, 1, 2, 3]),
        )

        encoder_channels = list(self.encoder.out_channels)

        # Noise-assisted module
        self.use_noise = noise_config.get("enabled", True)
        if self.use_noise:
            noise_type = noise_config.get("type", "srm")
            self.noise_module = NoiseAssistedModule(
                in_channels=3,
                out_channels=64,
                use_srm=(noise_type == "srm" or noise_type == "both"),
                use_learnable=(noise_type == "learnable" or noise_type == "both"),
            )

            # Noise-guided attention for decoder
            self.noise_attention = nn.ModuleList([
                NoiseGuidedAttention(ch, self.noise_module.out_channels)
                for ch in encoder_channels
            ])

        # Decoder
        decoder_channels = decoder_config.get("channels", [256, 128, 64, 32])
        upsample_mode = decoder_config.get("upsample_mode", "pixel_shuffle")

        self.decoder = LightweightDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            upsample_mode=upsample_mode,
            num_classes=model_config.get("output", {}).get("num_classes", 1),
        )

        # Output activation
        output_activation = model_config.get("output", {}).get("activation", "sigmoid")
        if output_activation == "sigmoid":
            self.output_act = nn.Sigmoid()
        elif output_activation == "softmax":
            self.output_act = nn.Softmax(dim=1)
        else:
            self.output_act = nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
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
                - mask: Predicted segmentation mask (B, 1, H, W)
                - logits: Raw logits before activation
                - features: (optional) Intermediate features
        """
        B, C, H, W = x.shape

        # Extract encoder features
        encoder_features = self.encoder(x)

        # Apply noise-guided attention if enabled
        if self.use_noise:
            noise_feat = self.noise_module(x)
            attended_features = []
            for feat, attn_module in zip(encoder_features, self.noise_attention):
                attended = attn_module(feat, noise_feat)
                attended_features.append(attended)
            encoder_features = attended_features

        # Decode
        logits = self.decoder(encoder_features, target_size=(H, W))

        # Apply activation
        mask = self.output_act(logits)

        output = {
            "mask": mask,
            "logits": logits,
        }

        if return_features:
            output["encoder_features"] = encoder_features
            if self.use_noise:
                output["noise_features"] = noise_feat

        return output

    def get_prediction(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get binary prediction mask and probability map.

        Args:
            x: Input image tensor (B, C, H, W)
            threshold: Threshold for binary mask

        Returns:
            Tuple of (binary_mask, probability_map)
        """
        output = self.forward(x)
        prob_map = output["mask"]
        binary_mask = (prob_map > threshold).float()
        return binary_mask, prob_map


class ForMaWithClassification(nn.Module):
    """ForMa with additional image-level classification head.

    Provides both localization mask and image-level tampering prediction.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize ForMa with classification.

        Args:
            config: Model configuration
        """
        super().__init__()

        # Base ForMa model
        self.forma = ForMa(config)

        # Classification head
        encoder_channels = list(self.forma.encoder.out_channels)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)
            return_features: Return intermediate features

        Returns:
            Dictionary with mask, logits, and class predictions
        """
        B, C, H, W = x.shape

        # Get encoder features
        encoder_features = self.forma.encoder(x)

        # Apply noise attention if enabled
        if self.forma.use_noise:
            noise_feat = self.forma.noise_module(x)
            attended_features = []
            for feat, attn_module in zip(encoder_features, self.forma.noise_attention):
                attended = attn_module(feat, noise_feat)
                attended_features.append(attended)
            encoder_features = attended_features

        # Decode for mask
        logits = self.forma.decoder(encoder_features, target_size=(H, W))
        mask = self.forma.output_act(logits)

        # Classify
        class_logits = self.classifier(encoder_features[-1])
        class_probs = F.softmax(class_logits, dim=-1)

        output = {
            "mask": mask,
            "logits": logits,
            "class_logits": class_logits,
            "class_probs": class_probs,
        }

        if return_features:
            output["encoder_features"] = encoder_features

        return output


def create_forma(config: Dict[str, Any]) -> ForMa:
    """Create ForMa model from config.

    Args:
        config: Model configuration

    Returns:
        ForMa model instance
    """
    return ForMa(config)
