"""
DIRE (Diffusion Reconstruction Error) model for deepfake detection.

Reference: "DIRE for Diffusion-Generated Image Detection" (ICCV 2023)
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
from timm import create_model

from diffusion_utils import DiffusionReconstructor

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class DIREClassifier(nn.Module):
    """
    Classifier for DIRE features.

    Takes DIRE error maps as input and outputs binary classification.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 1,
        pretrained: bool = True,
        input_channels: int = 3
    ):
        """
        Initialize DIRE classifier.

        Args:
            backbone: Backbone model name (any timm model)
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            input_channels: Number of input channels (3 for RGB DIRE maps)
        """
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes

        # Create backbone
        self.backbone = create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=input_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: DIRE error map [B, 3, H, W]

        Returns:
            logits: Classification logits [B, num_classes]
        """
        return self.backbone(x)


class DIREDetector(nn.Module):
    """
    Full DIRE detector combining diffusion reconstruction and classification.

    Pipeline:
    1. Input image -> DDIM Inversion -> Noisy latent
    2. Noisy latent -> DDIM Reconstruction -> Reconstructed image
    3. DIRE = |Original - Reconstructed|
    4. DIRE -> Classifier -> Prediction
    """

    def __init__(
        self,
        diffusion_model_id: str = "runwayml/stable-diffusion-v1-5",
        classifier_backbone: str = "resnet50",
        num_classes: int = 1,
        ddim_steps: int = 20,
        guidance_scale: float = 1.0,
        classifier_input_size: int = 224,
        device: str = "cuda"
    ):
        """
        Initialize DIRE detector.

        Args:
            diffusion_model_id: Hugging Face model ID
            classifier_backbone: Backbone for classifier
            num_classes: Number of output classes
            ddim_steps: Number of DDIM steps
            guidance_scale: Classifier-free guidance scale
            classifier_input_size: Input size for classifier
            device: Device to run on
        """
        super().__init__()

        self.ddim_steps = ddim_steps
        self.guidance_scale = guidance_scale
        self.classifier_input_size = classifier_input_size
        self.device = device

        # Initialize diffusion reconstructor
        logger.info("Initializing diffusion model...")
        self.reconstructor = DiffusionReconstructor(
            model_id=diffusion_model_id,
            device=device
        )

        # Initialize classifier
        logger.info(f"Initializing classifier: {classifier_backbone}")
        self.classifier = DIREClassifier(
            backbone=classifier_backbone,
            num_classes=num_classes,
            pretrained=True,
            input_channels=3
        )

    def compute_dire(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute DIRE features for input images.

        Args:
            images: Input images [B, 3, H, W] in range [-1, 1]

        Returns:
            dire: DIRE error maps [B, 3, H, W]
        """
        return self.reconstructor.compute_dire(
            images,
            num_steps=self.ddim_steps,
            guidance_scale=self.guidance_scale
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute DIRE and classify.

        Args:
            images: Input images [B, 3, H, W] in range [-1, 1]

        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Compute DIRE
        dire = self.compute_dire(images)

        # Resize DIRE for classifier
        if dire.shape[-1] != self.classifier_input_size:
            dire = nn.functional.interpolate(
                dire,
                size=(self.classifier_input_size, self.classifier_input_size),
                mode='bilinear',
                align_corners=False
            )

        # Classify
        logits = self.classifier(dire)
        return logits

    def forward_precomputed(self, dire_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with precomputed DIRE features.

        Use this when DIRE maps are precomputed and cached.

        Args:
            dire_features: Precomputed DIRE maps [B, 3, H, W]

        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Resize if needed
        if dire_features.shape[-1] != self.classifier_input_size:
            dire_features = nn.functional.interpolate(
                dire_features,
                size=(self.classifier_input_size, self.classifier_input_size),
                mode='bilinear',
                align_corners=False
            )

        return self.classifier(dire_features)


def create_dire_detector(
    diffusion_model_id: str = "runwayml/stable-diffusion-v1-5",
    classifier_backbone: str = "resnet50",
    num_classes: int = 1,
    ddim_steps: int = 20,
    classifier_input_size: int = 224,
    device: str = "cuda"
) -> DIREDetector:
    """
    Factory function to create DIRE detector.

    Args:
        diffusion_model_id: Hugging Face model ID
        classifier_backbone: Backbone for classifier
        num_classes: Number of output classes
        ddim_steps: Number of DDIM steps
        classifier_input_size: Input size for classifier
        device: Device to run on

    Returns:
        DIREDetector instance
    """
    return DIREDetector(
        diffusion_model_id=diffusion_model_id,
        classifier_backbone=classifier_backbone,
        num_classes=num_classes,
        ddim_steps=ddim_steps,
        classifier_input_size=classifier_input_size,
        device=device
    )


def create_dire_classifier(
    backbone: str = "resnet50",
    num_classes: int = 1,
    pretrained: bool = True
) -> DIREClassifier:
    """
    Factory function to create DIRE classifier only.

    Use this when training with precomputed DIRE features.

    Args:
        backbone: Backbone model name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights

    Returns:
        DIREClassifier instance
    """
    return DIREClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        input_channels=3
    )


if __name__ == "__main__":
    # Test the model
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test classifier only (doesn't require GPU)
    print("\n--- Testing DIREClassifier ---")
    classifier = create_dire_classifier(
        backbone="resnet50",
        num_classes=1,
        pretrained=False
    )

    dummy_dire = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = classifier(dummy_dire)

    print(f"Input shape: {dummy_dire.shape}")
    print(f"Output shape: {output.shape}")

    # Test full detector (requires GPU with ~8GB VRAM)
    if device == "cuda":
        print("\n--- Testing DIREDetector (requires GPU) ---")

        detector = create_dire_detector(
            diffusion_model_id="runwayml/stable-diffusion-v1-5",
            classifier_backbone="resnet50",
            ddim_steps=10,  # Reduced for testing
            device=device
        )
        detector.to(device)

        dummy_image = torch.randn(1, 3, 512, 512).to(device)
        dummy_image = dummy_image.clamp(-1, 1)

        with torch.no_grad():
            dire_output = detector(dummy_image)

        print(f"Input shape: {dummy_image.shape}")
        print(f"Output shape: {dire_output.shape}")
    else:
        print("\nSkipping full detector test (requires CUDA)")
