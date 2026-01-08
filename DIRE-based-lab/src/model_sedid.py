"""
SeDID (Stepwise Error for Diffusion-generated Image Detection) model.

Reference: "Exposing the Fake: Effective Diffusion-Generated Images Detection" (ICML Workshop 2023)
"""
import logging
from typing import List, Optional

import torch
import torch.nn as nn
from timm import create_model

from diffusion_utils import DiffusionReconstructor

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class SeDIDClassifier(nn.Module):
    """
    Classifier for SeDID features.

    Takes concatenated stepwise error maps as input.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 1,
        pretrained: bool = True,
        num_timesteps: int = 3,
        input_channels_per_timestep: int = 3
    ):
        """
        Initialize SeDID classifier.

        Args:
            backbone: Backbone model name (any timm model)
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            num_timesteps: Number of timesteps analyzed
            input_channels_per_timestep: Channels per timestep (3 for RGB)
        """
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.input_channels = num_timesteps * input_channels_per_timestep

        # Create backbone with adjusted input channels
        self.backbone = create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=self.input_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: SeDID features [B, num_timesteps*3, H, W]

        Returns:
            logits: Classification logits [B, num_classes]
        """
        return self.backbone(x)


class SeDIDDetector(nn.Module):
    """
    Full SeDID detector combining diffusion analysis and classification.

    Pipeline:
    1. For each timestep t in timesteps:
       - Add noise to image at timestep t
       - Predict noise with diffusion model
       - Compute error between actual and predicted noise
    2. Concatenate errors from all timesteps
    3. Classify concatenated features
    """

    def __init__(
        self,
        diffusion_model_id: str = "runwayml/stable-diffusion-v1-5",
        classifier_backbone: str = "resnet18",
        num_classes: int = 1,
        timesteps: List[int] = [250, 500, 750],
        ddim_steps: int = 20,
        classifier_input_size: int = 224,
        device: str = "cuda"
    ):
        """
        Initialize SeDID detector.

        Args:
            diffusion_model_id: Hugging Face model ID
            classifier_backbone: Backbone for classifier
            num_classes: Number of output classes
            timesteps: Timesteps to analyze
            ddim_steps: Number of DDIM steps
            classifier_input_size: Input size for classifier
            device: Device to run on
        """
        super().__init__()

        self.timesteps = timesteps
        self.ddim_steps = ddim_steps
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
        self.classifier = SeDIDClassifier(
            backbone=classifier_backbone,
            num_classes=num_classes,
            pretrained=True,
            num_timesteps=len(timesteps)
        )

    def compute_sedid(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute SeDID features for input images.

        Args:
            images: Input images [B, 3, H, W] in range [-1, 1]

        Returns:
            sedid: SeDID features [B, 3*num_timesteps, H, W]
        """
        return self.reconstructor.compute_sedid(
            images,
            timesteps=self.timesteps,
            num_steps=self.ddim_steps
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute SeDID features and classify.

        Args:
            images: Input images [B, 3, H, W] in range [-1, 1]

        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Compute SeDID features
        sedid = self.compute_sedid(images)

        # Resize for classifier
        if sedid.shape[-1] != self.classifier_input_size:
            sedid = nn.functional.interpolate(
                sedid,
                size=(self.classifier_input_size, self.classifier_input_size),
                mode='bilinear',
                align_corners=False
            )

        # Classify
        logits = self.classifier(sedid)
        return logits

    def forward_precomputed(self, sedid_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with precomputed SeDID features.

        Use this when SeDID features are precomputed and cached.

        Args:
            sedid_features: Precomputed SeDID features [B, 3*num_timesteps, H, W]

        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Resize if needed
        if sedid_features.shape[-1] != self.classifier_input_size:
            sedid_features = nn.functional.interpolate(
                sedid_features,
                size=(self.classifier_input_size, self.classifier_input_size),
                mode='bilinear',
                align_corners=False
            )

        return self.classifier(sedid_features)


class SeDIDStatistical:
    """
    Statistical variant of SeDID (SeDID_Stat).

    Uses statistical features of stepwise errors for detection
    without neural network classification.
    """

    def __init__(
        self,
        diffusion_model_id: str = "runwayml/stable-diffusion-v1-5",
        timesteps: List[int] = [250, 500, 750],
        ddim_steps: int = 20,
        threshold: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize SeDID statistical detector.

        Args:
            diffusion_model_id: Hugging Face model ID
            timesteps: Timesteps to analyze
            ddim_steps: Number of DDIM steps
            threshold: Decision threshold
            device: Device to run on
        """
        self.timesteps = timesteps
        self.ddim_steps = ddim_steps
        self.threshold = threshold
        self.device = device

        # Initialize diffusion reconstructor
        self.reconstructor = DiffusionReconstructor(
            model_id=diffusion_model_id,
            device=device
        )

    def compute_statistics(self, images: torch.Tensor) -> dict:
        """
        Compute statistical features from SeDID errors.

        Args:
            images: Input images [B, 3, H, W] in range [-1, 1]

        Returns:
            stats: Dictionary of statistical features
        """
        # Compute SeDID features
        sedid = self.reconstructor.compute_sedid(
            images,
            timesteps=self.timesteps,
            num_steps=self.ddim_steps
        )

        # Compute statistics
        stats = {
            'mean': sedid.mean(dim=[1, 2, 3]),
            'std': sedid.std(dim=[1, 2, 3]),
            'max': sedid.amax(dim=[1, 2, 3]),
            'min': sedid.amin(dim=[1, 2, 3]),
            'median': sedid.median(dim=3).values.median(dim=2).values.median(dim=1).values
        }

        return stats

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Predict using statistical decision.

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            predictions: Binary predictions [B]
        """
        stats = self.compute_statistics(images)

        # Simple decision rule: higher mean error -> more likely generated
        scores = stats['mean'] + 0.5 * stats['std']
        predictions = (scores > self.threshold).float()

        return predictions


def create_sedid_detector(
    diffusion_model_id: str = "runwayml/stable-diffusion-v1-5",
    classifier_backbone: str = "resnet18",
    num_classes: int = 1,
    timesteps: List[int] = [250, 500, 750],
    ddim_steps: int = 20,
    classifier_input_size: int = 224,
    device: str = "cuda"
) -> SeDIDDetector:
    """
    Factory function to create SeDID detector.

    Args:
        diffusion_model_id: Hugging Face model ID
        classifier_backbone: Backbone for classifier
        num_classes: Number of output classes
        timesteps: Timesteps to analyze
        ddim_steps: Number of DDIM steps
        classifier_input_size: Input size for classifier
        device: Device to run on

    Returns:
        SeDIDDetector instance
    """
    return SeDIDDetector(
        diffusion_model_id=diffusion_model_id,
        classifier_backbone=classifier_backbone,
        num_classes=num_classes,
        timesteps=timesteps,
        ddim_steps=ddim_steps,
        classifier_input_size=classifier_input_size,
        device=device
    )


def create_sedid_classifier(
    backbone: str = "resnet18",
    num_classes: int = 1,
    num_timesteps: int = 3,
    pretrained: bool = True
) -> SeDIDClassifier:
    """
    Factory function to create SeDID classifier only.

    Use this when training with precomputed SeDID features.

    Args:
        backbone: Backbone model name
        num_classes: Number of output classes
        num_timesteps: Number of timesteps in features
        pretrained: Whether to use pretrained weights

    Returns:
        SeDIDClassifier instance
    """
    return SeDIDClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        num_timesteps=num_timesteps
    )


if __name__ == "__main__":
    # Test the model
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test classifier only (doesn't require GPU)
    print("\n--- Testing SeDIDClassifier ---")
    classifier = create_sedid_classifier(
        backbone="resnet18",
        num_classes=1,
        num_timesteps=3,
        pretrained=False
    )

    # Input: 3 timesteps * 3 channels = 9 channels
    dummy_sedid = torch.randn(4, 9, 224, 224)
    with torch.no_grad():
        output = classifier(dummy_sedid)

    print(f"Input shape: {dummy_sedid.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected input channels: {classifier.input_channels}")

    # Test full detector (requires GPU with ~8GB VRAM)
    if device == "cuda":
        print("\n--- Testing SeDIDDetector (requires GPU) ---")

        detector = create_sedid_detector(
            diffusion_model_id="runwayml/stable-diffusion-v1-5",
            classifier_backbone="resnet18",
            timesteps=[250, 500],  # Reduced for testing
            ddim_steps=10,
            device=device
        )
        detector.to(device)

        dummy_image = torch.randn(1, 3, 512, 512).to(device)
        dummy_image = dummy_image.clamp(-1, 1)

        with torch.no_grad():
            sedid_output = detector(dummy_image)

        print(f"Input shape: {dummy_image.shape}")
        print(f"Output shape: {sedid_output.shape}")
    else:
        print("\nSkipping full detector test (requires CUDA)")
