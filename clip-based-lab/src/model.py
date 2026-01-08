"""
Model definitions for CLIP-based deepfake detection (UnivFD).

UnivFD: Universal Fake Detector using CLIP ViT-L/14 (frozen) + Linear Classifier
Reference: "Towards Universal Fake Image Detectors that Generalize Across Generative Models" (CVPR 2023)
"""
import torch
import torch.nn as nn

try:
    import clip
except ImportError:
    raise ImportError("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")


class UnivFDModel(nn.Module):
    """
    UnivFD model for deepfake detection.

    Uses CLIP ViT-L/14 as frozen feature extractor with a trainable linear classifier.
    Only the linear classifier is trained; CLIP backbone remains frozen.

    Input size: 224x224 (CLIP standard)
    Feature dimension: 768 (ViT-L/14)
    """

    # CLIP model configurations
    CLIP_CONFIGS = {
        "ViT-B/32": {"feature_dim": 512, "image_size": 224},
        "ViT-B/16": {"feature_dim": 512, "image_size": 224},
        "ViT-L/14": {"feature_dim": 768, "image_size": 224},
        "ViT-L/14@336px": {"feature_dim": 768, "image_size": 336},
    }

    def __init__(
        self,
        clip_model_name: str = "ViT-L/14",
        num_classes: int = 1,
        freeze_backbone: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize UnivFD model.

        Args:
            clip_model_name: CLIP model variant (ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px)
            num_classes: Number of output classes (1 for binary classification with BCEWithLogitsLoss)
            freeze_backbone: Whether to freeze CLIP backbone (should be True for UnivFD)
            device: Device to load CLIP model on
        """
        super().__init__()

        self.clip_model_name = clip_model_name
        self.freeze_backbone = freeze_backbone

        # Get CLIP configuration
        if clip_model_name not in self.CLIP_CONFIGS:
            raise ValueError(f"Unsupported CLIP model: {clip_model_name}. "
                           f"Supported: {list(self.CLIP_CONFIGS.keys())}")

        config = self.CLIP_CONFIGS[clip_model_name]
        self.feature_dim = config["feature_dim"]
        self.image_size = config["image_size"]

        # Load CLIP model
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=device)

        # Freeze CLIP backbone
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()

        # Linear classifier (trainable)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP features from images.

        Args:
            images: Preprocessed image tensor [B, C, H, W]

        Returns:
            features: Feature tensor [B, feature_dim]
        """
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.clip_model.encode_image(images)
        else:
            features = self.clip_model.encode_image(images)

        # Convert to float32 (CLIP outputs float16 on GPU)
        features = features.float()

        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: Preprocessed image tensor [B, C, H, W]

        Returns:
            logits: Classification logits [B, num_classes]
        """
        features = self.extract_features(images)
        logits = self.classifier(features)
        return logits

    def get_trainable_params(self):
        """Get only trainable parameters (classifier only)."""
        return self.classifier.parameters()

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_univfd_model(
    clip_model_name: str = "ViT-L/14",
    num_classes: int = 1,
    freeze_backbone: bool = True,
    device: str = "cuda"
) -> UnivFDModel:
    """
    Factory function to create UnivFD model.

    Args:
        clip_model_name: CLIP model variant
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze CLIP backbone
        device: Device to load model on

    Returns:
        UnivFDModel instance
    """
    model = UnivFDModel(
        clip_model_name=clip_model_name,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        device=device
    )
    return model


if __name__ == "__main__":
    # Test the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\n--- Testing UnivFD Model (ViT-L/14) ---")
    model = create_univfd_model(
        clip_model_name="ViT-L/14",
        num_classes=1,
        freeze_backbone=True,
        device=device
    )
    model.to(device)

    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Feature dimension: {model.feature_dim}")
    print(f"Image size: {model.image_size}")
    print(f"Trainable parameters: {model.get_num_trainable_params()}")

    # Test feature extraction
    features = model.extract_features(dummy_input)
    print(f"Feature shape: {features.shape}")
