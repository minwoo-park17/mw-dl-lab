"""
Model definitions for deepfake classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


class FFTModule(nn.Module):
    """FFT-based frequency feature extraction module."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency features using FFT.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            FFT magnitude tensor [B, C, H, W]
        """
        fft = torch.fft.fft2(x)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shift)
        magnitude = torch.log1p(magnitude)  # Stabilize values
        return magnitude


class XceptionWithFFT(nn.Module):
    """Xception model with FFT frequency features."""

    def __init__(self, num_classes: int = 1):
        """
        Initialize XceptionWithFFT model.

        Args:
            num_classes: Number of output classes (1 for binary classification)
        """
        super().__init__()
        self.fft_module = FFTModule()

        # Pretrained Xception
        self.base_model = create_model('xception', pretrained=True)

        # Remove original classifier
        self.base_model.fc = nn.Identity()

        # Channel reduction (6 -> 3)
        self.reduce_channels = nn.Conv2d(6, 3, kernel_size=1)

        # Custom classifier
        self.classifier = nn.Linear(self.base_model.num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Output logits [B, num_classes]
        """
        # Extract FFT frequency features
        x_fft = self.fft_module(x)

        # Normalize FFT features
        x_fft = (x_fft - x_fft.mean(dim=[1, 2, 3], keepdim=True)) / \
                (x_fft.std(dim=[1, 2, 3], keepdim=True) + 1e-6)

        # Concatenate spatial and frequency features
        x_combined = torch.cat([x, x_fft], dim=1)

        # Reduce to 3 channels for Xception
        x_combined = self.reduce_channels(x_combined)

        # Xception forward
        feat = self.base_model(x_combined)
        out = self.classifier(feat)

        return out


def create_classifier_model(
    model_name: str,
    num_classes: int = 1,
    pretrained: bool = True
) -> nn.Module:
    """
    Create a classifier model.

    Args:
        model_name: Name of the model (e.g., 'xception', 'xception_fft')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights

    Returns:
        Model instance
    """
    if model_name == "xception_fft":
        return XceptionWithFFT(num_classes=num_classes)

    # Default: use timm model
    model = create_model(model_name, pretrained=pretrained)
    in_features = model.get_classifier().in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test XceptionWithFFT
    dummy_input = torch.randn(4, 3, 299, 299)
    model = XceptionWithFFT(num_classes=1)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.squeeze().tolist()}")
