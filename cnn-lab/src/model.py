"""
Model definitions for deepfake classification.

Supports any timm model: xception, resnet, efficientnet, mobilenet, etc.
"""
import torch
import torch.nn as nn
from timm import create_model


class EfficientNetB4Classifier(nn.Module):
    """
    EfficientNet-B4 classifier for deepfake detection.

    Pretrained input size: 380x380
    """
    IMAGE_SIZE = 380  # EfficientNet-B4 pretrained input size

    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        self.model = create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=num_classes
        )
        self.num_features = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def create_classifier_model(
    model_name: str,
    num_classes: int = 1,
    pretrained: bool = True
) -> nn.Module:
    """
    Create a classifier model using timm.

    Args:
        model_name: Any timm model name (e.g., 'xception', 'resnet50', 'efficientnet_b0')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights

    Returns:
        Model instance

    Examples:
        >>> model = create_classifier_model('xception', num_classes=1)
        >>> model = create_classifier_model('resnet50', num_classes=1)
        >>> model = create_classifier_model('efficientnet_b0', num_classes=1)
    """
    model = create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test xception
    print("\n--- Testing xception ---")
    model = create_classifier_model('xception', num_classes=1, pretrained=False)
    dummy_input = torch.randn(4, 3, 299, 299)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Feature dim: {model.num_features}")

    # Test resnet18
    print("\n--- Testing resnet18 ---")
    model_resnet = create_classifier_model('resnet18', num_classes=1, pretrained=False)

    with torch.no_grad():
        output_resnet = model_resnet(torch.randn(4, 3, 224, 224))

    print(f"Output shape: {output_resnet.shape}")
    print(f"Feature dim: {model_resnet.num_features}")

    # Test EfficientNetB4Classifier
    print("\n--- Testing EfficientNetB4Classifier ---")
    model_effb4 = EfficientNetB4Classifier(num_classes=1, pretrained=False)
    img_size = EfficientNetB4Classifier.IMAGE_SIZE
    dummy_input_effb4 = torch.randn(4, 3, img_size, img_size)

    with torch.no_grad():
        output_effb4 = model_effb4(dummy_input_effb4)

    print(f"Input shape: {dummy_input_effb4.shape}")
    print(f"Output shape: {output_effb4.shape}")
    print(f"Feature dim: {model_effb4.num_features}")
    print(f"Recommended image size: {EfficientNetB4Classifier.IMAGE_SIZE}")
