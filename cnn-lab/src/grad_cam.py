"""
Grad-CAM visualization for deepfake classification.
"""
import os
import argparse
import logging
from typing import Optional, List

import yaml
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import create_classifier_model

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_target_layer(model: nn.Module, model_name: str, layer_name: Optional[str] = None) -> List[nn.Module]:
    """
    Get the target layer for Grad-CAM visualization.

    Args:
        model: The model
        model_name: Name of the model architecture
        layer_name: Optional specific layer name to use

    Returns:
        List containing the target layer
    """
    if layer_name is not None:
        # Try to get specific layer by name
        try:
            layer = dict(model.named_modules())[layer_name]
            logger.info(f"Using specified layer: {layer_name}")
            return [layer]
        except KeyError:
            logger.warning(f"Layer '{layer_name}' not found. Using default layer.")

    # Default layers for common architectures
    if "xception" in model_name.lower():
        if hasattr(model, 'base_model'):
            # For XceptionWithFFT
            return [model.base_model.conv4]
        elif hasattr(model, 'conv4'):
            return [model.conv4]
        else:
            # Try to find the last conv layer
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    logger.info(f"Using auto-detected layer: {name}")
                    return [module]

    elif "resnet" in model_name.lower():
        if hasattr(model, 'layer4'):
            return [model.layer4[-1]]

    elif "efficientnet" in model_name.lower():
        if hasattr(model, 'conv_head'):
            return [model.conv_head]

    elif "mobilenet" in model_name.lower():
        if hasattr(model, 'conv_head'):
            return [model.conv_head]
        elif hasattr(model, 'features'):
            return [model.features[-1]]

    # Fallback: find last conv layer
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            logger.info(f"Using auto-detected layer: {name}")
            return [module]

    raise ValueError(f"Could not find suitable target layer for model: {model_name}")


def generate_gradcam(
    csv_path: str,
    device: torch.device,
    model: nn.Module,
    model_name: str,
    actual_label: int,
    predict_label: int,
    save_dir: str,
    img_size: int,
    layer_name: Optional[str] = None
) -> None:
    """
    Generate Grad-CAM visualizations for filtered predictions.

    Args:
        csv_path: Path to CSV with prediction results
        device: Device to run on
        model: The model
        model_name: Name of the model architecture
        actual_label: Filter by actual label
        predict_label: Filter by predicted label
        save_dir: Directory to save visualizations
        img_size: Image size for the model
        layer_name: Optional specific layer name
    """
    # Setup transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Load and filter data
    csv_data = pd.read_csv(csv_path)
    filtered_data = csv_data[
        (csv_data['preds'] == predict_label) &
        (csv_data['labels'] == actual_label)
    ]

    if len(filtered_data) == 0:
        logger.warning(f"No samples found with actual={actual_label}, predicted={predict_label}")
        return

    logger.info(f"Processing {len(filtered_data)} samples (actual={actual_label}, pred={predict_label})")

    # Get target layer
    target_layers = get_target_layer(model, model_name, layer_name)

    model.eval()

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    cam.batch_size = 1
    targets = [BinaryClassifierOutputTarget(1)]

    for _, row in tqdm(filtered_data.iterrows(), total=len(filtered_data), desc="Generating Grad-CAM"):
        img_path = row['target_img_path']
        label = row['labels']

        try:
            # Load and process image
            img = Image.open(img_path).convert("RGB")
            original_height = img.height
            img_tensor = transform(img).to(device).unsqueeze(0)

            # Get prediction probability
            with torch.no_grad():
                pred_labels = model(img_tensor)
                pred_prob = torch.sigmoid(pred_labels)
                prob_value = pred_prob.item()

            # Generate Grad-CAM
            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]

            # Prepare visualization
            rgb_img = img.resize((img_size, img_size))
            rgb_img = np.array(rgb_img, dtype=np.float32) / 255.0
            gradcam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # Add probability text to original image
            original_image = (rgb_img * 255).astype(np.uint8)
            cv2.putText(
                original_image, f"{prob_value:.4f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )

            # Add original height info
            text_position = (rgb_img.shape[1] - 60, 30)
            cv2.putText(
                original_image, f"{original_height}px",
                text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )

            # Concatenate original and Grad-CAM
            concatenated_image = cv2.hconcat([original_image, gradcam_img])

            # Save
            save_path = os.path.join(save_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, concatenated_image[:, :, ::-1])

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue

    logger.info(f"Grad-CAM visualizations saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to CSV file with prediction results"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model configuration YAML file"
    )
    parser.add_argument(
        "--weight-path",
        type=str,
        required=True,
        help="Path to model weights"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save Grad-CAM visualizations"
    )
    parser.add_argument(
        "--actual-label",
        type=int,
        default=1,
        choices=[0, 1],
        help="Filter by actual label (0=real, 1=fake)"
    )
    parser.add_argument(
        "--predict-label",
        type=int,
        default=1,
        choices=[0, 1],
        help="Filter by predicted label (0=real, 1=fake)"
    )
    parser.add_argument(
        "--layer-name",
        type=str,
        default=None,
        help="Specific layer name for Grad-CAM (optional)"
    )
    args = parser.parse_args()

    # Load config
    with open(args.model_config, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    img_size = model_config["model"]["image-size"]
    num_classes = model_config["model"]["num-classes"]
    model_name = model_config["model"]["name"]

    # Create save directory
    save_subdir = os.path.join(args.save_dir, f"{args.actual_label}_{args.predict_label}")
    os.makedirs(save_subdir, exist_ok=True)

    # Load model
    model = create_classifier_model(model_name, num_classes=num_classes, pretrained=True)

    checkpoint = torch.load(args.weight_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    logger.info(f"Loaded weights from: {args.weight_path}")

    model.to(device)

    # Generate visualizations
    generate_gradcam(
        csv_path=args.csv_path,
        device=device,
        model=model,
        model_name=model_name,
        actual_label=args.actual_label,
        predict_label=args.predict_label,
        save_dir=save_subdir,
        img_size=img_size,
        layer_name=args.layer_name
    )


if __name__ == "__main__":
    main()
