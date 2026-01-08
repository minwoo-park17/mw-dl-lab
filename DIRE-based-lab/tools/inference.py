"""
Single image inference script for DIRE and SeDID detectors.
"""
import os
import sys
import logging
import argparse

import yaml
import torch
from PIL import Image
from torchvision import transforms

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_dire import create_dire_detector
from model_sedid import create_sedid_detector

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DIREInference:
    """DIRE inference class for single image prediction."""

    def __init__(
        self,
        model_config_path: str,
        classifier_weight_path: str,
        device: str = "cuda"
    ):
        """
        Initialize DIRE inference.

        Args:
            model_config_path: Path to model configuration file
            classifier_weight_path: Path to trained classifier weights
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load config
        with open(model_config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)

        self.img_size = model_config["model"]["image_size"]

        # Create detector
        self.detector = create_dire_detector(
            diffusion_model_id=model_config["model"]["diffusion_model"],
            classifier_backbone=model_config["model"]["classifier"],
            ddim_steps=model_config["model"]["ddim_steps"],
            classifier_input_size=model_config["model"]["classifier_input_size"],
            device=str(self.device)
        )

        # Load classifier weights
        if os.path.exists(classifier_weight_path):
            state_dict = torch.load(classifier_weight_path, map_location="cpu", weights_only=True)
            self.detector.classifier.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded classifier weights from: {classifier_weight_path}")
        else:
            raise FileNotFoundError(f"Weight file not found: {classifier_weight_path}")

        self.detector.to(self.device)
        self.detector.eval()

        # Setup transform
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def predict(self, image_path: str) -> dict:
        """
        Predict whether image is real or fake.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.detector(img_tensor)
            prob = torch.sigmoid(logits).item()

        # Determine prediction
        label = 1 if prob > 0.5 else 0
        prediction = "FAKE" if label == 1 else "REAL"

        return {
            "label": label,
            "confidence": prob,
            "prediction": prediction,
            "image_path": image_path
        }


class SeDIDInference:
    """SeDID inference class for single image prediction."""

    def __init__(
        self,
        model_config_path: str,
        classifier_weight_path: str,
        device: str = "cuda"
    ):
        """
        Initialize SeDID inference.

        Args:
            model_config_path: Path to model configuration file
            classifier_weight_path: Path to trained classifier weights
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load config
        with open(model_config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)

        self.img_size = model_config["model"]["image_size"]

        # Create detector
        self.detector = create_sedid_detector(
            diffusion_model_id=model_config["model"]["diffusion_model"],
            classifier_backbone=model_config["model"]["classifier"],
            timesteps=model_config["model"]["timesteps"],
            ddim_steps=20,
            classifier_input_size=model_config["model"]["classifier_input_size"],
            device=str(self.device)
        )

        # Load classifier weights
        if os.path.exists(classifier_weight_path):
            state_dict = torch.load(classifier_weight_path, map_location="cpu", weights_only=True)
            self.detector.classifier.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded classifier weights from: {classifier_weight_path}")
        else:
            raise FileNotFoundError(f"Weight file not found: {classifier_weight_path}")

        self.detector.to(self.device)
        self.detector.eval()

        # Setup transform
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def predict(self, image_path: str) -> dict:
        """
        Predict whether image is real or fake.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.detector(img_tensor)
            prob = torch.sigmoid(logits).item()

        # Determine prediction
        label = 1 if prob > 0.5 else 0
        prediction = "FAKE" if label == 1 else "REAL"

        return {
            "label": label,
            "confidence": prob,
            "prediction": prediction,
            "image_path": image_path
        }


def main():
    parser = argparse.ArgumentParser(description="DIRE/SeDID Single Image Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model-type", type=str, default="dire", choices=["dire", "sedid"],
                       help="Model type to use")
    parser.add_argument("--model-config", type=str, required=True,
                       help="Path to model config")
    parser.add_argument("--weight", type=str, required=True,
                       help="Path to trained classifier weights")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on")
    args = parser.parse_args()

    # Initialize inference
    if args.model_type == "dire":
        inference = DIREInference(
            model_config_path=args.model_config,
            classifier_weight_path=args.weight,
            device=args.device
        )
    else:
        inference = SeDIDInference(
            model_config_path=args.model_config,
            classifier_weight_path=args.weight,
            device=args.device
        )

    # Run prediction
    result = inference.predict(args.image)

    # Print result
    print("\n" + "="*50)
    print(f"Model: {args.model_type.upper()}")
    print(f"Image: {result['image_path']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
