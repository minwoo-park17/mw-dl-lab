"""
Inference script for CLIP-based deepfake detection (UnivFD).
Supports single image and batch evaluation with performance metrics.
Includes standalone GradCAM/Attention visualization.

This file is standalone - no external dependencies from src/ folder.
"""
import os
import logging
from glob import glob

import yaml
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

try:
    import clip
except ImportError:
    raise ImportError("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Definition (Standalone)
# =============================================================================
class UnivFDModel(nn.Module):
    """
    UnivFD model for deepfake detection.
    Uses CLIP ViT as frozen feature extractor with a trainable linear classifier.
    """

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
        super().__init__()

        self.clip_model_name = clip_model_name
        self.freeze_backbone = freeze_backbone

        if clip_model_name not in self.CLIP_CONFIGS:
            raise ValueError(f"Unsupported CLIP model: {clip_model_name}")

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
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.clip_model.encode_image(images)
        else:
            features = self.clip_model.encode_image(images)
        return features.float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(images)
        logits = self.classifier(features)
        return logits


# =============================================================================
# Inference Class
# =============================================================================
class UnivFDInference:
    """UnivFD inference class for image prediction and evaluation."""

    # CLIP normalization constants
    MEAN = [0.48145466, 0.4578275, 0.40821073]
    STD = [0.26862954, 0.26130258, 0.27577711]
    SUPPORT_TYPES = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']

    def __init__(
        self,
        model_config_path: str,
        weight_path: str,
        device: str = "cuda"
    ):
        """
        Initialize inference.

        Args:
            model_config_path: Path to model configuration file
            weight_path: Path to trained classifier weights
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load config
        with open(model_config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)

        self.img_size = model_config["model"]["image_size"]
        clip_model_name = model_config["model"]["clip_model"]
        num_classes = model_config["model"]["num_classes"]

        # Create model
        self.model = UnivFDModel(
            clip_model_name=clip_model_name,
            num_classes=num_classes,
            freeze_backbone=True,
            device=str(self.device)
        )

        # Load weights
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            self.model.classifier.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded classifier weights from: {weight_path}")
        else:
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        self.model.to(self.device)
        self.model.eval()

        # Setup transform
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def _get_image_files(self, paths: list) -> list:
        """Get all image files from paths (files or directories)."""
        image_files = []
        for path in paths:
            if os.path.isfile(path):
                if any(path.lower().endswith(ext) for ext in self.SUPPORT_TYPES):
                    image_files.append(path)
            elif os.path.isdir(path):
                for ext in self.SUPPORT_TYPES:
                    image_files.extend(glob(os.path.join(path, f"*{ext}")))
                    image_files.extend(glob(os.path.join(path, f"*{ext.upper()}")))
        return sorted(list(set(image_files)))

    def predict(self, image_path: str) -> dict:
        """
        Predict whether image is real or fake.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary with prediction results
        """
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            prob = torch.sigmoid(logits).item()

        label = 1 if prob > 0.5 else 0
        prediction = "FAKE" if label == 1 else "REAL"

        return {
            "label": label,
            "confidence": prob,
            "prediction": prediction,
            "image_path": image_path
        }

    def generate_attention_map(self, image_path: str) -> tuple:
        """
        Generate attention rollout visualization (standalone).

        Args:
            image_path: Path to input image

        Returns:
            Tuple of (original_image, attention_map, prediction_result)
        """
        img = Image.open(image_path).convert("RGB")
        original_img = img.copy()
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Get grid size from model
        patch_size = 14 if "14" in self.model.clip_model_name else 16
        grid_size = self.img_size // patch_size

        with torch.no_grad():
            x = img_tensor.type(self.model.clip_model.dtype)

            # Manual forward to capture attention
            x = self.model.clip_model.visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)

            class_emb = self.model.clip_model.visual.class_embedding.to(x.dtype)
            class_emb = class_emb.unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([class_emb, x], dim=1)

            x = x + self.model.clip_model.visual.positional_embedding.to(x.dtype)
            x = self.model.clip_model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)

            # Manual attention computation for each block
            all_attentions = []
            for block in self.model.clip_model.visual.transformer.resblocks:
                attn = block.attn
                x_norm = block.ln_1(x)

                # Compute Q, K, V
                qkv = torch.nn.functional.linear(x_norm, attn.in_proj_weight, attn.in_proj_bias)
                seq_len, batch_size, embed_dim = x_norm.shape
                num_heads = attn.num_heads
                head_dim = embed_dim // num_heads

                qkv = qkv.reshape(seq_len, batch_size, 3, num_heads, head_dim)
                qkv = qkv.permute(2, 1, 3, 0, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                # Compute attention weights
                scale = head_dim ** -0.5
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_weights = torch.softmax(attn_weights, dim=-1)
                attn_avg = attn_weights.mean(dim=1)
                all_attentions.append(attn_avg)

                # Continue forward pass
                attn_out = torch.matmul(attn_weights, v)
                attn_out = attn_out.permute(2, 0, 1, 3).reshape(seq_len, batch_size, embed_dim)
                attn_out = torch.nn.functional.linear(attn_out, attn.out_proj.weight, attn.out_proj.bias)
                x = x + attn_out
                x = x + block.mlp(block.ln_2(x))

            x = x.permute(1, 0, 2)
            x = self.model.clip_model.visual.ln_post(x[:, 0, :])
            if self.model.clip_model.visual.proj is not None:
                x = x @ self.model.clip_model.visual.proj

            logits = self.model.classifier(x.float())

        # Get prediction
        prob = torch.sigmoid(logits).item()
        prediction = "FAKE" if prob > 0.5 else "REAL"
        result = {
            "prediction": prediction,
            "confidence": prob,
            "label": 1 if prob > 0.5 else 0
        }

        # Compute attention rollout
        rollout = torch.eye(all_attentions[0].shape[-1], device=self.device).unsqueeze(0)
        for attn in all_attentions:
            attn_with_residual = 0.5 * attn + 0.5 * torch.eye(attn.shape[-1], device=self.device)
            rollout = torch.matmul(attn_with_residual, rollout)

        # Get attention from class token to spatial tokens
        cls_attention = rollout[0, 0, 1:]
        expected_size = grid_size * grid_size

        if cls_attention.numel() == expected_size:
            attention_map = cls_attention.reshape(grid_size, grid_size)
            attention_map = attention_map - attention_map.min()
            if attention_map.max() > 1e-8:
                attention_map = attention_map / attention_map.max()
            attention_map = attention_map.cpu().numpy()
        else:
            attention_map = np.ones((grid_size, grid_size)) * 0.5

        return original_img, attention_map, result

    def visualize_attention(
        self,
        image_path: str,
        output_path: str,
        alpha: float = 0.5
    ) -> dict:
        """
        Visualize attention map and save to file.

        Args:
            image_path: Path to input image
            output_path: Path to save visualization
            alpha: Transparency for overlay

        Returns:
            Prediction result dictionary
        """
        original_img, attention_map, result = self.generate_attention_map(image_path)

        # Use original image size for better quality
        orig_width, orig_height = original_img.size
        original_array = np.array(original_img) / 255.0

        # Resize heatmap to original image size (LANCZOS for high quality)
        heatmap_resized = Image.fromarray((attention_map * 255).astype(np.uint8))
        heatmap_resized = heatmap_resized.resize(
            (orig_width, orig_height),
            Image.Resampling.LANCZOS
        )
        heatmap_resized = np.array(heatmap_resized) / 255.0

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_array)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(heatmap_resized, cmap="jet")
        axes[1].set_title("Attention Heatmap")
        axes[1].axis("off")

        axes[2].imshow(original_array)
        axes[2].imshow(heatmap_resized, cmap="jet", alpha=alpha)
        axes[2].set_title(f"Overlay ({result['prediction']}, conf: {result['confidence']:.3f})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return result

    def evaluate(
        self,
        real_paths: list,
        fake_paths: list,
        show_errors: bool = False,
        save_attention: bool = False,
        attention_save_dir: str = None,
        max_attention_per_class: int = None
    ) -> dict:
        """
        Evaluate model on real and fake image lists.

        Args:
            real_paths: List of paths to real images (files or directories)
            fake_paths: List of paths to fake images (files or directories)
            show_errors: If True, print FN and FP image paths
            save_attention: If True, save attention visualizations
            attention_save_dir: Directory to save attention maps
            max_attention_per_class: Maximum number of attention maps per class (None for all)

        Returns:
            Dictionary with evaluation results
        """
        # Get all image files
        real_images = self._get_image_files(real_paths)
        fake_images = self._get_image_files(fake_paths)

        logger.info(f"Real images: {len(real_images)}")
        logger.info(f"Fake images: {len(fake_images)}")

        if len(real_images) == 0 and len(fake_images) == 0:
            logger.error("No images found!")
            return {}

        # Collect predictions
        all_labels = []
        all_preds = []
        all_paths = []

        # Process real images (label=0)
        logger.info("Processing real images...")
        for img_path in real_images:
            try:
                result = self.predict(img_path)
                all_labels.append(0)
                all_preds.append(result["label"])
                all_paths.append(img_path)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        # Process fake images (label=1)
        logger.info("Processing fake images...")
        for img_path in fake_images:
            try:
                result = self.predict(img_path)
                all_labels.append(1)
                all_preds.append(result["label"])
                all_paths.append(img_path)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        # Calculate metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)

        # Extract confusion matrix values
        tn, fp, fn, tp = cm.ravel()

        # Find error cases
        fn_paths = []  # Fake predicted as Real (missed fakes)
        fp_paths = []  # Real predicted as Fake (false alarms)

        for path, label, pred in zip(all_paths, all_labels, all_preds):
            if label == 1 and pred == 0:  # FN: Fake → Real
                fn_paths.append(path)
            elif label == 0 and pred == 1:  # FP: Real → Fake
                fp_paths.append(path)

        # Print results
        print("\n" + "=" * 60)
        print("                    EVALUATION RESULTS")
        print("=" * 60)

        print("\n[ Confusion Matrix ]")
        print("                  Predicted")
        print("                 Real    Fake")
        print(f"Actual Real     {tn:5d}   {fp:5d}")
        print(f"Actual Fake     {fn:5d}   {tp:5d}")

        print("\n[ Performance Metrics ]")
        print(f"  Accuracy:  {acc:.4f} ({int(acc * len(all_labels))}/{len(all_labels)})")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        print("\n[ Classification Summary ]")
        print(f"  TN (Real→Real):  {tn:5d}")
        print(f"  TP (Fake→Fake):  {tp:5d}")
        print(f"  FP (Real→Fake):  {fp:5d}  (False Alarm)")
        print(f"  FN (Fake→Real):  {fn:5d}  (Missed Fake)")

        print("=" * 60)

        # Show error paths if requested
        if show_errors:
            print("\n[ False Negative Paths (Fake predicted as Real) ]")
            if fn_paths:
                for i, path in enumerate(fn_paths, 1):
                    print(f"  {i}. {path}")
            else:
                print("  None")

            print("\n[ False Positive Paths (Real predicted as Fake) ]")
            if fp_paths:
                for i, path in enumerate(fp_paths, 1):
                    print(f"  {i}. {path}")
            else:
                print("  None")

            print("=" * 60)

        print()

        # Save attention visualizations if enabled
        if save_attention and attention_save_dir:
            logger.info("Generating attention visualizations...")

            # Create classification folders
            class_folders = {
                (0, 0): "0_0_TN",  # pred=Real, label=Real
                (0, 1): "0_1_FN",  # pred=Real, label=Fake
                (1, 0): "1_0_FP",  # pred=Fake, label=Real
                (1, 1): "1_1_TP",  # pred=Fake, label=Fake
            }

            for folder in class_folders.values():
                os.makedirs(os.path.join(attention_save_dir, folder), exist_ok=True)

            # Group images by classification
            class_images = {key: [] for key in class_folders.keys()}
            for path, label, pred in zip(all_paths, all_labels, all_preds):
                key = (int(pred), int(label))
                class_images[key].append(path)

            # Log statistics
            logger.info("Attention save statistics:")
            for key, folder in class_folders.items():
                count = len(class_images[key])
                logger.info(f"  {folder}: {count} images")

            # Process each class
            total_saved = 0
            for key, folder in class_folders.items():
                images = class_images[key]
                if max_attention_per_class:
                    images = images[:max_attention_per_class]

                folder_path = os.path.join(attention_save_dir, folder)

                for i, img_path in enumerate(images):
                    try:
                        filename = os.path.basename(img_path)
                        name, _ = os.path.splitext(filename)
                        output_path = os.path.join(folder_path, f"{name}_attention.png")

                        self.visualize_attention(img_path, output_path)
                        total_saved += 1

                        if (i + 1) % 10 == 0:
                            logger.info(f"  [{folder}] {i + 1}/{len(images)} saved")

                    except Exception as e:
                        logger.error(f"Error generating attention for {img_path}: {e}")

            logger.info(f"Total attention maps saved: {total_saved}")
            logger.info(f"Saved to: {attention_save_dir}")

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "fn_paths": fn_paths,
            "fp_paths": fp_paths,
            "total": len(all_labels)
        }


def main():
    # ============================================
    # Configuration - 직접 경로 설정
    # ============================================
    MODEL_CONFIG = rf"D:\study\mw-dl-lab\clip-based-lab\config\model_config.yaml"
    WEIGHT_PATH = rf"D:\study\mw-dl-lab\clip-based-lab\results\train_2601071358\Epoch_BEST\weight\Epoch_BEST.pth"
    DEVICE = "cuda"

    # 이미지 경로 리스트 (파일 또는 폴더 경로)
    REAL_PATHS = [
        rf"D:\Dataset\web_deepfake\reports_save\dev-deepfake_250510_to_251211\image_labeled_250509_to_251211\real_original",
        rf"D:\Dataset\web_deepfake\reports_save\dev-deepfake_250510_to_251211\image_labeled_250509_to_251211\real_edited",
    ]
    FAKE_PATHS = [
        rf"D:\Dataset\web_deepfake\reports_save\dev-deepfake_250510_to_251211\image_labeled_250509_to_251211\fake_face_genai",
    ]

    # True로 설정하면 FN, FP 이미지 경로 출력
    SHOW_ERRORS = True

    # ============================================
    # Attention 시각화 설정
    # ============================================
    SAVE_ATTENTION = False                          # True: Attention 맵 저장, False: 저장 안함
    ATTENTION_SAVE_DIR = "./attention_results"      # Attention 맵 저장 경로
    MAX_ATTENTION_PER_CLASS = 10                    # 클래스당 최대 저장 개수 (None이면 전체)
    # ============================================

    # Initialize inference
    inference = UnivFDInference(
        model_config_path=MODEL_CONFIG,
        weight_path=WEIGHT_PATH,
        device=DEVICE
    )

    # Run evaluation
    if REAL_PATHS or FAKE_PATHS:
        inference.evaluate(
            real_paths=REAL_PATHS,
            fake_paths=FAKE_PATHS,
            show_errors=SHOW_ERRORS,
            save_attention=SAVE_ATTENTION,
            attention_save_dir=ATTENTION_SAVE_DIR,
            max_attention_per_class=MAX_ATTENTION_PER_CLASS
        )
    else:
        print("Please set REAL_PATHS and/or FAKE_PATHS in the configuration.")


if __name__ == "__main__":
    main()
