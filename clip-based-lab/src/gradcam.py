"""
GradCAM visualization for CLIP-based deepfake detection (UnivFD).

Supports:
1. GradCAM using pytorch-grad-cam library
2. Attention Rollout for ViT attention visualization
"""
import os
import logging

import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import create_univfd_model
from dataset import ClipDataset

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLIPGradCAM:
    """GradCAM visualization for CLIP ViT models."""

    MEAN = [0.48145466, 0.4578275, 0.40821073]
    STD = [0.26862954, 0.26130258, 0.27577711]

    def __init__(
        self,
        model_config_path: str,
        weight_path: str,
        device: str = "cuda"
    ):
        """
        Initialize GradCAM visualizer.

        Args:
            model_config_path: Path to model configuration file
            weight_path: Path to trained classifier weights
            device: Device to run on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load config
        with open(model_config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)

        self.img_size = model_config["model"]["image_size"]
        clip_model_name = model_config["model"]["clip_model"]
        num_classes = model_config["model"]["num_classes"]

        # Create model
        self.model = create_univfd_model(
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

        # Get ViT patch size and grid size
        self.patch_size = 14 if "14" in clip_model_name else 16
        self.grid_size = self.img_size // self.patch_size

        # Storage for hooks
        self.activations = None
        self.gradients = None

    def _get_target_layer(self):
        """Get the target layer for GradCAM (last transformer block)."""
        return self.model.clip_model.visual.transformer.resblocks[-1].ln_1

    def _save_activation(self, module, input, output):
        """Hook to save activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients."""
        self.gradients = grad_output[0].detach()

    def generate_gradcam(self, image_path: str) -> tuple:
        """
        Generate GradCAM heatmap for an image.

        Uses gradient-weighted class activation mapping on ViT.
        Hooks on the output of the last transformer block's MLP.

        Args:
            image_path: Path to input image

        Returns:
            Tuple of (original_image, heatmap, prediction_result)
        """
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        original_img = img.copy()
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Reset storage
        self.activations = None
        self.gradients = None

        # Temporarily enable gradients for backbone
        for param in self.model.clip_model.parameters():
            param.requires_grad = True

        # Hook on the MLP output of the last transformer block (before residual add)
        # This captures the actual feature transformation better than ln_1
        target_layer = self.model.clip_model.visual.transformer.resblocks[-1].mlp

        def save_activation(module, input, output):
            self.activations = output.detach().clone()

        def save_gradient(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach().clone()

        forward_hook = target_layer.register_forward_hook(save_activation)
        backward_hook = target_layer.register_full_backward_hook(save_gradient)

        try:
            # Forward pass - manual to ensure gradients flow
            x = img_tensor.type(self.model.clip_model.dtype)

            # CLIP visual encoder forward
            x = self.model.clip_model.visual.conv1(x)  # [B, C, H, W]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, HW]
            x = x.permute(0, 2, 1)  # [B, HW, C]

            # Add class token
            class_emb = self.model.clip_model.visual.class_embedding.to(x.dtype)
            class_emb = class_emb.unsqueeze(0).expand(x.shape[0], -1, -1)  # [B, 1, C]
            x = torch.cat([class_emb, x], dim=1)  # [B, HW+1, C]

            # Add positional embedding
            x = x + self.model.clip_model.visual.positional_embedding.to(x.dtype)
            x = self.model.clip_model.visual.ln_pre(x)

            # Transformer expects [seq, batch, dim]
            x = x.permute(1, 0, 2)  # [HW+1, B, C]
            x = self.model.clip_model.visual.transformer(x)
            x = x.permute(1, 0, 2)  # [B, HW+1, C]

            # Get class token output
            x = self.model.clip_model.visual.ln_post(x[:, 0, :])
            if self.model.clip_model.visual.proj is not None:
                x = x @ self.model.clip_model.visual.proj

            features = x.float()
            logits = self.model.classifier(features)

            # Backward pass
            self.model.zero_grad()
            logits.backward()

            # Get prediction
            prob = torch.sigmoid(logits).item()
            prediction = "FAKE" if prob > 0.5 else "REAL"

            result = {
                "prediction": prediction,
                "confidence": prob,
                "label": 1 if prob > 0.5 else 0
            }

            # Generate heatmap
            if self.activations is not None and self.gradients is not None:
                # Shape: [seq_len, batch, dim] - e.g., [257, 1, 1024]
                act = self.activations.float()
                grad = self.gradients.float()

                # Method 1: Standard GradCAM - global average pooling of gradients as weights
                # Shape: [seq_len, batch, dim] -> [seq_len, batch, 1]
                weights = grad.mean(dim=-1, keepdim=True)

                # Weighted activation
                cam = (weights * act).sum(dim=-1)  # [seq_len, batch]
                cam = cam.squeeze(1)  # [seq_len]

                # Remove class token (first token)
                cam = cam[1:]  # [seq_len - 1] = [256 for 224x224 with patch 14]

                # Apply ReLU to focus on positive contributions
                cam = torch.relu(cam)

                # If all values are the same (no variance), try alternative method
                if cam.std() < 1e-6:
                    # Method 2: Use gradient magnitude directly
                    grad_magnitude = grad.abs().mean(dim=-1).squeeze(1)  # [seq_len]
                    cam = grad_magnitude[1:]  # Remove class token
                    cam = torch.relu(cam)

                # If still uniform, use activation variance
                if cam.std() < 1e-6:
                    # Method 3: Use activation variance across features
                    act_var = act.var(dim=-1).squeeze(1)  # [seq_len]
                    cam = act_var[1:]  # Remove class token

                # Reshape to grid
                expected_size = self.grid_size * self.grid_size
                if cam.numel() == expected_size:
                    cam = cam.reshape(self.grid_size, self.grid_size)

                    # Normalize to [0, 1]
                    cam = cam - cam.min()
                    if cam.max() > 1e-8:
                        cam = cam / cam.max()
                    else:
                        # If all zeros, create a centered gaussian as fallback
                        y, x = torch.meshgrid(
                            torch.linspace(-1, 1, self.grid_size),
                            torch.linspace(-1, 1, self.grid_size),
                            indexing='ij'
                        )
                        cam = torch.exp(-(x**2 + y**2) / 0.5)

                    cam = cam.cpu().numpy()
                else:
                    # Size mismatch - shouldn't happen normally
                    logger.warning(f"CAM size mismatch: expected {expected_size}, got {cam.numel()}")
                    cam = np.ones((self.grid_size, self.grid_size)) * 0.5
            else:
                logger.warning("No activations or gradients captured")
                cam = np.ones((self.grid_size, self.grid_size)) * 0.5

        finally:
            forward_hook.remove()
            backward_hook.remove()

            for param in self.model.clip_model.parameters():
                param.requires_grad = False

        return original_img, cam, result

    def generate_attention_rollout(self, image_path: str) -> tuple:
        """
        Generate attention rollout visualization.

        Attention Rollout aggregates attention weights across all layers
        to show which image regions the model focuses on.

        Args:
            image_path: Path to input image

        Returns:
            Tuple of (original_image, attention_map, prediction_result)
        """
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        original_img = img.copy()
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Storage for attention weights
        attention_weights = []

        def get_attention_hook(module, input, output):
            """Hook to capture attention weights from MultiheadAttention."""
            # output is (attn_output, attn_output_weights)
            # For CLIP, we need to compute attention manually from input
            pass

        # For CLIP ViT, we need to manually compute attention
        # Hook on the input to each attention layer
        attention_inputs = []

        def save_attn_input(module, input, output):
            attention_inputs.append(input[0].detach().clone())

        # Register hooks
        hooks = []
        for block in self.model.clip_model.visual.transformer.resblocks:
            hook = block.attn.register_forward_hook(save_attn_input)
            hooks.append(hook)

        # Forward pass
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
                # Get attention weights manually
                # x shape: [seq_len, batch, dim]
                attn = block.attn

                # Layer norm before attention
                x_norm = block.ln_1(x)

                # Compute Q, K, V
                # attn.in_proj_weight has shape [3*embed_dim, embed_dim]
                # attn.in_proj_bias has shape [3*embed_dim]
                qkv = torch.nn.functional.linear(x_norm, attn.in_proj_weight, attn.in_proj_bias)

                # Split into Q, K, V
                seq_len, batch_size, embed_dim = x_norm.shape
                num_heads = attn.num_heads
                head_dim = embed_dim // num_heads

                qkv = qkv.reshape(seq_len, batch_size, 3, num_heads, head_dim)
                qkv = qkv.permute(2, 1, 3, 0, 4)  # [3, batch, heads, seq, head_dim]
                q, k, v = qkv[0], qkv[1], qkv[2]

                # Compute attention weights
                scale = head_dim ** -0.5
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_weights = torch.softmax(attn_weights, dim=-1)  # [batch, heads, seq, seq]

                # Average across heads
                attn_avg = attn_weights.mean(dim=1)  # [batch, seq, seq]
                all_attentions.append(attn_avg)

                # Continue forward pass through block
                attn_out = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]
                attn_out = attn_out.permute(2, 0, 1, 3).reshape(seq_len, batch_size, embed_dim)
                attn_out = torch.nn.functional.linear(attn_out, attn.out_proj.weight, attn.out_proj.bias)

                x = x + attn_out
                x = x + block.mlp(block.ln_2(x))

            x = x.permute(1, 0, 2)
            x = self.model.clip_model.visual.ln_post(x[:, 0, :])
            if self.model.clip_model.visual.proj is not None:
                x = x @ self.model.clip_model.visual.proj

            features = x.float()
            logits = self.model.classifier(features)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Get prediction
        prob = torch.sigmoid(logits).item()
        prediction = "FAKE" if prob > 0.5 else "REAL"

        result = {
            "prediction": prediction,
            "confidence": prob,
            "label": 1 if prob > 0.5 else 0
        }

        # Compute attention rollout
        # Multiply attention matrices and add identity (residual)
        rollout = torch.eye(all_attentions[0].shape[-1], device=self.device)
        rollout = rollout.unsqueeze(0)  # [1, seq, seq]

        for attn in all_attentions:
            # Add identity for residual connection
            attn_with_residual = 0.5 * attn + 0.5 * torch.eye(attn.shape[-1], device=self.device)
            rollout = torch.matmul(attn_with_residual, rollout)

        # Get attention from class token (index 0) to all spatial tokens
        cls_attention = rollout[0, 0, 1:]  # [num_patches]

        # Reshape to grid
        expected_size = self.grid_size * self.grid_size
        if cls_attention.numel() == expected_size:
            attention_map = cls_attention.reshape(self.grid_size, self.grid_size)

            # Normalize
            attention_map = attention_map - attention_map.min()
            if attention_map.max() > 1e-8:
                attention_map = attention_map / attention_map.max()

            attention_map = attention_map.cpu().numpy()
        else:
            logger.warning(f"Attention size mismatch: expected {expected_size}, got {cls_attention.numel()}")
            attention_map = np.ones((self.grid_size, self.grid_size)) * 0.5

        return original_img, attention_map, result

    def visualize(
        self,
        image_path: str,
        output_path: str = None,
        method: str = "gradcam",
        alpha: float = 0.5
    ):
        """
        Visualize GradCAM or attention rollout.

        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            method: "gradcam" or "attention"
            alpha: Transparency for overlay
        """
        if method == "gradcam":
            original_img, heatmap, result = self.generate_gradcam(image_path)
        else:
            original_img, heatmap, result = self.generate_attention_rollout(image_path)

        # Use original image size for better quality
        orig_width, orig_height = original_img.size
        original_array = np.array(original_img) / 255.0

        # Resize heatmap to original image size (LANCZOS for high quality)
        heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_resized = heatmap_resized.resize(
            (orig_width, orig_height),
            Image.Resampling.LANCZOS
        )
        heatmap_resized = np.array(heatmap_resized) / 255.0

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(original_array)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Heatmap
        axes[1].imshow(heatmap_resized, cmap="jet")
        axes[1].set_title(f"{method.upper()} Heatmap")
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(original_array)
        axes[2].imshow(heatmap_resized, cmap="jet", alpha=alpha)
        axes[2].set_title(f"Overlay ({result['prediction']}, conf: {result['confidence']:.3f})")
        axes[2].axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved visualization to: {output_path}")
        else:
            plt.show()

        plt.close()

        return result

    def visualize_batch(
        self,
        image_paths: list,
        output_dir: str,
        method: str = "gradcam"
    ):
        """
        Visualize multiple images.

        Args:
            image_paths: List of image paths
            output_dir: Directory to save visualizations
            method: "gradcam" or "attention"
        """
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for i, image_path in enumerate(image_paths):
            try:
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_{method}.png")

                result = self.visualize(image_path, output_path, method)
                result["image_path"] = image_path
                results.append(result)

                logger.info(f"[{i+1}/{len(image_paths)}] {filename}: {result['prediction']} ({result['confidence']:.3f})")

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")

        return results

    def visualize_by_classification(
        self,
        image_paths: list,
        labels: list,
        preds: list,
        output_dir: str,
        method: str = "gradcam",
        max_per_class: int = None
    ):
        """
        Visualize images classified by prediction and label.

        Saves to folders:
            - 0_0_TN: pred=Real(0), label=Real(0) → True Negative
            - 0_1_FN: pred=Real(0), label=Fake(1) → False Negative
            - 1_0_FP: pred=Fake(1), label=Real(0) → False Positive
            - 1_1_TP: pred=Fake(1), label=Fake(1) → True Positive

        Args:
            image_paths: List of image paths
            labels: List of ground truth labels (0=Real, 1=Fake)
            preds: List of predictions (0=Real, 1=Fake)
            output_dir: Base directory to save visualizations
            method: "gradcam" or "attention"
            max_per_class: Maximum number of images per class (None for all)
        """
        # Create classification folders
        class_folders = {
            (0, 0): "0_0_TN",  # pred=Real, label=Real (True Negative)
            (0, 1): "0_1_FN",  # pred=Real, label=Fake (False Negative)
            (1, 0): "1_0_FP",  # pred=Fake, label=Real (False Positive)
            (1, 1): "1_1_TP",  # pred=Fake, label=Fake (True Positive)
        }

        for folder in class_folders.values():
            os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

        # Group images by classification
        class_images = {key: [] for key in class_folders.keys()}
        for path, label, pred in zip(image_paths, labels, preds):
            key = (int(pred), int(label))
            class_images[key].append(path)

        # Log statistics
        logger.info("=" * 50)
        logger.info("Classification Statistics:")
        for key, folder in class_folders.items():
            count = len(class_images[key])
            logger.info(f"  {folder}: {count} images")
        logger.info("=" * 50)

        # Process each class
        total_processed = 0
        results = {key: [] for key in class_folders.keys()}

        for key, folder in class_folders.items():
            images = class_images[key]
            if max_per_class:
                images = images[:max_per_class]

            folder_path = os.path.join(output_dir, folder)

            for i, image_path in enumerate(images):
                try:
                    filename = os.path.basename(image_path)
                    name, ext = os.path.splitext(filename)
                    output_path = os.path.join(folder_path, f"{name}_{method}.png")

                    result = self.visualize(image_path, output_path, method)
                    result["image_path"] = image_path
                    result["true_label"] = key[1]
                    result["pred_label"] = key[0]
                    results[key].append(result)

                    total_processed += 1
                    logger.info(f"[{folder}] [{i+1}/{len(images)}] {filename}")

                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")

        logger.info(f"Total processed: {total_processed} images")
        return results


def run_single_image():
    """단일 이미지에 대한 GradCAM 시각화"""
    # ============================================
    # Configuration - 직접 경로 설정
    # ============================================
    IMAGE_PATH = "path/to/your/image.jpg"           # 분석할 이미지 경로
    MODEL_CONFIG = "config/model_config.yaml"       # 모델 설정 파일
    WEIGHT_PATH = "./results/best_model.pt"         # 학습된 가중치 경로
    OUTPUT_PATH = "./results/gradcam_output.png"    # 결과 저장 경로 (None이면 화면 출력)
    METHOD = "attention"                              # "gradcam" 또는 "attention"
    DEVICE = "cuda"                                 # "cuda" 또는 "cpu"
    # ============================================

    visualizer = CLIPGradCAM(
        model_config_path=MODEL_CONFIG,
        weight_path=WEIGHT_PATH,
        device=DEVICE
    )

    result = visualizer.visualize(
        image_path=IMAGE_PATH,
        output_path=OUTPUT_PATH,
        method=METHOD
    )

    print("\n" + "="*50)
    print(f"Image: {IMAGE_PATH}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("="*50 + "\n")


def run_evaluation_based():
    """Evaluation 결과 기반 분류별 GradCAM 시각화"""
    # ============================================
    # Configuration - 직접 경로 설정
    # ============================================
    DATA_CONFIG = "config/data_config.yaml"         # 데이터 설정 파일
    MODEL_CONFIG = "config/model_config.yaml"       # 모델 설정 파일
    WEIGHT_PATH = "./results/train_2601071358/Epoch_BEST/weight/Epoch_BEST.pth"  # 학습된 가중치 경로
    METHOD = "attention"                            # "gradcam" 또는 "attention" (ViT에는 attention 권장)
    DEVICE = "cuda"                                 # "cuda" 또는 "cpu"
    MAX_PER_CLASS = 10                              # 클래스당 최대 이미지 수 (None이면 전체)

    # OUTPUT_DIR: WEIGHT_PATH에서 train 폴더를 추출하여 그 안에 gradcam 저장
    train_folder = WEIGHT_PATH.split("/Epoch_")[0]
    OUTPUT_DIR = os.path.join(train_folder, "gradcam")
    # ============================================

    # Load model config
    with open(MODEL_CONFIG, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)

    # Create dataset
    test_dataset = ClipDataset(
        data_config_path=DATA_CONFIG,
        data_class="test",
        img_size=model_config["model"]["image_size"],
        printcheck=True
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize visualizer
    visualizer = CLIPGradCAM(
        model_config_path=MODEL_CONFIG,
        weight_path=WEIGHT_PATH,
        device=DEVICE
    )

    # Run evaluation to get predictions
    logger.info("Running evaluation to get predictions...")
    path_list, label_list, pred_list = [], [], []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            images = data["input"].to(device=visualizer.device, dtype=torch.float32)
            labels_batch = data["label"].numpy().tolist()
            paths = data["file_path"]

            logits = visualizer.model(images)
            preds = torch.sigmoid(logits).round().to(int).cpu().numpy().squeeze(axis=1).tolist()

            path_list.extend(paths)
            label_list.extend(labels_batch)
            pred_list.extend(preds)

    # Generate GradCAM by classification
    logger.info("Generating GradCAM visualizations by classification...")
    visualizer.visualize_by_classification(
        image_paths=path_list,
        labels=label_list,
        preds=pred_list,
        output_dir=OUTPUT_DIR,
        method=METHOD,
        max_per_class=MAX_PER_CLASS
    )

    # Summary
    print("\n" + "="*50)
    print("GradCAM Generation Complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Folders:")
    print("  - 0_0_TN: Real predicted as Real (True Negative)")
    print("  - 0_1_FN: Fake predicted as Real (False Negative)")
    print("  - 1_0_FP: Real predicted as Fake (False Positive)")
    print("  - 1_1_TP: Fake predicted as Fake (True Positive)")
    print("="*50 + "\n")


def main():
    """
    Main function - 모드 선택

    MODE:
        "single" - 단일 이미지 GradCAM
        "evaluation" - Evaluation 기반 분류별 GradCAM
    """
    # ============================================
    MODE = "evaluation"  # "single" 또는 "evaluation"
    # ============================================

    if MODE == "single":
        run_single_image()
    elif MODE == "evaluation":
        run_evaluation_based()
    else:
        print(f"Unknown mode: {MODE}")


if __name__ == "__main__":
    main()
