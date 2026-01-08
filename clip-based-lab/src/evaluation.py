"""
Evaluation script for CLIP-based deepfake detection (UnivFD).
"""
import os
import logging

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LoadDataInfo, ClipDataset
from model import create_univfd_model
from utils import check_correct, postprocess

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate(
    data_config_path: str,
    device: torch.device,
    model: nn.Module,
    save_results_dir: str,
    test_loader: DataLoader,
    folder_name: str = "test"
) -> dict:
    """
    Evaluate the UnivFD model on test data.

    Args:
        data_config_path: Path to data configuration file
        device: Device to run on
        model: Trained UnivFD model
        save_results_dir: Directory to save results
        test_loader: Test data loader
        folder_name: Name for results folder

    Returns:
        Evaluation results dictionary
    """
    # Calculate class weights for loss calculation
    load_data_info = LoadDataInfo(data_config_path=data_config_path, data_class="test", printcheck=False)
    _, labels = load_data_info()
    n_fake = len([x for x in labels if x == 1])
    n_real = len([x for x in labels if x == 0])

    if n_fake > 0 and n_real > 0:
        bias_weight = torch.tensor([n_real / n_fake], device=device)
    else:
        bias_weight = torch.tensor([1.0], device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=bias_weight)

    results_template = {
        "loss": None, "n_data": None, "correct": None,
        "positive": None, "negative": None, "cm_total": None,
        "path_list": None, "label_list": None, "prob_list": None, "pred_list": None
    }

    # Evaluation
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        path_list, label_list, prob_list, pred_list = [], [], [], []
        correct, positive, negative = 0, 0, 0
        cm_total = np.zeros((2, 2))
        n_data = 0

        for data in tqdm(test_loader, desc=f"Evaluating - {folder_name}"):
            images = data["input"].to(device=device, dtype=torch.float32)
            labels_batch = torch.unsqueeze(data["label"], dim=1).to(device=device, dtype=torch.float32)
            paths = data["file_path"]

            pred_labels = model(images)
            loss = loss_fn(pred_labels, labels_batch)

            corrects, positive_class, negative_class, cm = check_correct(pred_labels, labels_batch, classes=[0, 1])
            loss_value = loss.item()

            n_data += len(data["label"])
            label_list += labels_batch.round().to(int).cpu().detach().numpy().squeeze(axis=1).tolist()
            pred_prob = torch.sigmoid(pred_labels)
            prob_list += pred_prob.to(torch.float).cpu().detach().numpy().squeeze(axis=1).tolist()
            pred_list += torch.sigmoid(pred_labels).round().to(int).cpu().detach().numpy().squeeze(axis=1).tolist()
            path_list += paths

            cm_total += cm
            correct += corrects
            positive += positive_class
            negative += negative_class
            total_loss += loss_value

    # Store results
    results_info = {
        folder_name: {
            "loss": total_loss, "n_data": n_data, "correct": correct,
            "positive": positive, "negative": negative, "cm_total": cm_total,
            "path_list": path_list, "label_list": label_list,
            "prob_list": prob_list, "pred_list": pred_list
        }
    }

    # Save results
    os.makedirs(save_results_dir, exist_ok=True)
    postprocess(results_info=results_info, save_dir=save_results_dir, folder_name=folder_name)

    # Calculate metrics
    accuracy = correct / n_data if n_data > 0 else 0
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0

    logger.info(f"Evaluation completed: accuracy={accuracy:.4f}, loss={avg_loss:.4f}")

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "n_samples": n_data,
        "correct": correct
    }


def main():
    """Main evaluation function."""
    # =============================================================================
    # Configuration - 여기서 값을 수정하세요
    # =============================================================================
    DATA_CONFIG_PATH = "config/data_config.yaml"      # 데이터 설정 파일 경로
    MODEL_CONFIG_PATH = "config/model_config.yaml"    # 모델 설정 파일 경로
    WEIGHT_PATH = "./results/train_2601071358/Epoch_BEST/weight/Epoch_BEST.pth"  # 학습된 가중치 경로

    # SAVE_DIR: WEIGHT_PATH에서 train 폴더를 추출하여 그 안에 evaluation 저장
    # 예: ./results/train_2601071358/Epoch_BEST/weight/Epoch_BEST.pth
    #  → ./results/train_2601071358/evaluation
    train_folder = WEIGHT_PATH.split("/Epoch_")[0]  # ./results/train_2601071358
    SAVE_DIR = os.path.join(train_folder, "evaluation")

    # Load configs
    with open(DATA_CONFIG_PATH, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    batch_size = model_config["training"]["bs"]
    img_size = model_config["model"]["image_size"]

    test_dataset = ClipDataset(
        data_config_path=DATA_CONFIG_PATH,
        data_class="test",
        img_size=img_size,
        printcheck=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    clip_model_name = model_config["model"]["clip_model"]
    num_classes = model_config["model"]["num_classes"]
    freeze_backbone = model_config["model"]["freeze_backbone"]

    model = create_univfd_model(
        clip_model_name=clip_model_name,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        device=str(device)
    )

    # Load trained classifier weights
    if os.path.exists(WEIGHT_PATH):
        state_dict = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=True)
        model.classifier.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded classifier weights from: {WEIGHT_PATH}")
    else:
        logger.warning(f"Weight file not found: {WEIGHT_PATH}")
        logger.warning("Using random weights for classifier.")

    model.to(device)

    # Evaluate
    result = evaluate(
        data_config_path=DATA_CONFIG_PATH,
        device=device,
        model=model,
        save_results_dir=SAVE_DIR,
        test_loader=test_loader,
        folder_name="test"
    )

    logger.info(f"Evaluation completed. Results saved to: {SAVE_DIR}")
    logger.info(f"Accuracy: {result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
