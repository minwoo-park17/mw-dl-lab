"""
Evaluation script for DIRE and SeDID detectors.
"""
import os
import logging

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LoadDataInfo, PrecomputedFeatureDataset
from model_dire import create_dire_classifier
from model_sedid import create_sedid_classifier
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
    Evaluate model on test data.

    Args:
        data_config_path: Path to data configuration file
        device: Device to run on
        model: Trained model
        save_results_dir: Directory to save results
        test_loader: Test data loader
        folder_name: Name for results folder

    Returns:
        Evaluation results dictionary
    """
    # Calculate class weights
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
            features = data["input"].to(device=device, dtype=torch.float32)
            labels_batch = torch.unsqueeze(data["label"], dim=1).to(device=device, dtype=torch.float32)
            paths = data["file_path"]

            pred_labels = model(features)
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
    MODEL_TYPE = "dire"                               # "dire" or "sedid"
    DATA_CONFIG_PATH = "config/data_config.yaml"
    MODEL_CONFIG_PATH = "config/dire_config.yaml"     # or sedid_config.yaml
    WEIGHT_PATH = "./results/dire_train_XXXXXX/Epoch_BEST/weight/Epoch_BEST.pth"
    SAVE_DIR = "./results/evaluation"
    PRECOMPUTED_DIR = "./dire_cache"                  # or sedid_cache

    # Load configs
    with open(DATA_CONFIG_PATH, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    batch_size = model_config["training"]["bs"]
    classifier_input_size = model_config["model"]["classifier_input_size"]

    # Create dataset
    test_dataset = PrecomputedFeatureDataset(
        feature_dir=os.path.join(PRECOMPUTED_DIR, "test"),
        data_config_path=DATA_CONFIG_PATH,
        data_class="test",
        classifier_input_size=classifier_input_size,
        printcheck=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    classifier_backbone = model_config["model"]["classifier"]
    num_classes = model_config["model"]["num_classes"]

    if MODEL_TYPE == "dire":
        model = create_dire_classifier(
            backbone=classifier_backbone,
            num_classes=num_classes,
            pretrained=False
        )
    else:  # sedid
        num_timesteps = len(model_config["model"]["timesteps"])
        model = create_sedid_classifier(
            backbone=classifier_backbone,
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            pretrained=False
        )

    # Load weights
    if os.path.exists(WEIGHT_PATH):
        state_dict = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded weights from: {WEIGHT_PATH}")
    else:
        logger.warning(f"Weight file not found: {WEIGHT_PATH}")

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
