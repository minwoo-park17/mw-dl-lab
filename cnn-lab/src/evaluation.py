"""
Evaluation script for deepfake classification.
"""
import os
import logging

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LoadDataInfo, CnnDataset
from model import create_classifier_model
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
    Evaluate the model on test data.

    Args:
        data_config_path: Path to data configuration file
        device: Device to evaluate on
        model: Model to evaluate
        save_results_dir: Directory to save results
        test_loader: Test data loader
        folder_name: Name for results folder

    Returns:
        Evaluation results dictionary
    """
    # Calculate class weights for loss function
    load_data_info = LoadDataInfo(data_config_path=data_config_path, data_class="test", printcheck=False)
    _, labels = load_data_info()
    n_fake = len([x for x in labels if x == 1])
    n_real = len([x for x in labels if x == 0])

    if n_fake > 0 and n_real > 0:
        bias_weight = torch.tensor([n_real / n_fake], device=device)
    else:
        bias_weight = torch.tensor([1.0], device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=bias_weight)

    # Create results directory
    os.makedirs(save_results_dir, exist_ok=True)

    results_template = {
        "loss": None, "n_data": None, "correct": None,
        "positive": None, "negative": None, "cm_total": None,
        "path_list": None, "label_list": None, "prob_list": None, "pred_list": None
    }

    total_test_loss = 0.0
    test_path_list, test_label_list, test_prob_list, test_pred_list = [], [], [], []
    test_correct, test_positive, test_negative = 0, 0, 0
    test_cm_total = np.zeros((2, 2))
    n_test_data = 0

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            images = data["input"].to(device=device, dtype=torch.float32)
            labels = torch.unsqueeze(data["label"], dim=1).to(device=device, dtype=torch.float32)
            paths = data["file_path"]

            pred_labels = model(images)
            pred_prob = torch.sigmoid(pred_labels)
            loss = loss_fn(pred_labels, labels)

            corrects, positive_class, negative_class, cm = check_correct(pred_labels, labels, classes=[0, 1])
            loss_value = loss.item()

            n_test_data += len(data["label"])
            test_prob_list += pred_prob.to(torch.float).cpu().detach().numpy().squeeze(axis=1).tolist()
            test_pred_list += torch.sigmoid(pred_labels).round().to(int).cpu().detach().numpy().squeeze(axis=1).tolist()
            test_label_list += labels.round().to(int).cpu().detach().numpy().squeeze(axis=1).tolist()
            test_path_list += paths

            test_cm_total += cm
            test_correct += corrects
            test_positive += positive_class
            test_negative += negative_class
            total_test_loss += loss_value

    # Store results
    results_info = {
        folder_name: {
            "loss": total_test_loss, "n_data": n_test_data, "correct": test_correct,
            "positive": test_positive, "negative": test_negative, "cm_total": test_cm_total,
            "path_list": test_path_list, "label_list": test_label_list,
            "prob_list": test_prob_list, "pred_list": test_pred_list
        }
    }

    # Calculate and log metrics
    accuracy = test_correct / n_test_data if n_test_data > 0 else 0
    avg_loss = total_test_loss / len(test_loader) if len(test_loader) > 0 else 0

    logger.info(f"Evaluation Results:")
    logger.info(f"  Total samples: {n_test_data}")
    logger.info(f"  Correct: {test_correct}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Average Loss: {avg_loss:.4f}")

    # Save results
    postprocess(results_info=results_info, save_dir=save_results_dir, folder_name=folder_name)

    logger.info(f"Results saved to: {save_results_dir}/{folder_name}")

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "n_samples": n_test_data,
        "correct": test_correct
    }


if __name__ == "__main__":
    ### PATH ###
    RESULT_NAME = "train_YYMMDDHHM"  # 수정 필요
    BASE_DIR = rf"./results/{RESULT_NAME}"

    DATA_CONFIG_PATH = rf"{BASE_DIR}/data_config.yaml"
    MODEL_CONFIG_PATH = rf"{BASE_DIR}/architecture.yaml"
    WEIGHT_PATH = rf"{BASE_DIR}/Epoch_BEST/weight/Epoch_BEST.pth"
    SAVE_RESULTS_DIR = rf"{BASE_DIR}/Analysis"

    FOLDER_NAME = "test_results"
    BATCH_SIZE = 8

    ### OPEN CONFIG FILE ###
    with open(DATA_CONFIG_PATH, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    img_size = model_config["model"]["image-size"]

    test_dataset = CnnDataset(
        data_config_path=DATA_CONFIG_PATH,
        data_class="test",
        img_size=img_size,
        printcheck=True
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create and load model
    num_classes = model_config["model"]["num-classes"]
    model_name = model_config["model"]["name"]

    model = create_classifier_model(model_name, num_classes=num_classes, pretrained=False)

    state_dict = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded weights from: {WEIGHT_PATH}")

    model.to(device)

    # Evaluate
    result = evaluate(
        data_config_path=DATA_CONFIG_PATH,
        device=device,
        model=model,
        save_results_dir=SAVE_RESULTS_DIR,
        test_loader=test_loader,
        folder_name=FOLDER_NAME
    )

    logger.info(f"Evaluation completed. Accuracy: {result['accuracy']:.4f}")
