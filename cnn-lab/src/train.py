"""
Training script for deepfake classification.
"""
import os
import argparse
import datetime
import logging
from typing import Optional

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from timm import create_model

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


def train(
    data_config_path: str,
    model_config: dict,
    device: torch.device,
    model: nn.Module,
    save_results_dir: str,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int = 200,
    best_condition: str = "loss"
) -> dict:
    """
    Train the model.

    Args:
        data_config_path: Path to data configuration file
        model_config: Model configuration dictionary
        device: Device to train on
        model: Model to train
        save_results_dir: Directory to save results
        train_loader: Training data loader
        valid_loader: Validation data loader
        epochs: Number of epochs to train
        best_condition: Best model selection criterion ('loss' or 'acc')

    Returns:
        Training results dictionary
    """
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config['training']['lr'],
        weight_decay=model_config['training']['weight-decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Calculate class weights for imbalanced data
    load_data_info = LoadDataInfo(data_config_path=data_config_path, data_class="train", printcheck=False)
    _, labels = load_data_info()
    n_fake = len([x for x in labels if x == 1])
    n_real = len([x for x in labels if x == 0])

    if n_fake > 0 and n_real > 0:
        bias_weight = torch.tensor([n_real / n_fake], device=device)
    else:
        bias_weight = torch.tensor([1.0], device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=bias_weight)
    early_stop = model_config['training']['early-stop']
    early_stop_count = 0

    best_loss = np.inf
    best_acc = 0
    best_epoch = 0
    total_epoch_train_loss, total_epoch_valid_loss = [], []
    total_epoch_train_acc, total_epoch_valid_acc = [], []

    results_template = {
        "loss": None, "n_data": None, "correct": None,
        "positive": None, "negative": None, "cm_total": None,
        "path_list": None, "label_list": None, "prob_list": None, "pred_list": None
    }

    # Setup save directory
    timestr = datetime.datetime.now().strftime('%y%m%d%H%M')
    base_save_dir = os.path.join(save_results_dir, f"train_{timestr}")
    os.makedirs(base_save_dir, exist_ok=True)

    # Save configs
    with open(data_config_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    with open(os.path.join(base_save_dir, "architecture.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(model_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    with open(os.path.join(base_save_dir, "data_config.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info(f"Training started. Results will be saved to: {base_save_dir}")
    logger.info(f"Total epochs: {epochs}, Early stop patience: {early_stop}")

    # Training loop
    actual_epochs = 0
    for epoch in range(epochs):
        actual_epochs = epoch + 1
        total_train_loss = 0.0
        total_val_loss = 0.0

        train_path_list, train_label_list, train_prob_list, train_pred_list = [], [], [], []
        results_info = {
            "train": results_template.copy(),
            "validation": results_template.copy()
        }

        train_correct, train_positive, train_negative = 0, 0, 0
        train_cm_total = np.zeros((2, 2))
        n_train_data = 0

        # Training phase
        model.train()
        for data in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
            images = data["input"].to(device=device, dtype=torch.float32)
            labels = torch.unsqueeze(data["label"], dim=1).to(device=device, dtype=torch.float32)
            paths = data["file_path"]

            pred_labels = model(images)
            loss = loss_fn(pred_labels, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            corrects, positive_class, negative_class, cm = check_correct(pred_labels, labels, classes=[0, 1])
            loss_value = loss.item()

            n_train_data += len(data["label"])
            train_label_list += labels.round().to(int).cpu().detach().numpy().squeeze(axis=1).tolist()
            pred_prob = torch.sigmoid(pred_labels)
            train_prob_list += pred_prob.to(torch.float).cpu().detach().numpy().squeeze(axis=1).tolist()
            train_pred_list += torch.sigmoid(pred_labels).round().to(int).cpu().detach().numpy().squeeze(axis=1).tolist()
            train_path_list += paths

            train_cm_total += cm
            train_correct += corrects
            train_positive += positive_class
            train_negative += negative_class
            total_train_loss += loss_value

        # Store training results
        results_info["train"] = {
            "loss": total_train_loss, "n_data": n_train_data, "correct": train_correct,
            "positive": train_positive, "negative": train_negative, "cm_total": train_cm_total,
            "path_list": train_path_list, "label_list": train_label_list,
            "prob_list": train_prob_list, "pred_list": train_pred_list
        }

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_path_list, val_label_list, val_prob_list, val_pred_list = [], [], [], []
            val_correct, val_positive, val_negative = 0, 0, 0
            val_cm_total = np.zeros((2, 2))
            n_val_data = 0

            for data in tqdm(valid_loader, desc=f"Epoch {epoch} - Validation"):
                images = data["input"].to(device=device, dtype=torch.float32)
                labels = torch.unsqueeze(data["label"], dim=1).to(device=device, dtype=torch.float32)
                paths = data["file_path"]

                pred_labels = model(images)
                loss = loss_fn(pred_labels, labels)

                corrects, positive_class, negative_class, cm = check_correct(pred_labels, labels, classes=[0, 1])
                loss_value = loss.item()

                n_val_data += len(data["label"])
                val_label_list += labels.round().to(int).cpu().detach().numpy().squeeze(axis=1).tolist()
                pred_prob = torch.sigmoid(pred_labels)
                val_prob_list += pred_prob.to(torch.float).cpu().detach().numpy().squeeze(axis=1).tolist()
                val_pred_list += torch.sigmoid(pred_labels).round().to(int).cpu().detach().numpy().squeeze(axis=1).tolist()
                val_path_list += paths

                val_cm_total += cm
                val_correct += corrects
                val_positive += positive_class
                val_negative += negative_class
                total_val_loss += loss_value

            # Store validation results
            results_info["validation"] = {
                "loss": total_val_loss, "n_data": n_val_data, "correct": val_correct,
                "positive": val_positive, "negative": val_negative, "cm_total": val_cm_total,
                "path_list": val_path_list, "label_list": val_label_list,
                "prob_list": val_prob_list, "pred_list": val_pred_list
            }

        scheduler.step(total_val_loss)

        # Calculate epoch metrics
        epoch_train_loss = total_train_loss / len(train_loader)
        epoch_train_acc = train_correct / n_train_data if n_train_data > 0 else 0
        total_epoch_train_acc.append(epoch_train_acc)
        total_epoch_train_loss.append(epoch_train_loss)

        epoch_valid_loss = total_val_loss / len(valid_loader)
        epoch_valid_acc = val_correct / n_val_data if n_val_data > 0 else 0
        total_epoch_valid_acc.append(epoch_valid_acc)
        total_epoch_valid_loss.append(epoch_valid_loss)

        logger.info(
            f"Epoch {epoch}: train_loss={epoch_train_loss:.4f}, train_acc={epoch_train_acc:.4f} | "
            f"val_loss={epoch_valid_loss:.4f}, val_acc={epoch_valid_acc:.4f}"
        )

        # Check for best model
        is_best = False
        if best_condition == "loss":
            if epoch_valid_loss < best_loss:
                best_loss = epoch_valid_loss
                best_epoch = epoch
                is_best = True
        elif best_condition == "acc":
            if epoch_valid_acc > best_acc:
                best_acc = epoch_valid_acc
                best_epoch = epoch
                is_best = True

        if is_best:
            epoch_save_dir = os.path.join(base_save_dir, "Epoch_BEST")
            os.makedirs(epoch_save_dir, exist_ok=True)
            epoch_model_save_dir = os.path.join(epoch_save_dir, "weight")
            os.makedirs(epoch_model_save_dir, exist_ok=True)
            epoch_model_weight_path = os.path.join(epoch_model_save_dir, "Epoch_BEST.pth")

            postprocess(results_info=results_info, save_dir=epoch_save_dir)
            torch.save(model.state_dict(), epoch_model_weight_path)
            logger.info(">>> Best model updated!")
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Save training curves
        _save_training_curves(
            base_save_dir, actual_epochs,
            total_epoch_train_acc, total_epoch_valid_acc,
            total_epoch_train_loss, total_epoch_valid_loss
        )

    # Save last epoch
    epoch_save_dir = os.path.join(base_save_dir, "Epoch_LAST")
    os.makedirs(epoch_save_dir, exist_ok=True)
    epoch_model_save_dir = os.path.join(epoch_save_dir, "weight")
    os.makedirs(epoch_model_save_dir, exist_ok=True)
    epoch_model_weight_path = os.path.join(epoch_model_save_dir, "Epoch_LAST.pth")

    postprocess(results_info=results_info, save_dir=epoch_save_dir)
    torch.save(model.state_dict(), epoch_model_weight_path)

    logger.info(f"Training completed. Best epoch: {best_epoch}")

    return {
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "best_acc": best_acc,
        "save_dir": base_save_dir
    }


def _save_training_curves(
    save_dir: str,
    epochs: int,
    train_acc: list,
    valid_acc: list,
    train_loss: list,
    valid_loss: list
):
    """Save training accuracy and loss curves."""
    train_df = pd.DataFrame({
        "accuracy": train_acc,
        "loss": train_loss,
        "data": "train",
        "epoch": list(range(epochs))
    })
    valid_df = pd.DataFrame({
        "accuracy": valid_acc,
        "loss": valid_loss,
        "data": "validation",
        "epoch": list(range(epochs))
    })
    combined_df = pd.concat([train_df, valid_df])

    # Save CSV
    combined_df.to_csv(os.path.join(save_dir, "acc_loss_info.csv"), encoding="utf-8-sig")

    # Accuracy plot
    plt.figure(figsize=(6, 5))
    sns.lineplot(data=combined_df, hue="data", x="epoch", y="accuracy")
    plt.title('Accuracy per Epoch')
    plt.savefig(os.path.join(save_dir, "acc_per_epoch_line.png"))
    plt.close()

    # Loss plot
    plt.figure(figsize=(6, 5))
    sns.lineplot(data=combined_df, hue="data", x="epoch", y="loss")
    plt.title('Loss per Epoch')
    plt.savefig(os.path.join(save_dir, "loss_per_epoch_line.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train deepfake classifier")
    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
        help="Path to data configuration YAML file"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model configuration YAML file"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save training results"
    )
    parser.add_argument(
        "--weight-path",
        type=str,
        default=None,
        help="Path to pretrained weights (optional)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--best-condition",
        type=str,
        default="loss",
        choices=["loss", "acc"],
        help="Criterion for best model selection"
    )
    args = parser.parse_args()

    # Load configs
    with open(args.data_config, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    with open(args.model_config, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load datasets
    batch_size = model_config["training"]["bs"]
    img_size = model_config["model"]["image-size"]

    train_dataset = CnnDataset(
        data_config_path=args.data_config,
        data_class="train",
        img_size=img_size,
        printcheck=True,
        enable_augmentation=True
    )
    valid_dataset = CnnDataset(
        data_config_path=args.data_config,
        data_class="validation",
        img_size=img_size,
        printcheck=True,
        enable_augmentation=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    num_classes = model_config["model"]["num-classes"]
    model_name = model_config["model"]["name"]

    model = create_classifier_model(model_name, num_classes=num_classes, pretrained=True)

    if args.weight_path is not None:
        state_dict = torch.load(args.weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded weights from: {args.weight_path}")

    model.to(device)

    # Train
    result = train(
        data_config_path=args.data_config,
        model_config=model_config,
        device=device,
        model=model,
        save_results_dir=args.save_dir,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=args.epochs,
        best_condition=args.best_condition
    )

    logger.info(f"Training completed. Results saved to: {result['save_dir']}")


if __name__ == "__main__":
    main()
