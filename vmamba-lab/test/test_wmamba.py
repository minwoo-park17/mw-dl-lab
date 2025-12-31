"""WMamba test script for face forgery detection."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import FFPPDataset, CelebDFDataset, DFDCDataset, get_test_transforms
from model import WMamba
from test.evaluator import ClassificationEvaluator
from utils.checkpoint import load_checkpoint
from utils.visualize import plot_roc_curve, plot_confusion_matrix


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_test_dataloaders(
    config: Dict[str, Any],
    data_config: Dict[str, Any],
) -> Dict[str, DataLoader]:
    """Create test data loaders for cross-dataset evaluation.

    Args:
        config: Main configuration
        data_config: Data configuration

    Returns:
        Dictionary of dataset names to dataloaders
    """
    data_conf = config.get("data", {})
    input_size = data_conf.get("input_size", 256)
    batch_size = config.get("eval", {}).get("batch_size", 32)
    num_workers = config.get("eval", {}).get("num_workers", 4)

    transform = get_test_transforms(input_size=input_size)

    dataloaders = {}

    # FF++ test set
    ff_config = data_config.get("face_forgery", {}).get("ff++", {})
    if ff_config.get("root"):
        ff_dataset = FFPPDataset(
            root=ff_config["root"],
            split="test",
            transform=transform,
            compression="c23",
        )
        if len(ff_dataset) > 0:
            dataloaders["FF++_c23"] = DataLoader(
                ff_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

    # Celeb-DF
    cdf_config = data_config.get("face_forgery", {}).get("celeb-df", {})
    if cdf_config.get("root"):
        cdf_dataset = CelebDFDataset(
            root=cdf_config["root"],
            split="test",
            transform=transform,
            version="v2",
        )
        if len(cdf_dataset) > 0:
            dataloaders["Celeb-DF"] = DataLoader(
                cdf_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

    # DFDC
    dfdc_config = data_config.get("face_forgery", {}).get("dfdc", {})
    if dfdc_config.get("root"):
        dfdc_dataset = DFDCDataset(
            root=dfdc_config["root"],
            split="test",
            transform=transform,
            subset="preview",
        )
        if len(dfdc_dataset) > 0:
            dataloaders["DFDC"] = DataLoader(
                dfdc_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

    return dataloaders


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test WMamba for face forgery detection")
    parser.add_argument(
        "--config",
        type=str,
        default="config/wmamba_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="data/data_path.yaml",
        help="Path to data config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/test_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save individual predictions",
    )
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    data_config = load_config(args.data_config)

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    print("Loading model...")
    model = WMamba(config)

    # Load checkpoint
    load_checkpoint(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Create evaluator
    evaluator = ClassificationEvaluator(model, device=device)

    # Create test dataloaders
    print("\nCreating test dataloaders...")
    dataloaders = create_test_dataloaders(config, data_config)

    if not dataloaders:
        print("No test datasets found. Please check data_path.yaml")
        return

    # Evaluate on all datasets
    print("\n" + "=" * 60)
    print("Cross-Dataset Evaluation Results")
    print("=" * 60)

    all_results = evaluator.evaluate_cross_dataset(dataloaders)

    # Print summary table
    print("\n" + "-" * 60)
    print(f"{'Dataset':<20} {'AUC':>10} {'ACC':>10} {'EER':>10} {'AP':>10}")
    print("-" * 60)

    for name, metrics in all_results.items():
        print(
            f"{name:<20} "
            f"{metrics.get('auc', 0):.4f}     "
            f"{metrics.get('accuracy', 0):.4f}     "
            f"{metrics.get('eer', 0):.4f}     "
            f"{metrics.get('ap', 0):.4f}"
        )

    print("-" * 60)

    # Save results
    import json
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    for name, metrics in all_results.items():
        # Plot ROC curve
        if args.save_predictions:
            results = evaluator.evaluate(dataloaders[name], return_predictions=True)

            y_true = [p["label"] for p in results["predictions"]]
            y_scores = [p["score"] for p in results["predictions"]]

            plot_roc_curve(
                y_true, y_scores,
                save_path=str(output_dir / f"roc_{name}.png"),
                title=f"ROC Curve - {name}",
            )

            y_pred = [p["prediction"] for p in results["predictions"]]
            plot_confusion_matrix(
                y_true, y_pred,
                class_names=["Real", "Fake"],
                save_path=str(output_dir / f"confusion_{name}.png"),
                title=f"Confusion Matrix - {name}",
            )

    print(f"Visualizations saved to {output_dir}")
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
