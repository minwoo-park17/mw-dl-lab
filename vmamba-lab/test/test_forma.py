"""ForMa test script for image tampering localization."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import (
    CASIADataset,
    ColumbiaDataset,
    CoverageDataset,
    NIST16Dataset,
    get_segmentation_transforms,
)
from model import ForMa
from test.evaluator import SegmentationEvaluator
from utils.checkpoint import load_checkpoint
from utils.visualize import visualize_segmentation


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_test_dataloaders(
    config: Dict[str, Any],
    data_config: Dict[str, Any],
) -> Dict[str, DataLoader]:
    """Create test data loaders for multi-dataset evaluation.

    Args:
        config: Main configuration
        data_config: Data configuration

    Returns:
        Dictionary of dataset names to dataloaders
    """
    data_conf = config.get("data", {})
    input_size = data_conf.get("input_size", 512)
    batch_size = config.get("eval", {}).get("batch_size", 16)
    num_workers = config.get("eval", {}).get("num_workers", 4)

    transform = get_segmentation_transforms(input_size=input_size, split="test")

    dataloaders = {}
    tampering_config = data_config.get("tampering", {})

    # CASIA v1
    casia_v1_config = tampering_config.get("casia", {}).get("v1", {})
    casia_root = tampering_config.get("casia", {}).get("root", "")
    if casia_root and Path(casia_root).exists():
        try:
            casia_v1_dataset = CASIADataset(
                root=casia_root,
                split="test",
                transform=transform,
                version="v1",
            )
            if len(casia_v1_dataset) > 0:
                dataloaders["CASIA_v1"] = DataLoader(
                    casia_v1_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
        except Exception as e:
            print(f"Warning: Could not load CASIA v1: {e}")

    # CASIA v2
    if casia_root and Path(casia_root).exists():
        try:
            casia_v2_dataset = CASIADataset(
                root=casia_root,
                split="test",
                transform=transform,
                version="v2",
            )
            if len(casia_v2_dataset) > 0:
                dataloaders["CASIA_v2"] = DataLoader(
                    casia_v2_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
        except Exception as e:
            print(f"Warning: Could not load CASIA v2: {e}")

    # Columbia
    columbia_config = tampering_config.get("columbia", {})
    if columbia_config.get("root") and Path(columbia_config["root"]).exists():
        try:
            columbia_dataset = ColumbiaDataset(
                root=columbia_config["root"],
                split="test",
                transform=transform,
            )
            if len(columbia_dataset) > 0:
                dataloaders["Columbia"] = DataLoader(
                    columbia_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
        except Exception as e:
            print(f"Warning: Could not load Columbia: {e}")

    # Coverage
    coverage_config = tampering_config.get("coverage", {})
    if coverage_config.get("root") and Path(coverage_config["root"]).exists():
        try:
            coverage_dataset = CoverageDataset(
                root=coverage_config["root"],
                split="test",
                transform=transform,
            )
            if len(coverage_dataset) > 0:
                dataloaders["Coverage"] = DataLoader(
                    coverage_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
        except Exception as e:
            print(f"Warning: Could not load Coverage: {e}")

    # NIST16
    nist_config = tampering_config.get("nist16", {})
    if nist_config.get("root") and Path(nist_config["root"]).exists():
        try:
            nist_dataset = NIST16Dataset(
                root=nist_config["root"],
                split="test",
                transform=transform,
            )
            if len(nist_dataset) > 0:
                dataloaders["NIST16"] = DataLoader(
                    nist_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
        except Exception as e:
            print(f"Warning: Could not load NIST16: {e}")

    return dataloaders


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test ForMa for image tampering localization")
    parser.add_argument(
        "--config",
        type=str,
        default="config/forma_config.yaml",
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
        "--save-masks",
        action="store_true",
        help="Save prediction masks",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations",
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=20,
        help="Number of samples to visualize",
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
    model = ForMa(config)

    # Load checkpoint
    load_checkpoint(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Create evaluator
    threshold = config.get("eval", {}).get("threshold", 0.5)
    evaluator = SegmentationEvaluator(model, device=device, threshold=threshold)

    # Create test dataloaders
    print("\nCreating test dataloaders...")
    dataloaders = create_test_dataloaders(config, data_config)

    if not dataloaders:
        print("No test datasets found. Please check data_path.yaml")
        return

    # Evaluate on all datasets
    print("\n" + "=" * 70)
    print("Multi-Dataset Evaluation Results")
    print("=" * 70)

    all_results = {}

    for name, dataloader in dataloaders.items():
        print(f"\nEvaluating on {name}...")

        # Create save directory for masks
        save_dir = None
        if args.save_masks:
            save_dir = str(output_dir / "masks" / name)

        results = evaluator.evaluate(
            dataloader,
            return_predictions=args.visualize,
            save_dir=save_dir,
        )
        all_results[name] = results

        print(f"  F1: {results['f1']:.4f}")
        print(f"  IoU: {results['iou']:.4f}")
        print(f"  Dice: {results['dice']:.4f}")
        if "pixel_auc" in results:
            print(f"  Pixel AUC: {results['pixel_auc']:.4f}")

    # Print summary table
    print("\n" + "-" * 70)
    print(f"{'Dataset':<15} {'F1':>10} {'IoU':>10} {'Dice':>10} {'Pixel AUC':>12}")
    print("-" * 70)

    for name, metrics in all_results.items():
        print(
            f"{name:<15} "
            f"{metrics.get('f1', 0):.4f}     "
            f"{metrics.get('iou', 0):.4f}     "
            f"{metrics.get('dice', 0):.4f}     "
            f"{metrics.get('pixel_auc', 0):.4f}"
        )

    print("-" * 70)

    # Calculate average
    avg_f1 = np.mean([m["f1"] for m in all_results.values()])
    avg_iou = np.mean([m["iou"] for m in all_results.values()])
    print(f"\nAverage F1: {avg_f1:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")

    # Save results
    import json

    # Remove non-serializable items
    results_to_save = {}
    for name, metrics in all_results.items():
        results_to_save[name] = {k: v for k, v in metrics.items() if k != "predictions"}

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        for name, results in all_results.items():
            if "predictions" not in results:
                continue

            dataset_vis_dir = vis_dir / name
            dataset_vis_dir.mkdir(exist_ok=True)

            predictions = results["predictions"][:args.num_vis]

            for i, pred in enumerate(predictions):
                try:
                    visualize_segmentation(
                        np.zeros((256, 256, 3)),  # Placeholder image
                        pred["pred_mask"],
                        pred["gt_mask"],
                        save_path=str(dataset_vis_dir / f"sample_{i:04d}.png"),
                    )
                except Exception as e:
                    print(f"Warning: Could not visualize sample {i}: {e}")

        print(f"Visualizations saved to {vis_dir}")

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
