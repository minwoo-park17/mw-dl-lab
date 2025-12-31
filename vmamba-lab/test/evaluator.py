"""Evaluator classes for model evaluation."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import ClassificationMetrics, SegmentationMetrics


class BaseEvaluator(ABC):
    """Abstract base evaluator class."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ):
        """Initialize evaluator.

        Args:
            model: Model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @abstractmethod
    def evaluate(
        self,
        dataloader: DataLoader,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate model on dataloader. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def evaluate_single(
        self,
        image: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate model on single image. Must be implemented by subclasses."""
        pass


class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for classification models (WMamba)."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
    ):
        """Initialize classification evaluator.

        Args:
            model: Classification model
            device: Device to run evaluation on
            threshold: Classification threshold
        """
        super().__init__(model, device)
        self.threshold = threshold
        self.metrics = ClassificationMetrics()

    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate model on dataloader.

        Args:
            dataloader: DataLoader with test data
            return_predictions: Return all predictions

        Returns:
            Dictionary with evaluation results
        """
        self.metrics.reset()
        all_predictions = [] if return_predictions else None

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                images = batch["image"].to(self.device)
                labels = batch["label"].numpy()

                output = self.model(images)
                probs = output["probs"].cpu().numpy()
                preds = probs.argmax(axis=-1)
                scores = probs[:, 1]

                self.metrics.update(labels, preds, scores)

                if return_predictions:
                    paths = batch.get("path", [""] * len(images))
                    for i in range(len(images)):
                        all_predictions.append({
                            "path": paths[i],
                            "label": int(labels[i]),
                            "prediction": int(preds[i]),
                            "score": float(scores[i]),
                        })

        results = self.metrics.compute()

        if return_predictions:
            results["predictions"] = all_predictions

        return results

    def evaluate_single(
        self,
        image: torch.Tensor,
    ) -> Dict[str, Any]:
        """Evaluate model on single image.

        Args:
            image: Input image tensor (C, H, W) or (1, C, H, W)

        Returns:
            Dictionary with prediction results
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            probs = output["probs"].cpu().numpy()[0]
            pred = int(probs.argmax())
            score = float(probs[1])

        return {
            "prediction": pred,
            "label": "fake" if pred == 1 else "real",
            "score": score,
            "confidence": float(max(probs)),
        }

    def evaluate_cross_dataset(
        self,
        dataloaders: Dict[str, DataLoader],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model on multiple datasets.

        Args:
            dataloaders: Dictionary of dataset names to dataloaders

        Returns:
            Dictionary of dataset names to metrics
        """
        results = {}

        for name, dataloader in dataloaders.items():
            print(f"\nEvaluating on {name}...")
            results[name] = self.evaluate(dataloader)
            print(f"  AUC: {results[name]['auc']:.4f}")
            print(f"  ACC: {results[name]['accuracy']:.4f}")
            print(f"  EER: {results[name]['eer']:.4f}")

        return results


class SegmentationEvaluator(BaseEvaluator):
    """Evaluator for segmentation models (ForMa)."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
    ):
        """Initialize segmentation evaluator.

        Args:
            model: Segmentation model
            device: Device to run evaluation on
            threshold: Segmentation threshold
        """
        super().__init__(model, device)
        self.threshold = threshold
        self.metrics = SegmentationMetrics(threshold=threshold)

    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate model on dataloader.

        Args:
            dataloader: DataLoader with test data
            return_predictions: Return all predictions
            save_dir: Directory to save prediction masks

        Returns:
            Dictionary with evaluation results
        """
        self.metrics.reset()
        all_predictions = [] if return_predictions else None

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                images = batch["image"].to(self.device)
                masks = batch["mask"].numpy()

                output = self.model(images)
                pred_probs = output["mask"].cpu().numpy()
                pred_masks = (pred_probs > self.threshold).astype(np.float32)

                # Update metrics
                for i in range(len(images)):
                    self.metrics.update(
                        pred_masks[i],
                        masks[i],
                        pred_probs[i],
                    )

                    if return_predictions:
                        paths = batch.get("path", [""] * len(images))
                        all_predictions.append({
                            "path": paths[i],
                            "pred_mask": pred_masks[i],
                            "gt_mask": masks[i],
                        })

                    if save_dir:
                        # Save prediction mask
                        import cv2
                        mask_img = (pred_probs[i].squeeze() * 255).astype(np.uint8)
                        cv2.imwrite(
                            str(save_path / f"pred_{batch_idx * len(images) + i:06d}.png"),
                            mask_img,
                        )

        results = self.metrics.compute()

        if return_predictions:
            results["predictions"] = all_predictions

        return results

    def evaluate_single(
        self,
        image: torch.Tensor,
    ) -> Dict[str, Any]:
        """Evaluate model on single image.

        Args:
            image: Input image tensor (C, H, W) or (1, C, H, W)

        Returns:
            Dictionary with prediction results
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            pred_prob = output["mask"].cpu().numpy()[0]
            pred_mask = (pred_prob > self.threshold).astype(np.float32)

        # Compute tampering ratio
        tampering_ratio = float(pred_mask.sum() / pred_mask.size)

        return {
            "pred_mask": pred_mask,
            "pred_prob": pred_prob,
            "is_tampered": tampering_ratio > 0.01,
            "tampering_ratio": tampering_ratio,
        }

    def evaluate_multi_dataset(
        self,
        dataloaders: Dict[str, DataLoader],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model on multiple datasets.

        Args:
            dataloaders: Dictionary of dataset names to dataloaders

        Returns:
            Dictionary of dataset names to metrics
        """
        results = {}

        for name, dataloader in dataloaders.items():
            print(f"\nEvaluating on {name}...")
            results[name] = self.evaluate(dataloader)
            print(f"  F1: {results[name]['f1']:.4f}")
            print(f"  IoU: {results[name]['iou']:.4f}")
            if "pixel_auc" in results[name]:
                print(f"  Pixel AUC: {results[name]['pixel_auc']:.4f}")

        return results
