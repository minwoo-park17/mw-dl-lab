"""
Segmentation Lab
Binary Segmentation 모델 학습을 위한 프레임워크
"""

from .dataset import (
    BinarySegmentationDataset,
    SegmentationDataset,  # alias for BinarySegmentationDataset
    InferenceDataset,
    create_dataloaders,
    create_inference_dataset
)

from .model import (
    ModelFactory,
    create_model,
    get_encoder_list,
    print_model_info
)

from .train import Trainer, train

from .validation import (
    evaluate,
    predict,
    predict_single,
    calculate_metrics,
    find_best_threshold
)

from .augmentation import (
    get_train_transform,
    get_val_transform,
    get_test_transform,
    get_inference_transform,
    get_heavy_train_transform,
    get_light_train_transform
)

from .utils import (
    load_config,
    save_config,
    setup_device,
    create_optimizer,
    create_scheduler,
    create_loss_function,
    set_seed,
    count_parameters
)

__version__ = "0.2.0"
__all__ = [
    # Dataset
    "BinarySegmentationDataset",
    "SegmentationDataset",
    "InferenceDataset",
    "create_dataloaders",
    "create_inference_dataset",
    # Model
    "ModelFactory",
    "create_model",
    "get_encoder_list",
    "print_model_info",
    # Training
    "Trainer",
    "train",
    # Validation
    "evaluate",
    "predict",
    "predict_single",
    "calculate_metrics",
    "find_best_threshold",
    # Augmentation
    "get_train_transform",
    "get_val_transform",
    "get_test_transform",
    "get_inference_transform",
    "get_heavy_train_transform",
    "get_light_train_transform",
    # Utils
    "load_config",
    "save_config",
    "setup_device",
    "create_optimizer",
    "create_scheduler",
    "create_loss_function",
    "set_seed",
    "count_parameters",
]
