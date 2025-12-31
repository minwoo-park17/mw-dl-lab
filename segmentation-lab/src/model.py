"""
Segmentation Model Module
다양한 segmentation 모델을 유연하게 선택하고 사용할 수 있는 팩토리 패턴 구현
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class BaseSegmentationModel(ABC):
    """Segmentation 모델의 기본 인터페이스"""

    @abstractmethod
    def get_model(self) -> nn.Module:
        """모델 인스턴스를 반환합니다."""
        pass


class UNetModel(BaseSegmentationModel):
    """U-Net 모델"""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1
    ):
        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.num_classes = num_classes

    def get_model(self) -> nn.Module:
        return smp.Unet(
            encoder_name=self.encoder_name,
            encoder_weights="imagenet" if self.pretrained else None,
            in_channels=self.in_channels,
            classes=self.num_classes,
            activation=None  # logits 출력
        )


class UNetPlusPlusModel(BaseSegmentationModel):
    """U-Net++ 모델"""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1
    ):
        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.num_classes = num_classes

    def get_model(self) -> nn.Module:
        return smp.UnetPlusPlus(
            encoder_name=self.encoder_name,
            encoder_weights="imagenet" if self.pretrained else None,
            in_channels=self.in_channels,
            classes=self.num_classes,
            activation=None
        )


class DeepLabV3Model(BaseSegmentationModel):
    """DeepLabV3 모델"""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1
    ):
        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.num_classes = num_classes

    def get_model(self) -> nn.Module:
        return smp.DeepLabV3(
            encoder_name=self.encoder_name,
            encoder_weights="imagenet" if self.pretrained else None,
            in_channels=self.in_channels,
            classes=self.num_classes,
            activation=None
        )


class DeepLabV3PlusModel(BaseSegmentationModel):
    """DeepLabV3+ 모델"""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1
    ):
        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.num_classes = num_classes

    def get_model(self) -> nn.Module:
        return smp.DeepLabV3Plus(
            encoder_name=self.encoder_name,
            encoder_weights="imagenet" if self.pretrained else None,
            in_channels=self.in_channels,
            classes=self.num_classes,
            activation=None
        )


class FPNModel(BaseSegmentationModel):
    """Feature Pyramid Network 모델"""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1
    ):
        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.num_classes = num_classes

    def get_model(self) -> nn.Module:
        return smp.FPN(
            encoder_name=self.encoder_name,
            encoder_weights="imagenet" if self.pretrained else None,
            in_channels=self.in_channels,
            classes=self.num_classes,
            activation=None
        )


class PSPNetModel(BaseSegmentationModel):
    """PSPNet 모델"""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1
    ):
        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.num_classes = num_classes

    def get_model(self) -> nn.Module:
        return smp.PSPNet(
            encoder_name=self.encoder_name,
            encoder_weights="imagenet" if self.pretrained else None,
            in_channels=self.in_channels,
            classes=self.num_classes,
            activation=None
        )


class MANetModel(BaseSegmentationModel):
    """MANet 모델"""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1
    ):
        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.num_classes = num_classes

    def get_model(self) -> nn.Module:
        return smp.MAnet(
            encoder_name=self.encoder_name,
            encoder_weights="imagenet" if self.pretrained else None,
            in_channels=self.in_channels,
            classes=self.num_classes,
            activation=None
        )


class ModelFactory:
    """모델 생성을 위한 팩토리 클래스"""

    _models: Dict[str, Type[BaseSegmentationModel]] = {
        "unet": UNetModel,
        "unet++": UNetPlusPlusModel,
        "unetplusplus": UNetPlusPlusModel,
        "deeplabv3": DeepLabV3Model,
        "deeplabv3+": DeepLabV3PlusModel,
        "deeplabv3plus": DeepLabV3PlusModel,
        "fpn": FPNModel,
        "pspnet": PSPNetModel,
        "manet": MANetModel,
    }

    @classmethod
    def register(cls, name: str, model_class: Type[BaseSegmentationModel]):
        """새로운 모델을 등록합니다."""
        cls._models[name.lower()] = model_class

    @classmethod
    def get_available_models(cls) -> list:
        """사용 가능한 모델 목록을 반환합니다."""
        return list(set(cls._models.values()))

    @classmethod
    def get_model_names(cls) -> list:
        """등록된 모델 이름 목록을 반환합니다."""
        return list(cls._models.keys())

    @classmethod
    def create(
        cls,
        model_name: str,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1
    ) -> nn.Module:
        """
        설정에 따라 모델을 생성합니다.

        Args:
            model_name: 모델 이름 (unet, deeplabv3, fpn 등)
            encoder_name: 인코더 이름 (resnet34, efficientnet-b0 등)
            pretrained: 사전 학습된 가중치 사용 여부
            in_channels: 입력 채널 수
            num_classes: 출력 클래스 수

        Returns:
            생성된 PyTorch 모델
        """
        model_name = model_name.lower()
        if model_name not in cls._models:
            available = ", ".join(cls.get_model_names())
            raise ValueError(
                f"Unknown model: {model_name}. Available models: {available}"
            )

        model_builder = cls._models[model_name](
            encoder_name=encoder_name,
            pretrained=pretrained,
            in_channels=in_channels,
            num_classes=num_classes
        )
        return model_builder.get_model()


def create_model(config: dict) -> nn.Module:
    """
    설정 딕셔너리에서 모델을 생성합니다.

    Args:
        config: 전체 설정 딕셔너리

    Returns:
        생성된 PyTorch 모델
    """
    model_config = config['model']

    model = ModelFactory.create(
        model_name=model_config['name'],
        encoder_name=model_config.get('encoder', 'resnet34'),
        pretrained=model_config.get('pretrained', True),
        in_channels=model_config.get('in_channels', 3),
        num_classes=model_config.get('num_classes', 1)
    )

    return model


def get_encoder_list() -> list:
    """사용 가능한 인코더 목록을 반환합니다."""
    return smp.encoders.get_encoder_names()


def print_model_info(model: nn.Module):
    """모델 정보를 출력합니다."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
