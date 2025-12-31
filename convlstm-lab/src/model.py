"""
ConvLSTM 기반 생성형 AI 탐지 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from typing import Dict, List, Tuple, Optional, Any


class ConvLSTMCell(nn.Module):
    """
    단일 ConvLSTM 셀

    Convolutional LSTM:
    - 기존 LSTM의 행렬 곱셈을 컨볼루션으로 대체
    - 공간적 구조를 유지하면서 시간적 의존성 학습

    Reference: https://arxiv.org/abs/1506.04214
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True
    ):
        """
        Args:
            input_channels: 입력 채널 수
            hidden_channels: hidden state 채널 수
            kernel_size: 컨볼루션 커널 크기
            bias: bias 사용 여부
        """
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # 4개의 게이트 (i, f, o, g)를 한번에 계산
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 입력 텐서 (B, C, H, W)
            hidden_state: (h, c) 튜플, 각각 (B, hidden_channels, H, W)

        Returns:
            h_next: 다음 hidden state (B, hidden_channels, H, W)
            c_next: 다음 cell state (B, hidden_channels, H, W)
        """
        batch_size, _, height, width = x.size()

        # hidden state 초기화
        if hidden_state is None:
            h = torch.zeros(batch_size, self.hidden_channels, height, width,
                          device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_channels, height, width,
                          device=x.device, dtype=x.dtype)
        else:
            h, c = hidden_state

        # 입력과 hidden state 결합
        combined = torch.cat([x, h], dim=1)

        # 게이트 계산
        gates = self.conv(combined)

        # 4개의 게이트로 분리
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        o = torch.sigmoid(o)  # output gate
        g = torch.tanh(g)     # cell gate

        # cell state 및 hidden state 업데이트
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    다층 ConvLSTM 모듈

    여러 ConvLSTMCell을 스택하여 깊은 시공간 특징 학습
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
        num_layers: int = 1,
        batch_first: bool = True,
        bidirectional: bool = False,
        return_all_layers: bool = False
    ):
        """
        Args:
            input_channels: 입력 채널 수
            hidden_channels: 각 레이어의 hidden 채널 수 리스트
            kernel_size: 커널 크기
            num_layers: 레이어 수
            batch_first: True면 입력이 (B, T, C, H, W)
            bidirectional: 양방향 LSTM 사용 여부
            return_all_layers: 모든 레이어의 출력 반환 여부
        """
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.return_all_layers = return_all_layers

        # hidden_channels가 리스트가 아니면 리스트로 변환
        if not isinstance(hidden_channels, list):
            hidden_channels = [hidden_channels] * num_layers

        assert len(hidden_channels) == num_layers

        # Forward 방향 레이어들
        cell_list = []
        for i in range(num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels[i - 1]
            cell_list.append(ConvLSTMCell(
                input_channels=cur_input_channels,
                hidden_channels=hidden_channels[i],
                kernel_size=kernel_size
            ))
        self.cell_list = nn.ModuleList(cell_list)

        # Backward 방향 레이어들 (bidirectional인 경우)
        if bidirectional:
            cell_list_backward = []
            for i in range(num_layers):
                cur_input_channels = input_channels if i == 0 else hidden_channels[i - 1]
                cell_list_backward.append(ConvLSTMCell(
                    input_channels=cur_input_channels,
                    hidden_channels=hidden_channels[i],
                    kernel_size=kernel_size
                ))
            self.cell_list_backward = nn.ModuleList(cell_list_backward)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: 입력 시퀀스 (B, T, C, H, W) if batch_first else (T, B, C, H, W)
            hidden_state: 초기 hidden state 리스트

        Returns:
            output: 마지막 레이어의 출력 시퀀스
            last_states: 각 레이어의 마지막 (h, c) 리스트
        """
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)  # (T, B, C, H, W) -> (B, T, C, H, W)

        batch_size, seq_len, _, height, width = x.size()

        # hidden state 초기화
        if hidden_state is None:
            hidden_state = [None] * self.num_layers

        # Forward 방향 처리
        layer_output_list = []
        last_state_list = []

        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h, c = None, None
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    cur_layer_input[:, t, :, :, :],
                    (h, c) if h is not None else None
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        # Bidirectional 처리
        if self.bidirectional:
            cur_layer_input = x

            for layer_idx in range(self.num_layers):
                h, c = None, None
                output_inner = []

                # 역방향 순회
                for t in range(seq_len - 1, -1, -1):
                    h, c = self.cell_list_backward[layer_idx](
                        cur_layer_input[:, t, :, :, :],
                        (h, c) if h is not None else None
                    )
                    output_inner.insert(0, h)

                layer_output_backward = torch.stack(output_inner, dim=1)

                # Forward와 Backward 결합
                layer_output_list[layer_idx] = torch.cat(
                    [layer_output_list[layer_idx], layer_output_backward], dim=2
                )
                cur_layer_input = layer_output_backward

        if self.return_all_layers:
            return layer_output_list, last_state_list
        else:
            return layer_output_list[-1], last_state_list[-1]


class CNNEncoder(nn.Module):
    """
    CNN 기반 특징 추출기

    프레임별로 공간적 특징을 추출
    """

    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Args:
            backbone: 백본 네트워크 ('resnet18', 'resnet34', 'efficientnet_b0')
            pretrained: 사전학습 가중치 사용 여부
            freeze_backbone: 백본 가중치 고정 여부
        """
        super().__init__()

        self.backbone_name = backbone

        if backbone.startswith('resnet'):
            if backbone == 'resnet18':
                self.encoder = models.resnet18(
                    weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                )
                self.feature_channels = 512
            elif backbone == 'resnet34':
                self.encoder = models.resnet34(
                    weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
                )
                self.feature_channels = 512
            else:
                raise ValueError(f"Unsupported ResNet variant: {backbone}")

            # 마지막 FC 레이어 제거
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        elif backbone.startswith('efficientnet'):
            self.encoder = timm.create_model(
                backbone,
                pretrained=pretrained,
                features_only=True,
                out_indices=[-1]
            )
            # EfficientNet 출력 채널 수 확인
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                out = self.encoder(dummy)
                self.feature_channels = out[0].shape[1]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 이미지 (B, C, H, W)

        Returns:
            features: 특징 맵 (B, feature_channels, H', W')
        """
        if self.backbone_name.startswith('efficientnet'):
            return self.encoder(x)[0]
        return self.encoder(x)


class ConvLSTMClassifier(nn.Module):
    """
    ConvLSTM 기반 이진 분류 모델

    전체 파이프라인:
    1. CNN Encoder: 각 프레임에서 공간적 특징 추출
    2. ConvLSTM: 시간적 패턴 학습
    3. Classifier Head: 이진 분류

    입력: (B, T, C, H, W) - 배치 x 시퀀스 x 채널 x 높이 x 너비
    출력: (B, num_classes) - 분류 로짓
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 모델 설정 딕셔너리
        """
        super().__init__()

        model_config = config.get('model', {})

        # CNN Encoder
        backbone = model_config.get('backbone', 'resnet18')
        pretrained = model_config.get('pretrained', True)
        freeze_backbone = model_config.get('freeze_backbone', False)

        self.encoder = CNNEncoder(
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

        # ConvLSTM
        convlstm_config = model_config.get('convlstm', {})
        hidden_channels = convlstm_config.get('hidden_channels', [64, 128])
        kernel_size = convlstm_config.get('kernel_size', 3)
        num_layers = convlstm_config.get('num_layers', 2)
        bidirectional = convlstm_config.get('bidirectional', False)

        self.convlstm = ConvLSTM(
            input_channels=self.encoder.feature_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # 출력 채널 수 계산
        lstm_out_channels = hidden_channels[-1]
        if bidirectional:
            lstm_out_channels *= 2

        # Classifier Head
        classifier_config = model_config.get('classifier', {})
        dropout = classifier_config.get('dropout', 0.5)
        num_classes = classifier_config.get('num_classes', 2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 시퀀스 (B, T, C, H, W)

        Returns:
            logits: 분류 로짓 (B, num_classes)
        """
        batch_size, seq_len, c, h, w = x.size()

        # 프레임별 특징 추출: (B*T, C, H, W) -> (B*T, F, H', W')
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.encoder(x)

        # (B*T, F, H', W') -> (B, T, F, H', W')
        _, f, h_feat, w_feat = features.size()
        features = features.view(batch_size, seq_len, f, h_feat, w_feat)

        # ConvLSTM으로 시간적 특징 학습
        lstm_out, (h_last, _) = self.convlstm(features)

        # 마지막 hidden state 사용
        # h_last: (B, hidden_channels, H', W')
        pooled = self.global_pool(h_last)  # (B, hidden_channels, 1, 1)
        pooled = pooled.view(batch_size, -1)  # (B, hidden_channels)

        # 분류
        logits = self.classifier(pooled)

        return logits

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        시각화를 위한 attention map 추출 (선택적)
        """
        batch_size, seq_len, c, h, w = x.size()

        x = x.view(batch_size * seq_len, c, h, w)
        features = self.encoder(x)

        _, f, h_feat, w_feat = features.size()
        features = features.view(batch_size, seq_len, f, h_feat, w_feat)

        # ConvLSTM 출력 시퀀스 전체
        lstm_out, _ = self.convlstm(features)

        # 채널 방향 평균으로 attention map 생성
        attention = lstm_out.mean(dim=2)  # (B, T, H', W')

        return attention


def build_model(config: Dict[str, Any]) -> ConvLSTMClassifier:
    """
    config로부터 모델 생성

    Args:
        config: 전체 config 딕셔너리

    Returns:
        ConvLSTMClassifier 인스턴스
    """
    model = ConvLSTMClassifier(config)

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model created: {model.__class__.__name__}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model
