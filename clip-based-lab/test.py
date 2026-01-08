"""
CLIP 기반 AI 생성 이미지 탐지 모델
- 기본 CLIP + Linear Probe
- UnivFD (CVPR 2023) 스타일
- NPR (CVPR 2024) 스타일
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


# =============================================================================
# 1. 기본 CLIP + Linear Probe
# =============================================================================
class CLIPLinearProbe(nn.Module):
    """
    가장 기본적인 CLIP 기반 탐지기
    CLIP ViT Encoder (frozen) → [CLS] token → Linear Layer → Sigmoid
    """
    def __init__(
        self,
        clip_model: str = "ViT-L/14",
        pretrained: str = "openai",
        num_classes: int = 1,
        freeze_encoder: bool = True
    ):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        
        # CLIP 모델 로드
        if OPEN_CLIP_AVAILABLE:
            self.encoder, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_model.replace("/", "-"), pretrained=pretrained
            )
            self.feature_dim = self.encoder.visual.output_dim
        elif CLIP_AVAILABLE:
            self.encoder, self.preprocess = clip.load(clip_model)
            self.feature_dim = self.encoder.visual.output_dim
        else:
            raise ImportError("open_clip 또는 clip 패키지를 설치하세요")
        
        # Encoder freeze
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        
        # Linear classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature 추출
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder.encode_image(x)
        else:
            features = self.encoder.encode_image(x)
        
        # float32로 변환 (CLIP은 fp16 사용할 수 있음)
        features = features.float()
        
        # Classification
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """특징 벡터만 추출"""
        with torch.no_grad():
            features = self.encoder.encode_image(x)
        return features.float()


# =============================================================================
# 2. UnivFD 스타일 (CVPR 2023)
# =============================================================================
class UnivFD(nn.Module):
    """
    UnivFD: Universal Fake Detector
    - CLIP ViT-L/14 frozen encoder
    - 단순 linear classifier
    - 핵심: data augmentation (blur, JPEG compression)이 학습 시 중요
    """
    def __init__(
        self,
        clip_model: str = "ViT-L-14",
        pretrained: str = "openai",
        num_classes: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # CLIP 모델 로드 (open_clip 사용 권장)
        if OPEN_CLIP_AVAILABLE:
            self.encoder, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_model, pretrained=pretrained
            )
            self.feature_dim = self.encoder.visual.output_dim
        else:
            raise ImportError("UnivFD는 open_clip 패키지가 필요합니다")
        
        # Encoder 완전 freeze
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        # Classifier head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.encoder.encode_image(x)
        features = features.float()
        return self.head(features)
    
    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_image(x).float()


class UnivFDAugmentation:
    """UnivFD 학습용 Augmentation"""
    def __init__(
        self,
        blur_prob: float = 0.5,
        jpeg_prob: float = 0.5,
        jpeg_quality_range: Tuple[int, int] = (30, 100)
    ):
        self.blur_prob = blur_prob
        self.jpeg_prob = jpeg_prob
        self.jpeg_quality_range = jpeg_quality_range
        
    def __call__(self, img):
        import torchvision.transforms.functional as TF
        from PIL import Image
        import io
        import random
        
        # Gaussian Blur
        if random.random() < self.blur_prob:
            sigma = random.uniform(0.1, 2.0)
            img = TF.gaussian_blur(img, kernel_size=5, sigma=sigma)
        
        # JPEG Compression
        if random.random() < self.jpeg_prob:
            quality = random.randint(*self.jpeg_quality_range)
            if isinstance(img, torch.Tensor):
                img = TF.to_pil_image(img)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
            
        return img


# =============================================================================
# 3. NPR 스타일 (CVPR 2024) - Neighboring Pixel Relationships
# =============================================================================
class NPRModule(nn.Module):
    """
    Neighboring Pixel Relationships 분석 모듈
    로컬 픽셀 관계의 이상 패턴 탐지
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # 픽셀 차이 계산용 필터
        self.register_buffer('h_filter', torch.tensor([
            [[[-1, 1, 0]]], [[[-1, 1, 0]]], [[[-1, 1, 0]]]
        ]).float())
        self.register_buffer('v_filter', torch.tensor([
            [[[-1], [1], [0]]], [[[-1], [1], [0]]], [[[-1], [1], [0]]]
        ]).float())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력: [B, 3, H, W] 이미지
        출력: [B, 6, H, W] NPR 특징 (수평/수직 각 채널별)
        """
        B, C, H, W = x.shape
        
        # 수평 차이
        h_diff = F.conv2d(x, self.h_filter.repeat(C, 1, 1, 1), 
                         padding=(0, 1), groups=C)
        # 수직 차이  
        v_diff = F.conv2d(x, self.v_filter.repeat(C, 1, 1, 1),
                         padding=(1, 0), groups=C)
        
        # 크기 맞추기
        h_diff = h_diff[:, :, :, :W]
        v_diff = v_diff[:, :, :H, :]
        
        return torch.cat([h_diff, v_diff], dim=1)


class NPREncoder(nn.Module):
    """NPR 특징을 인코딩하는 CNN"""
    def __init__(self, in_channels: int = 6, feature_dim: int = 512):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(512, feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.flatten(1)
        return self.fc(x)


class NPRDetector(nn.Module):
    """
    NPR 기반 탐지기 (CVPR 2024 스타일)
    픽셀 관계 분석 + CNN 인코더
    """
    def __init__(self, feature_dim: int = 512, num_classes: int = 1):
        super().__init__()
        self.npr = NPRModule()
        self.encoder = NPREncoder(in_channels=6, feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        npr_features = self.npr(x)
        encoded = self.encoder(npr_features)
        return self.classifier(encoded)
    
    def get_npr_features(self, x: torch.Tensor) -> torch.Tensor:
        """NPR 특징 맵 반환 (시각화용)"""
        return self.npr(x)


# =============================================================================
# 4. 하이브리드: CLIP + NPR 결합
# =============================================================================
class CLIPNPRHybrid(nn.Module):
    """
    CLIP 전역 특징 + NPR 로컬 특징 결합
    최고 성능을 위한 하이브리드 모델
    """
    def __init__(
        self,
        clip_model: str = "ViT-L-14",
        pretrained: str = "openai",
        npr_feature_dim: int = 256,
        fusion: str = "concat",  # 'concat', 'attention', 'gate'
        num_classes: int = 1
    ):
        super().__init__()
        self.fusion = fusion
        
        # CLIP branch
        if OPEN_CLIP_AVAILABLE:
            self.clip_encoder, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_model, pretrained=pretrained
            )
            self.clip_dim = self.clip_encoder.visual.output_dim
        else:
            raise ImportError("open_clip 패키지가 필요합니다")
            
        for param in self.clip_encoder.parameters():
            param.requires_grad = False
        
        # NPR branch
        self.npr = NPRModule()
        self.npr_encoder = NPREncoder(in_channels=6, feature_dim=npr_feature_dim)
        
        # Fusion
        if fusion == "concat":
            self.classifier = nn.Sequential(
                nn.Linear(self.clip_dim + npr_feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        elif fusion == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=256, num_heads=4, batch_first=True
            )
            self.clip_proj = nn.Linear(self.clip_dim, 256)
            self.npr_proj = nn.Linear(npr_feature_dim, 256)
            self.classifier = nn.Linear(256, num_classes)
        elif fusion == "gate":
            self.gate = nn.Sequential(
                nn.Linear(self.clip_dim + npr_feature_dim, 256),
                nn.Sigmoid()
            )
            self.clip_proj = nn.Linear(self.clip_dim, 256)
            self.npr_proj = nn.Linear(npr_feature_dim, 256)
            self.classifier = nn.Linear(256, num_classes)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CLIP features
        with torch.no_grad():
            clip_feat = self.clip_encoder.encode_image(x).float()
        
        # NPR features
        npr_map = self.npr(x)
        npr_feat = self.npr_encoder(npr_map)
        
        # Fusion
        if self.fusion == "concat":
            combined = torch.cat([clip_feat, npr_feat], dim=1)
            return self.classifier(combined)
        
        elif self.fusion == "attention":
            clip_proj = self.clip_proj(clip_feat).unsqueeze(1)
            npr_proj = self.npr_proj(npr_feat).unsqueeze(1)
            tokens = torch.cat([clip_proj, npr_proj], dim=1)
            attn_out, _ = self.attention(tokens, tokens, tokens)
            pooled = attn_out.mean(dim=1)
            return self.classifier(pooled)
        
        elif self.fusion == "gate":
            combined = torch.cat([clip_feat, npr_feat], dim=1)
            gate_weight = self.gate(combined)
            clip_proj = self.clip_proj(clip_feat)
            npr_proj = self.npr_proj(npr_feat)
            gated = gate_weight * clip_proj + (1 - gate_weight) * npr_proj
            return self.classifier(gated)


# =============================================================================
# 5. 모델 팩토리 함수
# =============================================================================
def create_detector(
    model_type: str = "univfd",
    **kwargs
) -> nn.Module:
    """
    모델 생성 팩토리 함수
    
    Args:
        model_type: 'clip_linear', 'univfd', 'npr', 'hybrid'
        **kwargs: 각 모델별 추가 인자
        
    Returns:
        nn.Module: 생성된 모델
    """
    models = {
        "clip_linear": CLIPLinearProbe,
        "univfd": UnivFD,
        "npr": NPRDetector,
        "hybrid": CLIPNPRHybrid
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type](**kwargs)


# =============================================================================
# 사용 예시
# =============================================================================
if __name__ == "__main__":
    # 테스트용 더미 입력
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print("=" * 60)
    print("모델 테스트")
    print("=" * 60)
    
    # 1. NPR Detector (CLIP 없이 테스트 가능)
    print("\n[NPR Detector]")
    npr_model = NPRDetector()
    out = npr_model(dummy_input)
    print(f"  Input: {dummy_input.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in npr_model.parameters()):,}")
    
    # 2. CLIP 기반 모델 (open_clip 설치 필요)
    if OPEN_CLIP_AVAILABLE:
        print("\n[UnivFD]")
        univfd = UnivFD(clip_model="ViT-B-32", pretrained="openai")
        out = univfd(dummy_input)
        print(f"  Input: {dummy_input.shape}")
        print(f"  Output: {out.shape}")
        
        print("\n[CLIP-NPR Hybrid]")
        hybrid = CLIPNPRHybrid(
            clip_model="ViT-B-32", 
            pretrained="openai",
            fusion="concat"
        )
        out = hybrid(dummy_input)
        print(f"  Input: {dummy_input.shape}")
        print(f"  Output: {out.shape}")
    else:
        print("\n[!] open_clip 미설치 - CLIP 기반 모델 테스트 스킵")
        print("    설치: pip install open-clip-torch")