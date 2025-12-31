# Mamba 기반 Forgery Detection 프로젝트 TODO

## 프로젝트 개요

두 가지 Mamba 기반 위조 탐지 기법을 통합 구현:

| 모델 | 태스크 | 특징 |
|------|--------|------|
| **WMamba** | Face Forgery Detection | Wavelet 기반, 얼굴 윤곽선 경계 불일치 탐지 |
| **ForMa** | Image Tampering Localization | 조작된 영역 위치화 (Segmentation Mask 출력) |

---

## 📁 프로젝트 구조

```
vmamba-lab/
├── config/
│   ├── default.yaml              # 공통 기본 설정
│   ├── wmamba_config.yaml        # WMamba 전용 설정
│   └── forma_config.yaml         # ForMa 전용 설정
│
├── data/
│   ├── raw/                      # 원본 데이터 (다운로드 후 저장)
│   ├── processed/                # 전처리된 데이터
│   └── data_path.yaml            # 데이터 경로 설정 파일
│
├── dataset/
│   ├── __init__.py
│   ├── base_dataset.py           # 공통 Dataset 베이스 클래스
│   ├── face_forgery_dataset.py   # WMamba용 (FF++, CDF, DFDC 등)
│   ├── tampering_dataset.py      # ForMa용 (CASIA, Columbia 등)
│   ├── transforms.py             # 데이터 augmentation
│   └── sampler.py                # 클래스 불균형 처리 샘플러
│
├── model/
│   ├── __init__.py
│   ├── backbone/
│   │   ├── __init__.py
│   │   ├── vmamba.py             # VMamba 백본 (공통)
│   │   └── vss_block.py          # Visual State Space Block
│   │
│   ├── wmamba/
│   │   ├── __init__.py
│   │   ├── wmamba.py             # WMamba 메인 모델
│   │   ├── hwfeb.py              # Hierarchical Wavelet Feature Extraction Branch
│   │   ├── dcconv.py             # Dynamic Contour Convolution
│   │   └── wavelet_utils.py      # DWT/IDWT 유틸리티
│   │
│   └── forma/
│       ├── __init__.py
│       ├── forma.py              # ForMa 메인 모델
│       ├── encoder.py            # VSS Encoder
│       ├── decoder.py            # Lightweight Decoder (Pixel Shuffle)
│       └── noise_module.py       # Noise-assisted Decoding
│
├── train/
│   ├── __init__.py
│   ├── trainer.py                # 공통 Trainer 클래스
│   ├── train_wmamba.py           # WMamba 학습 스크립트
│   ├── train_forma.py            # ForMa 학습 스크립트
│   └── losses/
│       ├── __init__.py
│       ├── classification_loss.py  # BCE, Focal Loss 등
│       └── segmentation_loss.py    # Dice, IoU Loss 등
│
├── test/
│   ├── __init__.py
│   ├── evaluator.py              # 공통 평가 클래스
│   ├── test_wmamba.py            # WMamba 테스트 스크립트
│   ├── test_forma.py             # ForMa 테스트 스크립트
│   └── metrics/
│       ├── __init__.py
│       ├── classification_metrics.py  # AUC, ACC, EER
│       └── segmentation_metrics.py    # F1, IoU, Pixel-ACC
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                 # 로깅 유틸리티
│   ├── checkpoint.py             # 모델 저장/로드
│   ├── visualize.py              # 결과 시각화
│   ├── face_utils.py             # 얼굴 검출/크롭 유틸리티
│   └── device.py                 # GPU/CPU 설정
│
├── scripts/
│   ├── download_datasets.sh      # 데이터셋 다운로드 스크립트
│   ├── preprocess_ff++.py        # FF++ 전처리
│   └── extract_faces.py          # 얼굴 추출 스크립트
│
├── requirements.txt
├── README.md
└── TODO.md
```

---

## ✅ 구현 체크리스트

### Phase 1: 환경 설정 및 기반 구축

- [ ] **환경 설정**
  - [ ] `requirements.txt` 작성
    - torch, torchvision
    - mamba-ssm (또는 causal-conv1d)
    - pywavelets (Wavelet 변환)
    - opencv-python, pillow
    - albumentations (augmentation)
    - timm (pretrained backbones)
    - wandb/tensorboard (로깅)
    - facenet-pytorch 또는 insightface (얼굴 검출)
  - [ ] CUDA/cuDNN 버전 호환성 확인
  - [ ] Mamba 설치 (Linux 권장, Windows는 WSL 필요할 수 있음)

- [ ] **Config 시스템 구축**
  - [ ] `config/default.yaml` - 공통 설정 (seed, device, logging)
  - [ ] `config/wmamba_config.yaml`
    ```yaml
    model:
      name: wmamba
      wavelet: db1  # Daubechies wavelet
      wavelet_levels: 3
      backbone: vmamba_tiny
      num_classes: 2

    train:
      batch_size: 32
      epochs: 50
      lr: 1e-4
      optimizer: adamw
      scheduler: cosine

    data:
      input_size: 256
      train_dataset: ff++
      compression: c23
    ```
  - [ ] `config/forma_config.yaml`
    ```yaml
    model:
      name: forma
      backbone: vmamba_small
      decoder_channels: [256, 128, 64, 32]
      noise_assisted: true

    train:
      batch_size: 16
      epochs: 100
      lr: 5e-5

    data:
      input_size: 512
      train_datasets: [casia, coverage, columbia]
    ```
  - [ ] `data/data_path.yaml` - 데이터 경로 설정

---

### Phase 2: 데이터셋 준비

- [ ] **WMamba용 데이터셋**
  - [ ] FaceForensics++ (FF++) 다운로드 및 전처리
    - [ ] 얼굴 검출 및 크롭 (RetinaFace/MTCNN)
    - [ ] c23/c40 압축 버전 준비
    - [ ] Deepfakes, Face2Face, FaceSwap, NeuralTextures 분류
  - [ ] Celeb-DF-v2 다운로드
  - [ ] DFDC (선택적 - 용량 큼)
  - [ ] `dataset/face_forgery_dataset.py` 구현
    ```python
    class FaceForgeryDataset(Dataset):
        def __init__(self, data_root, split, transform, compression='c23'):
            # Real/Fake 이진 분류
            pass

        def __getitem__(self, idx):
            # return image, label (0: real, 1: fake)
            pass
    ```

- [ ] **ForMa용 데이터셋**
  - [ ] CASIA v1/v2 다운로드
  - [ ] Columbia 다운로드
  - [ ] Coverage 다운로드
  - [ ] NIST16 다운로드
  - [ ] IMD2020 (선택적)
  - [ ] `dataset/tampering_dataset.py` 구현
    ```python
    class TamperingDataset(Dataset):
        def __init__(self, data_root, split, transform):
            # 이미지 + Mask 쌍
            pass

        def __getitem__(self, idx):
            # return image, mask (binary segmentation mask)
            pass
    ```

- [ ] **공통 데이터 유틸리티**
  - [ ] `dataset/transforms.py` - Augmentation 파이프라인
    ```python
    def get_train_transforms(input_size):
        # RandomHorizontalFlip, RandomRotation, ColorJitter 등
        pass

    def get_test_transforms(input_size):
        # Resize, Normalize만
        pass
    ```
  - [ ] `dataset/sampler.py` - 클래스 불균형 처리

---

### Phase 3: 모델 구현

- [ ] **공통 백본 (VMamba)**
  - [ ] `model/backbone/vss_block.py` - SS2D 블록 구현
    ```python
    class SS2D(nn.Module):
        """2D Selective Scan"""
        def __init__(self, d_model, d_state, d_conv, expand):
            pass

    class VSSBlock(nn.Module):
        """Visual State Space Block"""
        def __init__(self, hidden_dim, drop_path):
            pass
    ```
  - [ ] `model/backbone/vmamba.py` - VMamba 백본
    ```python
    class VMamba(nn.Module):
        def __init__(self, depths, dims, drop_path_rate):
            pass

        def forward_features(self, x):
            # Multi-scale features 반환
            pass
    ```

- [ ] **WMamba 모델**
  - [ ] `model/wmamba/wavelet_utils.py` - DWT 구현
    ```python
    class DWT2D(nn.Module):
        """2D Discrete Wavelet Transform"""
        def __init__(self, wavelet='db1'):
            pass

        def forward(self, x):
            # return LL, LH, HL, HH
            pass
    ```
  - [ ] `model/wmamba/hwfeb.py` - Hierarchical Wavelet Feature Extraction
    ```python
    class HWFEB(nn.Module):
        """Hierarchical Wavelet Feature Extraction Branch"""
        def __init__(self, wavelet, levels):
            # Multi-level DWT + Feature extraction
            pass
    ```
  - [ ] `model/wmamba/dcconv.py` - Dynamic Contour Convolution
    ```python
    class DCConv(nn.Module):
        """Dynamic Contour Convolution for slender facial contours"""
        def __init__(self, in_channels, out_channels):
            # Deformable convolution variant
            pass
    ```
  - [ ] `model/wmamba/wmamba.py` - 메인 모델
    ```python
    class WMamba(nn.Module):
        def __init__(self, config):
            self.hwfeb = HWFEB(...)
            self.vmamba = VMamba(...)
            self.classifier = nn.Linear(...)

        def forward(self, x):
            # Wavelet features + VMamba features 결합
            # return logits
            pass
    ```

- [ ] **ForMa 모델**
  - [ ] `model/forma/encoder.py` - VSS Encoder
    ```python
    class VSSEncoder(nn.Module):
        """VMamba-based encoder for multi-scale features"""
        def __init__(self, config):
            pass

        def forward(self, x):
            # return multi-scale features [f1, f2, f3, f4]
            pass
    ```
  - [ ] `model/forma/decoder.py` - Lightweight Decoder
    ```python
    class LightweightDecoder(nn.Module):
        """Pixel Shuffle based decoder"""
        def __init__(self, in_channels_list, out_channels):
            # PixelShuffle upsampling
            pass

        def forward(self, features):
            # return segmentation mask
            pass
    ```
  - [ ] `model/forma/noise_module.py` - Noise-assisted Decoding
    ```python
    class NoiseAssistedModule(nn.Module):
        """Extract noise features for manipulation detection"""
        def __init__(self):
            # SRM filters or learnable noise extractor
            pass
    ```
  - [ ] `model/forma/forma.py` - 메인 모델
    ```python
    class ForMa(nn.Module):
        def __init__(self, config):
            self.encoder = VSSEncoder(...)
            self.noise_module = NoiseAssistedModule(...)
            self.decoder = LightweightDecoder(...)

        def forward(self, x):
            # return segmentation mask (H x W)
            pass
    ```

---

### Phase 4: 학습 파이프라인

- [ ] **Loss 함수**
  - [ ] `train/losses/classification_loss.py`
    ```python
    class FocalLoss(nn.Module):
        """클래스 불균형 대응"""
        pass

    class LabelSmoothingLoss(nn.Module):
        pass
    ```
  - [ ] `train/losses/segmentation_loss.py`
    ```python
    class DiceLoss(nn.Module):
        pass

    class BCEDiceLoss(nn.Module):
        """BCE + Dice 결합"""
        pass

    class IoULoss(nn.Module):
        pass
    ```

- [ ] **Trainer 구현**
  - [ ] `train/trainer.py` - 공통 Trainer
    ```python
    class BaseTrainer:
        def __init__(self, model, train_loader, val_loader, config):
            pass

        def train_epoch(self):
            pass

        def validate(self):
            pass

        def save_checkpoint(self):
            pass
    ```
  - [ ] `train/train_wmamba.py`
    ```python
    class WMambaTrainer(BaseTrainer):
        # Classification 특화
        pass

    if __name__ == "__main__":
        # argparse로 config 경로 받기
        # python train/train_wmamba.py --config config/wmamba_config.yaml
        pass
    ```
  - [ ] `train/train_forma.py`
    ```python
    class ForMaTrainer(BaseTrainer):
        # Segmentation 특화
        pass
    ```

- [ ] **유틸리티**
  - [ ] `utils/logger.py` - WandB/TensorBoard 로깅
  - [ ] `utils/checkpoint.py` - 모델 저장/로드
  - [ ] `utils/device.py` - Multi-GPU 지원 (DDP)

---

### Phase 5: 테스트 및 평가

- [ ] **평가 메트릭**
  - [ ] `test/metrics/classification_metrics.py`
    ```python
    def compute_auc(y_true, y_pred):
        pass

    def compute_eer(y_true, y_pred):
        """Equal Error Rate"""
        pass

    def compute_accuracy(y_true, y_pred, threshold=0.5):
        pass
    ```
  - [ ] `test/metrics/segmentation_metrics.py`
    ```python
    def compute_f1(pred_mask, gt_mask, threshold=0.5):
        pass

    def compute_iou(pred_mask, gt_mask):
        pass

    def compute_pixel_auc(pred_mask, gt_mask):
        pass
    ```

- [ ] **테스트 스크립트**
  - [ ] `test/test_wmamba.py`
    ```python
    # Cross-dataset evaluation
    # Train: FF++ -> Test: CDF, DFDC, DFDCP
    def evaluate_cross_dataset(model, test_datasets):
        pass
    ```
  - [ ] `test/test_forma.py`
    ```python
    # Multi-dataset evaluation
    # Test on: CASIA, Columbia, Coverage, NIST16
    def evaluate_tampering_localization(model, test_datasets):
        pass
    ```

- [ ] **시각화**
  - [ ] `utils/visualize.py`
    ```python
    def visualize_wmamba_attention(model, image):
        """Attention/Grad-CAM 시각화"""
        pass

    def visualize_forma_prediction(image, pred_mask, gt_mask):
        """Segmentation 결과 오버레이"""
        pass

    def plot_roc_curve(results):
        pass
    ```

---

### Phase 6: 추가 기능 (선택적)

- [ ] **성능 최적화**
  - [ ] Mixed Precision Training (AMP)
  - [ ] Gradient Checkpointing (메모리 절약)
  - [ ] Model EMA (Exponential Moving Average)

- [ ] **실험 관리**
  - [ ] WandB sweep 설정 (하이퍼파라미터 탐색)
  - [ ] 실험 결과 자동 기록

- [ ] **추론 최적화**
  - [ ] ONNX 변환
  - [ ] TensorRT 최적화
  - [ ] 배치 추론 지원

- [ ] **추가 기능**
  - [ ] 비디오 입력 지원 (연속 프레임 분석)
  - [ ] Ensemble 모델 (WMamba + ForMa 결합)
  - [ ] Gradio/Streamlit 데모 UI

---

## 📊 예상 데이터셋 용량

| 데이터셋 | 용도 | 대략적 용량 |
|----------|------|-------------|
| FaceForensics++ (c23) | WMamba Train | ~50GB |
| Celeb-DF-v2 | WMamba Test | ~5GB |
| DFDC | WMamba Test | ~470GB (선택적) |
| CASIA v2 | ForMa Train/Test | ~500MB |
| Columbia | ForMa Test | ~200MB |
| Coverage | ForMa Test | ~100MB |
| NIST16 | ForMa Test | ~1GB |

---

## 🔗 참고 자료

### 논문
- [WMamba: Wavelet-based Mamba for Face Forgery Detection](https://arxiv.org/abs/2501.09617)
- [ForMa: A Lightweight and Effective Image Tampering Localization Network with Vision Mamba](https://arxiv.org/abs/2502.09941)
- [VMamba: Visual State Space Model](https://arxiv.org/abs/2401.10166)

### 코드 참고
- [VMamba Official](https://github.com/MzeroMiko/VMamba)
- [Mamba Official](https://github.com/state-spaces/mamba)
- [Awesome-Comprehensive-Deepfake-Detection](https://github.com/qiqitao77/Awesome-Comprehensive-Deepfake-Detection)

---

## ⚠️ 주의사항

1. **Mamba 설치**: Windows에서 직접 설치가 어려울 수 있음. WSL2 또는 Linux 환경 권장
2. **GPU 메모리**: WMamba ~8GB, ForMa ~12GB 이상 권장
3. **데이터셋 라이센스**: 연구 목적으로만 사용, 상업적 사용 제한 확인 필요
4. **FF++ 다운로드**: 별도 신청 필요 (https://github.com/ondyari/FaceForensics)

---

## 🚀 Quick Start (목표)

```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 데이터 경로 설정
vim data/data_path.yaml

# 3. WMamba 학습
python train/train_wmamba.py --config config/wmamba_config.yaml

# 4. ForMa 학습
python train/train_forma.py --config config/forma_config.yaml

# 5. 테스트
python test/test_wmamba.py --checkpoint outputs/wmamba_best.pth
python test/test_forma.py --checkpoint outputs/forma_best.pth
```
