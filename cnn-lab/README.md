# Deepfake Image Classification

PyTorch 기반 딥페이크 이미지 이진 분류 시스템입니다. 이미지를 Real(0) 또는 Fake(1)로 분류합니다.

## 환경

- **OS**: Windows 10/11
- **Python**: 3.10+
- **CUDA**: 11.8
- **GPU**: NVIDIA GPU (권장)

## 설치

### 1. 가상환경 생성 (권장)

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. PyTorch 설치 (CUDA 11.8)

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

> **Note**: 다른 CUDA 버전 사용 시 [PyTorch 공식 사이트](https://pytorch.org/get-started/locally/)에서 적절한 명령어를 확인하세요.

### 3. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 설치 확인

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 프로젝트 구조

```
_test_classification/
├── config/
│   ├── architecture.yaml    # 모델 및 학습 설정
│   └── data_config.yaml     # 데이터 경로 설정
├── src/
│   ├── train.py            # 학습 스크립트
│   ├── evaluation.py       # 평가 스크립트
│   ├── grad_cam.py         # Grad-CAM 시각화
│   ├── model.py            # 모델 정의
│   ├── dataset.py          # 데이터셋 클래스
│   └── utils.py            # 유틸리티 함수
├── results/                 # 학습/평가 결과 저장
├── requirements.txt
└── README.md
```

## 사용법

### 1. 데이터 설정

`config/data_config.yaml` 파일을 수정하여 데이터 경로를 설정합니다:

```yaml
train:
  fake:
    - D:/datasets/deepfake/train/fake
  real:
    - D:/datasets/deepfake/train/real

validation:
  fake:
    - D:/datasets/deepfake/validation/fake
  real:
    - D:/datasets/deepfake/validation/real

test:
  fake:
    - D:/datasets/deepfake/test/fake
  real:
    - D:/datasets/deepfake/test/real
```

### 2. 학습 (Training)

```bash
cd src
python train.py ^
    --data-config ../config/data_config.yaml ^
    --model-config ../config/architecture.yaml ^
    --save-dir ../results ^
    --epochs 200 ^
    --best-condition loss
```

**주요 옵션:**
- `--epochs`: 학습 에폭 수 (기본값: 200)
- `--best-condition`: 최적 모델 선택 기준 (`loss` 또는 `acc`)
- `--weight-path`: 사전 학습된 가중치 경로 (선택)

### 3. 평가 (Evaluation)

```bash
cd src
python evaluation.py ^
    --data-config ../config/data_config.yaml ^
    --model-config ../config/architecture.yaml ^
    --weight-path ../results/train_XXXXXX/Epoch_BEST/weight/Epoch_BEST.pth ^
    --save-dir ../results/evaluation ^
    --folder-name test_results
```

### 4. Grad-CAM 시각화

```bash
cd src
python grad_cam.py ^
    --csv-path ../results/evaluation/test_results/pred_results_info.csv ^
    --model-config ../config/architecture.yaml ^
    --weight-path ../results/train_XXXXXX/Epoch_BEST/weight/Epoch_BEST.pth ^
    --save-dir ../results/gradcam ^
    --actual-label 1 ^
    --predict-label 1
```

**Grad-CAM 옵션:**
- `--actual-label`: 실제 라벨 필터 (0=Real, 1=Fake)
- `--predict-label`: 예측 라벨 필터 (0=Real, 1=Fake)
- `--layer-name`: 특정 레이어 지정 (선택)

## 모델 설정

`config/architecture.yaml`:

```yaml
training:
  lr: 0.001           # 학습률
  weight-decay: 0.001 # 가중치 감쇠
  bs: 8               # 배치 사이즈
  early-stop: 20      # 조기 종료 patience

model:
  name: xception      # 모델 이름 (timm 지원 모델)
  image-size: 299     # 입력 이미지 크기
  num-classes: 1      # 출력 클래스 수 (이진 분류)
```

**지원 모델:**
- `xception` (기본값)
- `xception_fft` (FFT 주파수 특성 결합)
- timm 라이브러리 지원 모델 (`resnet50`, `efficientnet_b0` 등)

## 출력 결과

학습 완료 후 `results/train_XXXXXX/` 디렉토리에 저장:

```
train_XXXXXX/
├── architecture.yaml          # 사용된 모델 설정
├── data_config.yaml          # 사용된 데이터 설정
├── acc_per_epoch_line.png    # 에폭별 정확도 그래프
├── loss_per_epoch_line.png   # 에폭별 손실 그래프
├── acc_loss_info.csv         # 학습 기록
├── Epoch_BEST/
│   ├── weight/
│   │   └── Epoch_BEST.pth    # 최적 모델 가중치
│   ├── confusion_matrix_heatmap.png
│   ├── roc_curve.png
│   └── performance.csv       # 성능 지표
└── Epoch_LAST/
    └── ...
```

## 성능 지표

- **Accuracy**: 전체 정확도
- **Balanced Accuracy**: 클래스 균형 정확도
- **Precision**: 정밀도
- **Recall**: 재현율
- **F1-score**: F1 점수
- **AUC**: ROC 곡선 아래 면적

## 라이선스

MIT License
