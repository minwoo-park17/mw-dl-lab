# ConvLSTM 기반 생성형 AI 탐지 프로젝트

## 프로젝트 개요
연속된 이미지 프레임의 시간적 일관성(temporal consistency)을 분석하여 생성형 AI로 만들어진 영상/이미지를 탐지하는 시스템

### 핵심 아이디어
- 생성형 AI(딥페이크 등)가 만든 영상은 프레임 간 미세한 불일치 존재
- ConvLSTM으로 연속 프레임의 시공간적(spatiotemporal) 패턴 학습
- N개의 연속 프레임을 입력으로 받아 Real/Fake 이진 분류

---

## 디렉토리 구조

```
convlstm-lab/
├── config/
│   ├── data_config.yaml       # 데이터 경로, 전처리 설정
│   └── model_config.yaml      # 모델 하이퍼파라미터
├── src/
│   ├── __init__.py
│   ├── dataset.py             # 데이터셋 및 데이터로더
│   ├── model.py               # ConvLSTM 모델 정의
│   ├── transforms.py          # 데이터 증강 및 전처리
│   └── utils.py               # 유틸리티 함수
├── data/
│   ├── train/
│   │   ├── real/              # 실제 영상 프레임 시퀀스
│   │   └── fake/              # 생성형 AI 영상 프레임 시퀀스
│   └── val/
│       ├── real/
│       └── fake/
├── checkpoints/               # 학습된 모델 저장
├── logs/                      # 학습 로그 (TensorBoard)
├── train.py                   # 학습 스크립트
├── test.py                    # 추론/평가 스크립트
├── requirements.txt           # 의존성 패키지
└── TODO.md
```

---

## 구현 계획

### Phase 1: 기본 구조 설정

- [ ] **1.1 프로젝트 디렉토리 생성**
  - config/, src/, data/, checkpoints/, logs/ 폴더 생성

- [ ] **1.2 requirements.txt 작성**
  ```
  torch>=2.0.0
  torchvision>=0.15.0
  opencv-python>=4.8.0
  numpy>=1.24.0
  pyyaml>=6.0
  tensorboard>=2.14.0
  tqdm>=4.65.0
  scikit-learn>=1.3.0
  albumentations>=1.3.0
  ```

- [ ] **1.3 config 파일 작성**
  - data_config.yaml: 데이터 경로, 시퀀스 길이, 이미지 크기
  - model_config.yaml: 모델 아키텍처, 학습률, 배치 크기

---

### Phase 2: 데이터 파이프라인 (dataset.py)

- [ ] **2.1 FrameSequenceDataset 클래스**
  ```python
  class FrameSequenceDataset(Dataset):
      """
      연속 프레임 시퀀스를 로드하는 데이터셋

      입력: 폴더 내 정렬된 프레임 이미지들
      출력: (sequence, label)
        - sequence: (T, C, H, W) 텐서 (T=시퀀스 길이)
        - label: 0(real) or 1(fake)
      """
  ```

- [ ] **2.2 데이터 로딩 전략**
  - 각 영상 폴더에서 연속 N프레임 샘플링
  - 슬라이딩 윈도우 방식으로 다중 샘플 생성
  - 클래스 불균형 처리 (WeightedRandomSampler)

- [ ] **2.3 transforms.py 구현**
  - 프레임 리사이즈, 정규화
  - 시퀀스 단위 증강 (동일 변환을 모든 프레임에 적용)
  - 랜덤 크롭, 플립, 컬러 지터링

---

### Phase 3: 모델 구현 (model.py)

- [ ] **3.1 ConvLSTMCell 구현**
  ```python
  class ConvLSTMCell(nn.Module):
      """
      단일 ConvLSTM 셀
      - 입력: (batch, channels, height, width)
      - hidden/cell state 유지
      """
  ```

- [ ] **3.2 ConvLSTM 레이어 구현**
  ```python
  class ConvLSTM(nn.Module):
      """
      다층 ConvLSTM
      - 양방향(bidirectional) 옵션
      - 여러 레이어 스택 가능
      """
  ```

- [ ] **3.3 전체 분류 모델 구현**
  ```python
  class ConvLSTMClassifier(nn.Module):
      """
      전체 파이프라인:
      1. CNN Encoder (특징 추출) - ResNet/EfficientNet backbone
      2. ConvLSTM (시간적 패턴 학습)
      3. Classifier Head (이진 분류)

      입력: (B, T, C, H, W)
      출력: (B, 2) logits
      """
  ```

- [ ] **3.4 모델 변형 옵션**
  - Backbone 선택: ResNet18/34, EfficientNet-B0/B2
  - Attention 메커니즘 추가 (선택적)
  - 마지막 hidden state vs 전체 시퀀스 pooling

---

### Phase 4: 학습 스크립트 (train.py)

- [ ] **4.1 학습 루프 구현**
  ```python
  def train():
      # 1. Config 로드
      # 2. 데이터셋/데이터로더 초기화
      # 3. 모델, 옵티마이저, 스케줄러 초기화
      # 4. 학습 루프
      #    - Forward pass
      #    - Loss 계산 (CrossEntropy + 선택적 정규화)
      #    - Backward pass
      #    - 검증 및 체크포인트 저장
  ```

- [ ] **4.2 학습 설정**
  - Optimizer: AdamW
  - Scheduler: CosineAnnealingLR 또는 OneCycleLR
  - Mixed Precision Training (AMP)
  - Gradient Clipping

- [ ] **4.3 로깅 및 모니터링**
  - TensorBoard 로깅
  - Train/Val Loss, Accuracy, AUC 추적
  - Best model 자동 저장

- [ ] **4.4 CLI 인터페이스**
  ```bash
  python train.py --config config/model_config.yaml --epochs 50
  ```

---

### Phase 5: 추론/평가 스크립트 (test.py)

- [ ] **5.1 단일 영상 추론**
  ```python
  def inference_video(video_path, model, config):
      """
      영상 파일에서 프레임 추출 후 예측
      - 슬라이딩 윈도우로 여러 시퀀스 평가
      - 앙상블 (평균/투표)로 최종 판정
      """
  ```

- [ ] **5.2 프레임 폴더 추론**
  ```python
  def inference_frames(frame_dir, model, config):
      """
      이미 추출된 프레임 폴더에서 예측
      """
  ```

- [ ] **5.3 평가 메트릭**
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC, PR-AUC
  - Confusion Matrix 시각화

- [ ] **5.4 CLI 인터페이스**
  ```bash
  # 단일 영상
  python test.py --video path/to/video.mp4 --checkpoint checkpoints/best.pth

  # 프레임 폴더
  python test.py --frames path/to/frames/ --checkpoint checkpoints/best.pth

  # 테스트셋 전체 평가
  python test.py --eval --data_dir data/test/ --checkpoint checkpoints/best.pth
  ```

---

### Phase 6: 유틸리티 (utils.py)

- [ ] **6.1 영상→프레임 추출**
  ```python
  def extract_frames(video_path, output_dir, fps=None):
      """OpenCV로 영상에서 프레임 추출"""
  ```

- [ ] **6.2 체크포인트 관리**
  ```python
  def save_checkpoint(model, optimizer, epoch, path)
  def load_checkpoint(path, model, optimizer=None)
  ```

- [ ] **6.3 시각화**
  ```python
  def visualize_prediction(frames, prediction, save_path)
  def plot_training_curves(log_path)
  ```

---

## Config 파일 상세

### data_config.yaml
```yaml
data:
  train_dir: "data/train"
  val_dir: "data/val"
  test_dir: "data/test"

sequence:
  length: 16              # 연속 프레임 수
  stride: 8               # 샘플링 간격

image:
  height: 224
  width: 224
  channels: 3

augmentation:
  horizontal_flip: true
  random_crop: true
  color_jitter: true
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

### model_config.yaml
```yaml
model:
  backbone: "resnet18"      # resnet18, resnet34, efficientnet_b0
  pretrained: true
  freeze_backbone: false

  convlstm:
    hidden_channels: [64, 128]
    kernel_size: 3
    num_layers: 2
    bidirectional: false

  classifier:
    dropout: 0.5
    num_classes: 2

training:
  batch_size: 8
  epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.01

  scheduler:
    type: "cosine"
    warmup_epochs: 5

  early_stopping:
    patience: 10
    min_delta: 0.001

device: "cuda"
seed: 42
```

---

## 데이터 준비 가이드

### 폴더 구조 예시
```
data/train/real/video001/
├── frame_0001.jpg
├── frame_0002.jpg
├── ...
└── frame_0100.jpg

data/train/fake/video001/
├── frame_0001.jpg
├── ...
```

### 권장 데이터셋
- FaceForensics++ (딥페이크)
- Celeb-DF
- DFDC (Deepfake Detection Challenge)
- 자체 수집 데이터

---

## 실행 순서

1. 환경 설정
   ```bash
   pip install -r requirements.txt
   ```

2. 데이터 준비
   - 영상에서 프레임 추출
   - train/val/test 분할
   - real/fake 폴더 분류

3. 학습
   ```bash
   python train.py --config config/model_config.yaml
   ```

4. 평가
   ```bash
   python test.py --eval --checkpoint checkpoints/best.pth
   ```

5. 추론
   ```bash
   python test.py --video suspicious_video.mp4 --checkpoint checkpoints/best.pth
   ```

---

## 성능 최적화 고려사항

- [ ] 프레임 캐싱 (LMDB, HDF5)
- [ ] 멀티 GPU 학습 (DistributedDataParallel)
- [ ] 모델 경량화 (Quantization, Pruning)
- [ ] TensorRT 변환 (추론 속도 향상)

---

## 참고 자료

- [ConvLSTM 논문](https://arxiv.org/abs/1506.04214)
- [FaceForensics++ 논문](https://arxiv.org/abs/1901.08971)
- PyTorch Video Classification Examples
