"""
ConvLSTM 모델 추론/평가 스크립트

사용법:
    # 단일 영상 추론
    python test.py --video path/to/video.mp4 --checkpoint checkpoints/best.pth

    # 프레임 폴더 추론
    python test.py --frames path/to/frames/ --checkpoint checkpoints/best.pth

    # 테스트셋 전체 평가
    python test.py --eval --data_dir data/test/ --checkpoint checkpoints/best.pth
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from src.dataset import FrameSequenceDataset, VideoFrameDataset, get_dataloader
from src.model import build_model
from src.transforms import get_val_transforms
from src.utils import (
    load_config,
    merge_configs,
    load_checkpoint,
    get_device,
    extract_frames,
    visualize_prediction
)


def parse_args():
    parser = argparse.ArgumentParser(description='ConvLSTM Inference/Evaluation')

    # 입력 소스 (상호 배타적)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='추론할 비디오 파일 경로')
    input_group.add_argument('--frames', type=str, help='추론할 프레임 폴더 경로')
    input_group.add_argument('--eval', action='store_true', help='테스트셋 평가 모드')

    # 필수 인자
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='모델 체크포인트 경로')

    # 설정
    parser.add_argument('--data_config', type=str, default='config/data_config.yaml',
                        help='데이터 config 파일 경로')
    parser.add_argument('--model_config', type=str, default='config/model_config.yaml',
                        help='모델 config 파일 경로')
    parser.add_argument('--data_dir', type=str, default='data/test',
                        help='평가 모드시 테스트 데이터 디렉토리')

    # 추론 옵션
    parser.add_argument('--stride', type=int, default=4,
                        help='슬라이딩 윈도우 간격')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='분류 임계값')
    parser.add_argument('--visualize', action='store_true',
                        help='예측 결과 시각화 저장')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='출력 저장 디렉토리')

    return parser.parse_args()


@torch.no_grad()
def inference_frames(
    frame_dir: str,
    model: nn.Module,
    config: Dict,
    device: torch.device,
    stride: int = 4,
    threshold: float = 0.5
) -> Dict:
    """
    프레임 폴더에서 추론

    Args:
        frame_dir: 프레임 폴더 경로
        model: 학습된 모델
        config: 설정
        device: 디바이스
        stride: 슬라이딩 윈도우 간격
        threshold: 분류 임계값

    Returns:
        예측 결과 딕셔너리
    """
    model.eval()

    seq_length = config.get('sequence', {}).get('length', 16)
    transform = get_val_transforms(config)

    # 데이터셋 생성
    dataset = VideoFrameDataset(
        frame_dir=frame_dir,
        sequence_length=seq_length,
        stride=stride,
        transform=transform
    )

    if len(dataset) == 0:
        return {
            'prediction': -1,
            'confidence': 0.0,
            'message': 'Not enough frames for inference'
        }

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_probs = []

    for sequences, _ in dataloader:
        sequences = sequences.to(device)
        outputs = model(sequences)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs[:, 1].cpu().numpy())

    # 앙상블: 평균 확률
    avg_prob = np.mean(all_probs)

    # 최종 예측
    prediction = 1 if avg_prob >= threshold else 0
    label = 'FAKE' if prediction == 1 else 'REAL'

    return {
        'prediction': prediction,
        'label': label,
        'confidence': float(avg_prob if prediction == 1 else 1 - avg_prob),
        'fake_probability': float(avg_prob),
        'num_sequences': len(all_probs),
        'sequence_probs': [float(p) for p in all_probs]
    }


@torch.no_grad()
def inference_video(
    video_path: str,
    model: nn.Module,
    config: Dict,
    device: torch.device,
    stride: int = 4,
    threshold: float = 0.5,
    fps: Optional[float] = None
) -> Dict:
    """
    비디오 파일에서 추론

    Args:
        video_path: 비디오 파일 경로
        model: 학습된 모델
        config: 설정
        device: 디바이스
        stride: 슬라이딩 윈도우 간격
        threshold: 분류 임계값
        fps: 프레임 추출 FPS

    Returns:
        예측 결과 딕셔너리
    """
    # 임시 디렉토리에 프레임 추출
    with tempfile.TemporaryDirectory() as temp_dir:
        img_config = config.get('image', {})
        resize = (img_config.get('width', 224), img_config.get('height', 224))

        print(f"Extracting frames from video...")
        num_frames = extract_frames(
            video_path,
            temp_dir,
            fps=fps,
            resize=resize
        )

        if num_frames == 0:
            return {
                'prediction': -1,
                'confidence': 0.0,
                'message': 'Failed to extract frames from video'
            }

        # 프레임 폴더로 추론
        result = inference_frames(
            temp_dir, model, config, device, stride, threshold
        )
        result['video_path'] = video_path
        result['total_frames'] = num_frames

        return result


@torch.no_grad()
def evaluate_dataset(
    data_dir: str,
    model: nn.Module,
    config: Dict,
    device: torch.device
) -> Dict:
    """
    테스트 데이터셋 전체 평가

    Args:
        data_dir: 테스트 데이터 디렉토리
        model: 학습된 모델
        config: 설정
        device: 디바이스

    Returns:
        평가 결과 딕셔너리
    """
    model.eval()

    # 데이터로더 생성
    dataloader = get_dataloader(
        data_dir, config, is_train=False, use_weighted_sampler=False
    )

    all_preds = []
    all_labels = []
    all_probs = []

    print(f"\nEvaluating on {len(dataloader.dataset)} samples...")

    pbar = tqdm(dataloader, desc='Evaluating')

    for sequences, labels in pbar:
        sequences = sequences.to(device)

        outputs = model(sequences)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    # 메트릭 계산
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=['Real', 'Fake'],
        digits=4
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'total_samples': len(all_labels),
        'predictions': {
            'labels': all_labels,
            'preds': all_preds,
            'probs': all_probs
        }
    }


def print_results(result: Dict, mode: str):
    """결과 출력"""
    print("\n" + "=" * 50)

    if mode == 'eval':
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Total Samples: {result['total_samples']}")
        print(f"\nMetrics:")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1 Score:  {result['f1']:.4f}")
        print(f"  AUC:       {result['auc']:.4f}")
        print(f"\nConfusion Matrix:")
        cm = result['confusion_matrix']
        print(f"              Pred Real  Pred Fake")
        print(f"  True Real   {cm[0][0]:8d}  {cm[0][1]:8d}")
        print(f"  True Fake   {cm[1][0]:8d}  {cm[1][1]:8d}")
        print(f"\nClassification Report:")
        print(result['classification_report'])

    else:
        print("INFERENCE RESULT")
        print("=" * 50)

        if result.get('prediction', -1) == -1:
            print(f"Error: {result.get('message', 'Unknown error')}")
            return

        label = result['label']
        conf = result['confidence']
        fake_prob = result['fake_probability']

        color = '\033[91m' if label == 'FAKE' else '\033[92m'
        reset = '\033[0m'

        print(f"Prediction: {color}{label}{reset}")
        print(f"Confidence: {conf:.2%}")
        print(f"Fake Probability: {fake_prob:.4f}")
        print(f"Sequences Analyzed: {result['num_sequences']}")

        if 'video_path' in result:
            print(f"Video: {result['video_path']}")
            print(f"Total Frames: {result['total_frames']}")

    print("=" * 50)


def main():
    args = parse_args()

    # Config 로드
    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    config = merge_configs(data_config, model_config)

    # 디바이스 설정
    device = get_device(config.get('device', 'cuda'))

    # 모델 로드
    print("\n=== Loading Model ===")
    model = build_model(config)
    load_checkpoint(args.checkpoint, model, device=str(device))
    model = model.to(device)
    model.eval()

    # 출력 디렉토리 생성
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)

    # 모드별 실행
    if args.eval:
        # 테스트셋 평가
        result = evaluate_dataset(args.data_dir, model, config, device)
        print_results(result, 'eval')

        # 결과 저장
        import json
        result_path = os.path.join(args.output_dir, 'evaluation_results.json')
        save_result = {k: v for k, v in result.items() if k != 'predictions'}
        with open(result_path, 'w') as f:
            json.dump(save_result, f, indent=2)
        print(f"\nResults saved to: {result_path}")

    elif args.video:
        # 비디오 추론
        result = inference_video(
            args.video, model, config, device,
            stride=args.stride, threshold=args.threshold
        )
        print_results(result, 'inference')

    elif args.frames:
        # 프레임 폴더 추론
        result = inference_frames(
            args.frames, model, config, device,
            stride=args.stride, threshold=args.threshold
        )
        print_results(result, 'inference')

        # 시각화
        if args.visualize and result['prediction'] != -1:
            frame_dir = Path(args.frames)
            frames = sorted(frame_dir.glob('*.jpg'))[:16]

            if frames:
                frame_images = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
                               for f in frames]
                vis_path = os.path.join(
                    args.output_dir,
                    f"{frame_dir.name}_prediction.png"
                )
                visualize_prediction(
                    np.array(frame_images),
                    result['prediction'],
                    result['confidence'],
                    vis_path
                )


if __name__ == '__main__':
    main()
