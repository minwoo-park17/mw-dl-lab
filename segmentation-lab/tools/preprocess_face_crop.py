"""
얼굴 기반 정사각형 크롭 전처리 스크립트
- MTCNN으로 얼굴 검출
- 얼굴 중심으로 정사각형 크롭 (짧은 변 기준)
- 얼굴 없으면 센터 크롭
"""

import os
import glob
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

import warnings
warnings.simplefilter("ignore", category=FutureWarning)


def get_face_detector(device: str = None):
    """MTCNN 얼굴 검출기 초기화"""
    try:
        from facenet_pytorch import MTCNN
    except ImportError:
        raise ImportError(
            "facenet-pytorch가 설치되지 않았습니다.\n"
            "설치: pip install facenet-pytorch"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return MTCNN(
        margin=0,
        thresholds=[0.85, 0.85, 0.85],
        device=device
    )


def crop_to_square_with_face(
    image: Image.Image,
    face_detector
) -> Image.Image:
    """
    짧은 변 기준 정사각형 크롭 (얼굴이 포함되는 방향으로)

    - 크롭 크기: 항상 짧은 변 (min_dim)
    - 얼굴 있으면: 얼굴이 포함되도록 크롭 위치 조정
    - 얼굴 없으면: 센터 크롭

    Args:
        image: PIL Image (RGB)
        face_detector: MTCNN 검출기 (None이면 센터 크롭)

    Returns:
        정사각형 PIL Image (min_dim x min_dim)
    """
    img_w, img_h = image.size
    min_dim = min(img_w, img_h)

    # 이미 정사각형이면 그대로 반환
    if img_w == img_h:
        return image

    # 얼굴 검출 시도
    face_center = None
    if face_detector is not None:
        try:
            boxes, confidences = face_detector.detect(image, landmarks=False)
            if boxes is not None and len(boxes) > 0:
                # 가장 큰 얼굴 선택
                areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
                max_idx = np.argmax(areas)
                x1, y1, x2, y2 = boxes[max_idx]
                face_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        except Exception:
            pass

    if img_w > img_h:
        # 가로가 긴 경우: 세로 전체 사용, 가로 위치만 결정
        crop_y1 = 0
        crop_y2 = img_h  # = min_dim

        if face_center is not None:
            # 얼굴 중심 기준으로 가로 위치 조정
            face_x = face_center[0]
            crop_x1 = face_x - min_dim / 2
            # 경계 보정
            if crop_x1 < 0:
                crop_x1 = 0
            elif crop_x1 + min_dim > img_w:
                crop_x1 = img_w - min_dim
        else:
            # 센터 크롭
            crop_x1 = (img_w - min_dim) / 2

        crop_x2 = crop_x1 + min_dim

    else:
        # 세로가 긴 경우: 가로 전체 사용, 세로 위치만 결정
        crop_x1 = 0
        crop_x2 = img_w  # = min_dim

        if face_center is not None:
            # 얼굴 중심 기준으로 세로 위치 조정
            face_y = face_center[1]
            crop_y1 = face_y - min_dim / 2
            # 경계 보정
            if crop_y1 < 0:
                crop_y1 = 0
            elif crop_y1 + min_dim > img_h:
                crop_y1 = img_h - min_dim
        else:
            # 센터 크롭
            crop_y1 = (img_h - min_dim) / 2

        crop_y2 = crop_y1 + min_dim

    # 크롭
    crop_x1, crop_y1 = int(crop_x1), int(crop_y1)
    crop_x2, crop_y2 = int(crop_x2), int(crop_y2)
    cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    return cropped


def process_directory(
    input_dir: str,
    output_dir: str,
    face_detector,
    resize_to: int = None,
    extensions: tuple = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')
):
    """
    디렉토리 내 모든 이미지를 얼굴 기반 정사각형으로 크롭

    Args:
        input_dir: 입력 디렉토리
        output_dir: 출력 디렉토리
        face_detector: MTCNN 검출기
        resize_to: 리사이즈할 크기 (None이면 원본 크기 유지)
        extensions: 지원 확장자
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 이미지 파일 찾기 (재귀, 중복 제거)
    image_files = set()
    for ext in extensions:
        image_files.update(input_path.glob(f"**/*{ext}"))
        image_files.update(input_path.glob(f"**/*{ext.upper()}"))
    image_files = sorted(image_files)

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images")

    success_count = 0
    fail_count = 0
    no_face_count = 0

    for img_path in tqdm(image_files, desc="Processing"):
        try:
            # 상대 경로 유지
            rel_path = img_path.relative_to(input_path)
            out_path = output_path / rel_path

            # 출력 디렉토리 생성
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # 이미지 로드
            image = Image.open(img_path).convert("RGB")

            # 얼굴 기반 정사각형 크롭
            cropped = crop_to_square_with_face(image, face_detector)

            # 리사이즈 (옵션)
            if resize_to is not None:
                w, h = cropped.size
                if w > resize_to or h > resize_to:
                    cropped = cropped.resize((resize_to, resize_to), Image.BILINEAR)

            # 저장
            if out_path.suffix.lower() in ('.jpg', '.jpeg'):
                cropped.save(out_path, quality=95)
            else:
                cropped.save(out_path)

            success_count += 1

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            fail_count += 1

    print(f"\nComplete: {success_count} success, {fail_count} failed")


if __name__ == "__main__":
    # =========================================================================
    # 설정
    # =========================================================================
    INPUT_DIR = r"D:\Dataset\image\vsln_segmentation_20251226\origin\real"      # 입력 디렉토리
    OUTPUT_DIR = r"D:\Dataset\image\vsln_segmentation_20251226\origin\real"    # 출력 디렉토리

    RESIZE_TO = None        # 리사이즈 크기 (None이면 원본 크기 유지)
    USE_FACE_DETECTION = True  # False면 센터 크롭만

    # =========================================================================
    # 실행
    # =========================================================================
    face_detector = None
    if USE_FACE_DETECTION:
        print("Loading face detector...")
        face_detector = get_face_detector()

    process_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        face_detector=face_detector,
        resize_to=RESIZE_TO
    )

    print("모든 이미지 처리 완료!")
