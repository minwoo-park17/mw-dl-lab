"""
Inference Script
학습된 모델로 이미지에 대해 추론하고 마스크를 저장합니다.
"""

import sys
from pathlib import Path

# 패키지 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model import create_model
from src.utils import load_config, setup_device

# 얼굴 검출용 (선택적)
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False


SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


def get_face_detector(device='cuda'):
    """MTCNN 얼굴 검출기 생성"""
    if not MTCNN_AVAILABLE:
        print("Warning: facenet-pytorch not installed. Using center crop.")
        return None
    return MTCNN(
        keep_all=True,
        min_face_size=50,
        margin=0,
        thresholds=[0.85, 0.85, 0.85],
        device=device
    )


def crop_to_square_with_face(image: Image.Image, face_detector) -> Image.Image:
    """
    짧은 변 기준 정사각형 크롭 (얼굴이 포함되는 방향으로)

    Args:
        image: PIL Image (RGB)
        face_detector: MTCNN 검출기 (None이면 센터 크롭)

    Returns:
        정사각형 PIL Image
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
            boxes, _ = face_detector.detect(image, landmarks=False)
            if boxes is not None and len(boxes) > 0:
                # 가장 큰 얼굴 선택
                areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
                max_idx = np.argmax(areas)
                x1, y1, x2, y2 = boxes[max_idx]
                face_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        except Exception:
            pass

    if img_w > img_h:
        # 가로가 긴 경우
        crop_y1, crop_y2 = 0, img_h
        if face_center is not None:
            crop_x1 = face_center[0] - min_dim / 2
            crop_x1 = max(0, min(crop_x1, img_w - min_dim))
        else:
            crop_x1 = (img_w - min_dim) / 2
        crop_x2 = crop_x1 + min_dim
    else:
        # 세로가 긴 경우
        crop_x1, crop_x2 = 0, img_w
        if face_center is not None:
            crop_y1 = face_center[1] - min_dim / 2
            crop_y1 = max(0, min(crop_y1, img_h - min_dim))
        else:
            crop_y1 = (img_h - min_dim) / 2
        crop_y2 = crop_y1 + min_dim

    return image.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))


def load_model(checkpoint_path: str, arch_config_path: str, device: torch.device):
    """
    체크포인트에서 모델을 로드합니다.

    Args:
        checkpoint_path: 체크포인트 파일 경로
        arch_config_path: 아키텍처 설정 파일 경로
        device: 디바이스

    Returns:
        로드된 모델
    """
    arch_config = load_config(arch_config_path)
    model = create_model(arch_config)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    best_value = checkpoint.get('best_value', 'unknown')
    print(f"Loaded model from epoch {epoch} (best_value: {best_value})")

    return model


def get_image_list(image_dir: str):
    """이미지 파일 목록을 반환합니다."""
    image_dir = Path(image_dir)
    images = set()
    for ext in SUPPORTED_EXTENSIONS:
        images.update(image_dir.glob(f"*{ext}"))
        images.update(image_dir.glob(f"*{ext.upper()}"))
    return sorted(images)


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    원본 이미지와 마스크를 오버레이합니다.

    Args:
        image: 원본 이미지 (H, W, 3)
        mask: 이진 마스크 (H, W), 0 또는 1
        alpha: 마스크 투명도

    Returns:
        오버레이된 이미지
    """
    overlay = image.copy()

    # 마스크가 1인 영역에 파란색 오버레이
    blue_overlay = np.zeros_like(image)
    blue_overlay[:, :, 2] = 255  # Blue channel (RGB에서 index 2)

    mask_3d = np.stack([mask, mask, mask], axis=-1)
    overlay = np.where(mask_3d > 0,
                       (1 - alpha) * image + alpha * blue_overlay,
                       image).astype(np.uint8)

    return overlay


def pad_to_multiple(image: np.ndarray, multiple: int = 32) -> tuple:
    """
    이미지를 multiple의 배수 크기로 패딩합니다.

    Returns:
        (padded_image, (pad_h, pad_w))
    """
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h == 0 and pad_w == 0:
        return image, (0, 0)

    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    return padded, (pad_h, pad_w)


def run_inference(
    model,
    image_dir: str,
    output_dir: str,
    device: torch.device,
    resize: int = 1024,  # Validation과 동일 (1024)
    threshold: float = 0.5,
    save_overlay: bool = True,
    face_detector=None
):
    """
    디렉토리의 이미지들에 대해 추론을 수행하고 마스크를 저장합니다.
    (Validation과 동일한 방식: 정사각형 크롭 → resize로 리사이즈 → 추론)

    Args:
        model: 추론에 사용할 모델
        image_dir: 입력 이미지 디렉토리
        output_dir: 출력 디렉토리
        device: 디바이스
        resize: 추론 시 리사이즈 크기 (학습 시 validation resize와 동일하게)
        threshold: 이진화 임계값
        save_overlay: 오버레이 이미지 저장 여부
        face_detector: 얼굴 검출기 (None이면 비정사각형 이미지는 센터 크롭)
    """
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    if save_overlay:
        overlays_dir = output_dir / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)

    # 변환 생성 (Validation과 동일: Resize → Normalize)
    transform = A.Compose([
        A.Resize(height=resize, width=resize),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # 이미지 목록
    image_paths = get_image_list(image_dir)
    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images")
    print(f"Output directory: {output_dir}")
    if face_detector:
        print("Face-aware square crop: enabled")
    else:
        print("Face-aware square crop: disabled (center crop for non-square)")

    model.eval()
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Inference"):
            # 이미지 로드
            original_image = Image.open(img_path).convert('RGB')

            # 정사각형이 아니면 얼굴 기반 크롭 (또는 센터 크롭)
            cropped_image = crop_to_square_with_face(original_image, face_detector)
            cropped_np = np.array(cropped_image)
            cropped_size = cropped_np.shape[0]  # 정사각형이므로 H=W

            # 변환 적용 (Resize → Normalize)
            transformed = transform(image=cropped_np)
            image_tensor = transformed['image'].unsqueeze(0).to(device)

            # 추론
            output = model(image_tensor)
            prob = torch.sigmoid(output)
            pred = (prob > threshold).float()

            # 마스크를 원본 크롭 크기로 리사이즈
            mask_np = pred.squeeze().cpu().numpy()
            mask_resized = np.array(
                Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                    (cropped_size, cropped_size), Image.NEAREST
                )
            )
            mask_binary = (mask_resized > 127).astype(np.uint8)

            # 마스크 저장 (0: real, 255: fake)
            mask_save = (mask_binary * 255).astype(np.uint8)
            mask_image = Image.fromarray(mask_save, mode='L')
            mask_image.save(masks_dir / f"{img_path.stem}.png")

            # 오버레이 저장 (크롭된 이미지 기준)
            if save_overlay:
                overlay = create_overlay(cropped_np, mask_binary, alpha=0.4)
                overlay_image = Image.fromarray(overlay)
                overlay_image.save(overlays_dir / f"{img_path.stem}.png")

    print(f"\nInference complete!")
    print(f"  Masks saved to: {masks_dir}")
    if save_overlay:
        print(f"  Overlays saved to: {overlays_dir}")


if __name__ == "__main__":
    # =========================================================================
    # 설정
    # =========================================================================
    # 모델 경로 (학습 결과 폴더에서 best_model.pth 또는 last_model.pth)
    MODEL_PATH = rf"D:\study\mw-dl-lab\segmentation-lab\results\train_251226181859\best_model.pth"

    # 아키텍처 설정 (학습 시 사용한 설정 파일)
    ARCH_CONFIG = "config/architecture.yaml"

    # 입력 이미지 디렉토리
    INPUT_DIR = rf"C:\Users\sands\Pictures\Screenshots\core"

    # 출력 디렉토리 (마스크 저장 위치)
    OUTPUT_DIR = rf"D:\study\mw-dl-lab\segmentation-lab\results\train_251226181859\test_result_mask"

    # 추론 설정
    RESIZE = 1024               # 추론 시 리사이즈 크기 (학습 시 validation resize와 동일하게)
    THRESHOLD = 0.5             # 이진화 임계값
    SAVE_OVERLAY = True         # 오버레이 이미지 저장 여부
    USE_FACE_DETECTION = True   # 비정사각형 이미지에 얼굴 기반 크롭 적용

    # =========================================================================
    # 실행
    # =========================================================================
    device = setup_device()
    print(f"Using device: {device}")

    # 모델 로드
    model = load_model(MODEL_PATH, ARCH_CONFIG, device)

    # 얼굴 검출기 (비정사각형 이미지 처리용)
    face_detector = None
    if USE_FACE_DETECTION:
        print("Loading face detector...")
        face_detector = get_face_detector(device)

    # 추론 실행
    run_inference(
        model=model,
        image_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        device=device,
        resize=RESIZE,
        threshold=THRESHOLD,
        save_overlay=SAVE_OVERLAY,
        face_detector=face_detector
    )
