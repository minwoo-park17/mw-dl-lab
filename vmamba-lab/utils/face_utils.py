"""Face detection and processing utilities."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


class FaceDetector:
    """Face detector wrapper supporting multiple backends."""

    def __init__(
        self,
        detector_type: str = "mtcnn",
        device: Optional[torch.device] = None,
        min_face_size: int = 20,
        thresholds: Optional[List[float]] = None,
    ):
        """Initialize face detector.

        Args:
            detector_type: Type of detector ('mtcnn', 'retinaface', 'opencv')
            device: Torch device for neural network detectors
            min_face_size: Minimum face size to detect
            thresholds: Detection thresholds
        """
        self.detector_type = detector_type.lower()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_face_size = min_face_size
        self.thresholds = thresholds or [0.6, 0.7, 0.7]

        self._init_detector()

    def _init_detector(self):
        """Initialize the selected detector."""
        if self.detector_type == "mtcnn":
            try:
                from facenet_pytorch import MTCNN

                self.detector = MTCNN(
                    image_size=160,
                    margin=0,
                    min_face_size=self.min_face_size,
                    thresholds=self.thresholds,
                    factor=0.709,
                    post_process=False,
                    device=self.device,
                    keep_all=True,
                )
            except ImportError:
                raise ImportError("Please install facenet-pytorch: pip install facenet-pytorch")

        elif self.detector_type == "retinaface":
            try:
                from retinaface import RetinaFace

                self.detector = RetinaFace
            except ImportError:
                raise ImportError("Please install retinaface: pip install retinaface")

        elif self.detector_type == "opencv":
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)

        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")

    def detect(
        self,
        image: Union[np.ndarray, torch.Tensor],
        return_landmarks: bool = False,
    ) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
        """Detect faces in image.

        Args:
            image: Input image (H, W, C) in RGB format
            return_landmarks: Whether to return facial landmarks

        Returns:
            Tuple of (bounding_boxes, landmarks)
            bounding_boxes: List of [x1, y1, x2, y2] arrays
            landmarks: List of landmark arrays or None
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)

        boxes = []
        landmarks = None

        if self.detector_type == "mtcnn":
            detected_boxes, probs, detected_landmarks = self.detector.detect(image, landmarks=True)

            if detected_boxes is not None:
                boxes = detected_boxes.tolist()
                if return_landmarks and detected_landmarks is not None:
                    landmarks = detected_landmarks.tolist()

        elif self.detector_type == "retinaface":
            resp = self.detector.detect_faces(image)
            for face_id, face_data in resp.items():
                box = face_data["facial_area"]
                boxes.append([box[0], box[1], box[2], box[3]])
                if return_landmarks:
                    if landmarks is None:
                        landmarks = []
                    lm = face_data["landmarks"]
                    landmarks.append([
                        lm["left_eye"],
                        lm["right_eye"],
                        lm["nose"],
                        lm["mouth_left"],
                        lm["mouth_right"],
                    ])

        elif self.detector_type == "opencv":
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            detected = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
            )
            for (x, y, w, h) in detected:
                boxes.append([x, y, x + w, y + h])

        return boxes, landmarks


def extract_faces(
    image: np.ndarray,
    boxes: List[List[float]],
    output_size: Tuple[int, int] = (256, 256),
    margin: float = 0.3,
) -> List[np.ndarray]:
    """Extract and crop face regions from image.

    Args:
        image: Input image (H, W, C)
        boxes: List of bounding boxes [x1, y1, x2, y2]
        output_size: Output face size (width, height)
        margin: Margin ratio to add around face

    Returns:
        List of cropped face images
    """
    faces = []
    h, w = image.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # Add margin
        bw, bh = x2 - x1, y2 - y1
        margin_x = int(bw * margin)
        margin_y = int(bh * margin)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)

        # Crop and resize
        face = image[y1:y2, x1:x2]
        if face.size > 0:
            face = cv2.resize(face, output_size)
            faces.append(face)

    return faces


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: Tuple[int, int] = (256, 256),
    eye_center_y_ratio: float = 0.35,
) -> np.ndarray:
    """Align face using eye landmarks.

    Args:
        image: Input image (H, W, C)
        landmarks: Facial landmarks with at least left_eye and right_eye
        output_size: Output image size
        eye_center_y_ratio: Vertical position ratio for eyes

    Returns:
        Aligned face image
    """
    # Get eye positions (assuming landmarks[0] = left_eye, landmarks[1] = right_eye)
    left_eye = np.array(landmarks[0])
    right_eye = np.array(landmarks[1])

    # Calculate angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Calculate center and scale
    eye_center = (left_eye + right_eye) / 2
    eye_dist = np.linalg.norm(right_eye - left_eye)
    desired_eye_dist = output_size[0] * 0.35
    scale = desired_eye_dist / eye_dist

    # Rotation matrix
    M = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale)

    # Adjust translation
    M[0, 2] += output_size[0] / 2 - eye_center[0]
    M[1, 2] += output_size[1] * eye_center_y_ratio - eye_center[1]

    # Warp
    aligned = cv2.warpAffine(
        image, M, output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return aligned


def extract_frames_from_video(
    video_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    num_frames: Optional[int] = None,
    frame_interval: int = 1,
    start_frame: int = 0,
) -> List[np.ndarray]:
    """Extract frames from video file.

    Args:
        video_path: Path to video file
        output_dir: Optional directory to save frames
        num_frames: Number of frames to extract (None for all)
        frame_interval: Extract every N-th frame
        start_frame: Starting frame index

    Returns:
        List of extracted frames as numpy arrays
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    frame_idx = 0
    extracted_count = 0

    # Set start position
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)

            extracted_count += 1
            if num_frames and extracted_count >= num_frames:
                break

        frame_idx += 1

    cap.release()
    return frames


def get_face_from_video(
    video_path: Union[str, Path],
    detector: FaceDetector,
    num_frames: int = 32,
    output_size: Tuple[int, int] = (256, 256),
) -> List[np.ndarray]:
    """Extract faces from video frames.

    Args:
        video_path: Path to video file
        detector: FaceDetector instance
        num_frames: Number of frames to extract
        output_size: Output face size

    Returns:
        List of face images
    """
    frames = extract_frames_from_video(
        video_path,
        num_frames=num_frames * 2,  # Extract more to ensure enough faces
        frame_interval=1,
    )

    faces = []
    for frame in frames:
        boxes, _ = detector.detect(frame)
        if boxes:
            # Take the largest face
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            best_idx = np.argmax(areas)
            extracted = extract_faces(frame, [boxes[best_idx]], output_size)
            if extracted:
                faces.append(extracted[0])

        if len(faces) >= num_frames:
            break

    return faces[:num_frames]
