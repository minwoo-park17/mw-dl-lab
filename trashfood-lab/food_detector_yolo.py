import torch
import timm
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as T
import os


class FoodClassifier:
    """
    Food101 pretrained EfficientNet 모델로 음식 이미지 분류
    """
    def __init__(self, device):
        self.device = device

        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=101
        ).to(device).eval()

        # Food101 클래스 이름 로드
        self.classes = self.load_food101_labels()

        # 이미지 변환
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    @staticmethod
    def load_food101_labels():
        # Food-101 dataset의 클래스 101개
        return [
            c.strip() for c in open("food101_classes.txt", "r", encoding="utf-8").readlines()
        ]

    def classify(self, img):
        """
        음식 이미지 한 장 → 음식 이름 반환
        """
        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(img)
            idx = pred.argmax(dim=1).item()

        return self.classes[idx]


class FoodDetector:
    """
    YOLO + Food classifier 통합 파이프라인
    """
    def __init__(self, yolo_model="yolov8s.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # YOLO 불러오기
        self.yolo = YOLO(yolo_model)

        # Food101 classifier
        self.food_classifier = FoodClassifier(self.device)

    def detect_foods(self, image_path):
        image = Image.open(image_path).convert("RGB")

        # YOLO detection 실행
        results = self.yolo(image)[0]

        foods = []

        for box in results.boxes:
            cls_id = int(box.cls)

            # 음식이 아닌 클래스는 무시
            # (YOLO 기본 모델은 다양한 사물을 탐지하므로 음식 가능성 높은 것만 필터링)
            if cls_id not in [40, 41, 42, 46, 47, 48, 49, 51, 52, 53, 55]:  
                # (YOLO80 클래스 중 음식 관련 일부)
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = image.crop((x1, y1, x2, y2))

            # classify food
            food_name = self.food_classifier.classify(crop)
            foods.append(food_name)

        # 중복 제거하고 정렬
        foods = sorted(list(set(foods)))

        return foods


if __name__ == "__main__":
    detector = FoodDetector()

    img_path = r"D:\Dataset\food\1.png"
    foods = detector.detect_foods(img_path)

    print("\nDetected foods:")
    for idx, f in enumerate(foods, 1):
        print(f"{idx}. {f}")
