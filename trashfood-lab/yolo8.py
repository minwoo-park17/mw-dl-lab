import torch
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
import os

# ---------------------------------------------------------
#  식재료 후보 리스트 (원하면 더 추가 가능)
# ---------------------------------------------------------
INGREDIENTS = [
    "carrot", "onion", "garlic", "ginger", "tomato", "lettuce", "cabbage", "potato",
    "sweet potato", "radish", "cucumber", "zucchini", "pumpkin", "broccoli",
    "cauliflower", "spinach", "mushroom", "bell pepper", "chili pepper",
    "eggplant", "green onion", "shallot", "leek",
    "chicken", "beef", "pork", "lamb", "duck", "turkey",
    "shrimp", "crab", "lobster", "salmon", "tuna", "mackerel",
    "egg", "cheese", "milk", "yogurt", "butter", "cream",
    "rice", "noodles", "bread", "pasta", "tofu",
    "beans", "lentils", "peas", "corn"
]

# ---------------------------------------------------------
#   CLIP Zero-shot Classifier
# ---------------------------------------------------------
class ClipIngredientClassifier:
    def __init__(self, device):
        print("Loading CLIP...")
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # 미리 텍스트 임베딩 계산
        with torch.no_grad():
            text_inputs = self.processor(
                text=[f"a photo of {i}" for i in INGREDIENTS],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            self.text_features = self.model.get_text_features(**text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def classify(self, image: Image.Image):
        """ crop 이미지를 받아 식재료 이름 추론 """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            img_features = self.model.get_image_features(**inputs)
            img_features /= img_features.norm(dim=-1, keepdim=True)

        similarity = (img_features @ self.text_features.T).softmax(dim=-1)
        idx = similarity.argmax().item()
        return INGREDIENTS[idx], similarity[0][idx].item()


# ---------------------------------------------------------
#    Main Detector (YOLO + CLIP)
# ---------------------------------------------------------
class IngredientDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        print("Loading YOLOv8 food detector...")
        self.yolo = YOLO("food-ingredients-yolov8n.pt")   # 작은 모델로 빠른 추론 가능

        self.clip = ClipIngredientClassifier(self.device)

    def detect_ingredients(self, img_path):
        results = self.yolo(img_path)[0]
        image = Image.open(img_path).convert("RGB")

        detected = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = image.crop((x1, y1, x2, y2))
            name, score = self.clip.classify(crop)

            detected.append((name, score))

        return detected


# ---------------------------------------------------------
#   Run Test
# ---------------------------------------------------------
if __name__ == "__main__":
    detector = IngredientDetector()

    img_path = r"D:\Dataset\food\1.png"

    ingredients = detector.detect_ingredients(img_path)

    print("\nDetected Ingredients:")
    for name, score in ingredients:
        print(f"- {name}  ({score:.2f})")
