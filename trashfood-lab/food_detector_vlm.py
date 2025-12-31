from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image, ImageOps
import torch
from glob import glob
import os

class FoodDetector:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b"):
        """
        BLIP-2 모델 초기화
        - blip2-opt-2.7b: 가벼운 버전 (권장)
        - blip2-opt-6.7b: 더 정확하지만 무거움
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(self.device)

    def detect_foods(self, image_path: str) -> list[str]:
        """
        이미지에서 음식 목록 추출
        """
        image = Image.open(image_path).convert("RGB")

        # ✅ 1024x1024로 맞추기 (비율 유지 + 패딩)
        image = ImageOps.contain(image, (768, 768), method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (768, 768), (0, 0, 0))  # 패딩 색(검정)
        canvas.paste(
            image,
            ((768 - image.width) // 2, (768 - image.height) // 2)
        )
        image = canvas

        prompt = "List the ingredients you can recognize in the food in this image : "

        inputs = self.processor(image, text=prompt, return_tensors="pt").to(
            self.device, torch.float32
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=5,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=3,     # ✅ 반복 구절 억제
            repetition_penalty=1.15,    # ✅ 같은 토큰 반복 페널티
            length_penalty=0.8          # (선택) 너무 길게 늘어지는 답 억제
        )

        answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()

        return answer


if __name__ == "__main__":
    image_root = rf"D:\Dataset\food"

    image_paths = glob(os.path.join(image_root, "*.png"))
    detector = FoodDetector()

    for idx, image_path in enumerate(image_paths):       
        basename = os.path.basename(image_path)
        foods = detector.detect_foods(image_path)

        print(f"{basename} : {foods}")