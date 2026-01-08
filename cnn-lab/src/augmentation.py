import random
import io
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms


# ============================================================
# 1. SimpleColorBlend
# ============================================================
class SimpleColorBlend:
    """Simple color blending augmentation."""

    def __init__(self, p=0.001):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_cv = np.array(img)[:, :, ::-1].copy()

        r = random.randint(50, 255)
        g = random.randint(50, 255)
        b = random.randint(50, 255)
        overlay_color = (b, g, r)

        bg_overlay = np.full_like(img_cv, overlay_color, dtype=np.uint8)

        alpha = random.uniform(0.6, 0.9)
        beta = 1 - alpha

        blended = cv2.addWeighted(img_cv, alpha, bg_overlay, beta, 0)

        return Image.fromarray(blended[:, :, ::-1])


# ============================================================
# 2. RandomDownUp
# ============================================================
class RandomDownUp:
    """랜덤 다운스케일 후 target_size로 업스케일 (해상도 손실 시뮬레이션)

    EfficientNet-B4용: MTCNN crop 입력 → 250~원본 범위 다운스케일 → 380px 업스케일
    """
    def __init__(self, min_size=250, target_size=380, p=0.5):
        self.min_size = min_size
        self.target_size = target_size
        self.p = p
        self.methods = [Image.BILINEAR, Image.BICUBIC]

    def __call__(self, img):
        if random.random() > self.p:
            return img.resize((self.target_size, self.target_size), Image.BILINEAR)

        w, h = img.size
        min_dim = min(w, h)

        if min_dim <= self.min_size:
            down_size = self.min_size
        else:
            # 원본 크기까지 다운스케일 가능 (250 ~ min_dim)
            down_size = random.randint(self.min_size, min_dim)

        scale = down_size / min_dim
        new_w, new_h = int(w * scale), int(h * scale)

        method = random.choice(self.methods)

        img_down = img.resize((new_w, new_h), method)
        img_up = img_down.resize((self.target_size, self.target_size), method)

        return img_up


# ============================================================
# 2. JPEGCompression
# ============================================================
class JPEGCompression:
    """JPEG 압축 시뮬레이션"""
    def __init__(self, quality_range=(65, 95), p=1.0):
        self.quality_range = quality_range
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        quality = random.randint(*self.quality_range)
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        return Image.open(buffer).convert('RGB')


# ============================================================
# 3. CenterBiasedRandomCrop
# ============================================================
class CenterBiasedRandomCrop:
    """중앙(얼굴) 영역을 높은 확률로 포함하는 RandomCrop"""
    def __init__(self, output_size=380, center_bias=0.7):
        self.output_size = output_size
        self.center_bias = center_bias
    
    def __call__(self, img):
        w, h = img.size
        crop_size = self.output_size
        
        if w < crop_size or h < crop_size:
            return img.resize((crop_size, crop_size), Image.BILINEAR)
        
        if w == crop_size and h == crop_size:
            return img
        
        max_offset_x = w - crop_size
        max_offset_y = h - crop_size
        
        if random.random() < self.center_bias:
            offset_range_x = max(1, max_offset_x // 4)
            offset_range_y = max(1, max_offset_y // 4)
            
            center_left = max_offset_x // 2
            center_top = max_offset_y // 2
            
            left = center_left + random.randint(-offset_range_x, offset_range_x)
            top = center_top + random.randint(-offset_range_y, offset_range_y)
            
            left = max(0, min(left, max_offset_x))
            top = max(0, min(top, max_offset_y))
        else:
            left = random.randint(0, max_offset_x)
            top = random.randint(0, max_offset_y)
        
        return img.crop((left, top, left + crop_size, top + crop_size))

class SimpleRandomCrop:
    """MTCNN 후처리용 - 단순 랜덤 크롭으로 다양성 확보"""
    def __init__(self, output_size=380):
        self.output_size = output_size
    
    def __call__(self, img):
        w, h = img.size
        
        if w < self.output_size or h < self.output_size:
            return img.resize((self.output_size, self.output_size), Image.BILINEAR)
        
        if w == self.output_size and h == self.output_size:
            return img
        
        left = random.randint(0, w - self.output_size)
        top = random.randint(0, h - self.output_size)
        
        return img.crop((left, top, left + self.output_size, top + self.output_size))

# ============================================================
# 4. GridShuffle
# ============================================================
class GridShuffle:
    """이미지를 grid로 나눠 일부 셀을 셔플"""
    def __init__(self, grid_size=4, shuffle_ratio=0.25, p=0.25):
        self.grid_size = grid_size
        self.shuffle_ratio = shuffle_ratio
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        gh = h // self.grid_size
        gw = w // self.grid_size
        
        cells = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = img_np[i*gh:(i+1)*gh, j*gw:(j+1)*gw].copy()
                cells.append(cell)
        
        n_cells = len(cells)
        n_shuffle = max(2, int(n_cells * self.shuffle_ratio))
        
        shuffle_indices = random.sample(range(n_cells), n_shuffle)
        
        temp = cells[shuffle_indices[0]].copy()
        for k in range(len(shuffle_indices) - 1):
            cells[shuffle_indices[k]] = cells[shuffle_indices[k + 1]].copy()
        cells[shuffle_indices[-1]] = temp
        
        result = img_np.copy()
        idx = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                result[i*gh:(i+1)*gh, j*gw:(j+1)*gw] = cells[idx]
                idx += 1
        
        return Image.fromarray(result)


# ============================================================
# Train Transform (Real / Fake 비대칭)
# ============================================================

class TrainTransformReal:
    """Real 이미지용 학습 Transform (EfficientNet-B4: 380x380)"""
    def __init__(self, img_size=380,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

        self.transform = transforms.Compose([
            RandomDownUp(min_size=250, target_size=img_size, p=0.8),
            JPEGCompression(quality_range=(60, 90), p=0.8),
            # GridShuffle(grid_size=7, shuffle_ratio=0.75, p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transform(img)


class TrainTransformFake:
    """Fake 이미지용 학습 Transform (더 aggressive, EfficientNet-B4: 380x380)"""
    def __init__(self, img_size=380,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

        self.transform = transforms.Compose([
            RandomDownUp(min_size=250, target_size=img_size, p=0.9),
            JPEGCompression(quality_range=(60, 90), p=0.8),
            # GridShuffle(grid_size=7, shuffle_ratio=0.75, p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transform(img)


# ============================================================
# Validation Transform (Real / Fake 비대칭)
# ============================================================

class ValTransformReal:
    """Real 이미지용 검증 Transform (EfficientNet-B4: 380x380)"""
    def __init__(self, img_size=380,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

        self.transform = transforms.Compose([
            RandomDownUp(min_size=250, target_size=img_size, p=0.7),
            JPEGCompression(quality_range=(80, 100), p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transform(img)


class ValTransformFake:
    """Fake 이미지용 검증 Transform (EfficientNet-B4: 380x380)"""
    def __init__(self, img_size=380,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

        self.transform = transforms.Compose([
            RandomDownUp(min_size=250, target_size=img_size, p=0.7),
            JPEGCompression(quality_range=(80, 100), p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transform(img)


# ============================================================
# Test Transform (Augmentation 최소화 - Resize만)
# ============================================================

class TestTransform:
    """테스트용 Transform - 직접 Resize (EfficientNet-B4: 380x380)"""
    def __init__(self, img_size=380,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transform(img)


# ============================================================
# Dataset 사용 예시
# ============================================================

class DeepfakeDataset:
    """Dataset 클래스 예시"""
    def __init__(self, image_paths, labels, mode='train'):
        """
        Args:
            image_paths: 이미지 경로 리스트
            labels: 레이블 리스트 (0: Real, 1: Fake)
            mode: 'train', 'val', 'test'
        """
        self.image_paths = image_paths
        self.labels = labels
        self.mode = mode
        
        if mode == 'train':
            self.transform_real = TrainTransformReal()
            self.transform_fake = TrainTransformFake()
        elif mode == 'val':
            self.transform_real = ValTransformReal()
            self.transform_fake = ValTransformFake()
        else:  # test
            self.transform = TestTransform()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.mode == 'test':
            img = self.transform(img)
        else:
            if label == 0:  # Real
                img = self.transform_real(img)
            else:           # Fake
                img = self.transform_fake(img)
        
        return img, label

if __name__=="__main__":
    pass

# ============================================================
# 파이프라인 요약 (EfficientNet-B4: 380x380)
# ============================================================
"""
┌─────────────────────────────────────────────────────────────┐
│         EfficientNet-B4 파이프라인 (생성형 AI 탐지)           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  입력: MTCNN 2x crop → RandomDownUp → 직접 380px           │
│  (Crop 없음 - 외곽 정보 보존)                                │
│                                                             │
│  Normalize: mean=[0.485, 0.456, 0.406]                     │
│             std=[0.229, 0.224, 0.225]  (ImageNet 기준)      │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Train]                                                    │
│  ┌─────────────────┬─────────────────┐                     │
│  │      Real       │      Fake       │                     │
│  ├─────────────────┼─────────────────┤                     │
│  │ RandomDownUp    │ RandomDownUp    │                     │
│  │  min=250, p=0.8 │  min=220, p=0.9 │                     │
│  │  → 380x380      │  → 380x380      │                     │
│  ├─────────────────┼─────────────────┤                     │
│  │ JPEG 60~90      │ JPEG 60~90      │                     │
│  │  p=0.8          │  p=0.8          │                     │
│  └─────────────────┴─────────────────┘                     │
│                                                             │
│  [Validation]                                               │
│  ┌─────────────────┬─────────────────┐                     │
│  │      Real       │      Fake       │                     │
│  ├─────────────────┼─────────────────┤                     │
│  │ RandomDownUp    │ RandomDownUp    │                     │
│  │  min=250, p=0.7 │  min=220, p=0.7 │                     │
│  │  → 380x380      │  → 380x380      │                     │
│  ├─────────────────┼─────────────────┤                     │
│  │ JPEG 80~100     │ JPEG 80~100     │  ← 더 높은 품질     │
│  │  p=0.8          │  p=0.8          │                     │
│  └─────────────────┴─────────────────┘                     │
│                                                             │
│  [Test]                                                     │
│  ┌─────────────────────────────────────┐                   │
│  │ Resize(380) - 직접 리사이즈         │                   │
│  │ Augmentation 없음                   │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
"""