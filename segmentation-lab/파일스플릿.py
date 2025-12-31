import os
import shutil
import random

def split_dataset(src_dir, dst_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, shuffle=True):
    # 비율 검증
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "비율 합이 1이어야 함"
    
    # 파일 목록
    files = os.listdir(src_dir)
    if shuffle:
        random.shuffle(files)
    
    # 분할 인덱스 계산
    n = len(files)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    splits = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }
    
    # 복사
    for split_name, split_files in splits.items():
        split_dir = os.path.join(dst_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for f in split_files:
            src = os.path.join(src_dir, f)
            dst = os.path.join(split_dir, f)
            shutil.copy2(src, dst)
        
        print(f"{split_name}: {len(split_files)}개")

# 사용
split_dataset(
    src_dir=rf"D:\Dataset\image\vsln_segmentation_20251226\origin\fake",
    dst_dir=rf"D:\Dataset\image\vsln_segmentation_20251226\split\fake",
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1
)