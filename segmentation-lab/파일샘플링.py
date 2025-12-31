import os
import shutil
import random

def random_copy(src_dir, dst_dir, n):
    # 소스 경로의 파일 목록
    files = os.listdir(src_dir)
    
    # N개 랜덤 샘플링
    sampled = random.sample(files, min(n, len(files)))
    
    # 대상 폴더 생성
    os.makedirs(dst_dir, exist_ok=True)
    
    # 복사
    for f in sampled:
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        shutil.copy2(src, dst)
    
    print(f"{len(sampled)}개 복사 완료")

# 사용
copy_path = rf"D:\Dataset\image\vsln_20251219\origin\real\real\noasian\crawling"
paste_path = rf"D:\Dataset\image\vsln_segmentation_20251226\origin\real"
counts = 308
random_copy(copy_path, paste_path, counts)