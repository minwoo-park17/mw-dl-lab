"""
이미지 나이/성별 분류기
- 10대, 20-30대, 40대 이상으로 나이 분류
- 남/여 성별 분류
"""

import os
import shutil
from pathlib import Path

# 필요한 라이브러리 설치: pip install deepface opencv-python

def classify_images(input_folder: str, output_folder: str = None):
    """
    이미지 폴더를 입력받아 나이/성별별로 분류합니다.
    
    Args:
        input_folder: 분류할 이미지가 있는 폴더 경로
        output_folder: 분류된 이미지를 저장할 폴더 (None이면 input_folder/classified)
    """
    from deepface import DeepFace
    
    # 출력 폴더 설정
    if output_folder is None:
        output_folder = os.path.join(input_folder, "classified")
    
    # 분류 카테고리 폴더 생성
    categories = [
        "10대_남성", "10대_여성",
        "20-30대_남성", "20-30대_여성",
        "40대이상_남성", "40대이상_여성",
        "분류실패"
    ]
    
    for category in categories:
        Path(os.path.join(output_folder, category)).mkdir(parents=True, exist_ok=True)
    
    # 지원하는 이미지 확장자
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
    
    # 결과 저장용
    results = {cat: [] for cat in categories}
    
    # 이미지 파일 목록 가져오기
    image_files = [
        f for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f)) 
        and Path(f).suffix.lower() in image_extensions
    ]
    
    print(f"총 {len(image_files)}개의 이미지를 분류합니다...\n")
    
    for idx, filename in enumerate(image_files, 1):
        filepath = os.path.join(input_folder, filename)
        print(f"[{idx}/{len(image_files)}] 처리 중: {filename}")
        
        try:
            # DeepFace로 나이/성별 분석
            analysis = DeepFace.analyze(
                img_path=filepath,
                actions=['age', 'gender'],
                enforce_detection=True,
                silent=True
            )
            
            # 여러 얼굴이 감지된 경우 첫 번째 얼굴 사용
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            age = analysis['age']
            gender = analysis['dominant_gender']  # 'Man' or 'Woman'
            
            # 나이 그룹 분류
            if age < 20:
                age_group = "10대"
            elif age < 50:
                age_group = "20-40대"
            else:
                age_group = "50대이상"
            
            # 성별 한글 변환
            gender_kr = "남성" if gender == "Man" else "여성"
            
            # 카테고리 결정
            category = f"{age_group}_{gender_kr}"
            
            print(f"  → 나이: {age}세 ({age_group}), 성별: {gender_kr}")
            
        except Exception as e:
            print(f"  → 분류 실패: {str(e)}")
            category = "분류실패"
        
        # 파일 복사
        dest_path = os.path.join(output_folder, category, filename)
        shutil.copy2(filepath, dest_path)
        results[category].append(filename)
    
    # 결과 출력
    print("\n" + "="*50)
    print("분류 결과 요약")
    print("="*50)
    
    for category, files in results.items():
        if files:
            print(f"\n[{category}] - {len(files)}개")
            for f in files[:5]:  # 최대 5개만 표시
                print(f"  • {f}")
            if len(files) > 5:
                print(f"  ... 외 {len(files)-5}개")
    
    print(f"\n분류된 이미지 저장 위치: {output_folder}")
    return results


def analyze_single_image(image_path: str):
    """단일 이미지의 나이/성별을 분석합니다."""
    from deepface import DeepFace
    
    try:
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'gender'],
            enforce_detection=True,
            silent=True
        )
        
        if isinstance(analysis, list):
            print(f"감지된 얼굴 수: {len(analysis)}")
            for i, face in enumerate(analysis, 1):
                age = face['age']
                gender = "남성" if face['dominant_gender'] == "Man" else "여성"
                confidence = face['gender'][face['dominant_gender']]
                print(f"  얼굴 {i}: 나이 {age}세, 성별 {gender} (신뢰도: {confidence:.1f}%)")
        else:
            age = analysis['age']
            gender = "남성" if analysis['dominant_gender'] == "Man" else "여성"
            print(f"나이: {age}세, 성별: {gender}")
            
    except Exception as e:
        print(f"분석 실패: {str(e)}")


if __name__ == "__main__":
    print("="*50)
    print("이미지 나이/성별 분류기")
    print("="*50)
    
    # 폴더 경로 입력
    # input_path = input("\n분류할 이미지 폴더 경로를 입력하세요: ").strip()
    input_path = r""
    output_path = r""
    if not os.path.isdir(input_path):
        print("유효하지 않은 폴더 경로입니다.")
    # else:
    #     output_path = input("결과 저장 폴더 (Enter시 기본값): ").strip() or None
        classify_images(input_path, output_path)