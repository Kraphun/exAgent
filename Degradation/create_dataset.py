import os
import json
import random
import numpy as np
from PIL import Image

# Hyper parameters
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.dirname(script_dir)

DATASET_ROOT = os.path.join(project_root, "dataset")
GENERATED_DIR_NAME = "Noised"  # Denoise 폴더 내부에 생성될 폴더명

# [고정 프롬프트] Agent가 학습할 명령어 지정
FIXED_PROMPT = "Analyze the image degradation. Output the result in the strict report format."

# 폴더 매핑
# Key: 데이터셋 경로, Value: 처리 타입
# "denoise" 타입은 Clean 데이터를 의미하며, 노이즈를 생성하여 추가합니다.
FOLDER_TO_TYPE_MAP = {
    "Denoise/BSD400": "denoise",       # Clean -> Clean 라벨링 + Noise 생성
    "Derain/rain100L/rainy": "derain", # 기존 데이터 그대로 사용
}

LABELING_MAP = {
    "clean": "Clean",
    "denoise": "Noised",
    "derain": "Rain Streak",
}

OUTPUT_FILE = "train_data_augmented.json"

# 가우시안 노이즈 추가로 노이즈 데이터셋 생성
def add_gaussian_noise(image_path, save_path, mean=0, sigma=25):
    """
    이미지를 읽어 가우시안 노이즈를 추가하고 저장합니다.
    sigma 값으로 노이즈의 강도를 조절합니다.
    """
    try:
        # 이미지 로드 및 배열 변환
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        # 노이즈 생성 (이미지와 같은 크기)
        gauss = np.random.normal(mean, sigma, img_array.shape)
        
        # 이미지에 노이즈 더하기
        noisy_img_array = img_array + gauss
        
        # 0~255 사이 값으로 자르기 (Clip) 및 정수형 변환
        noisy_img_array = np.clip(noisy_img_array, 0, 255).astype('uint8')
        
        # 이미지 저장
        noisy_img = Image.fromarray(noisy_img_array)
        noisy_img.save(save_path)
        return True
    
    except Exception as e:
        print(f"⚠️ 노이즈 생성 실패 ({image_path}): {e}")
        return False

# =========================================================
# 3. 데이터셋 생성 로직
# =========================================================
def create_dataset():
    final_data = []
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    print(f"📂 데이터셋 생성을 시작합니다...")

    for folder_name, degradation_type in FOLDER_TO_TYPE_MAP.items():
        folder_path = os.path.join(DATASET_ROOT, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"⚠️ 폴더 없음: {folder_path}")
            continue
            
        images = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_extensions]
        print(f"   -> [{folder_name}] 처리 중... ({len(images)}장)")

        if degradation_type == "denoise":
            # 경로 파싱: "Denoise/BSD400" -> "Denoise"
            root_category = folder_name.split("/")[0] 
            noise_save_dir = os.path.join(DATASET_ROOT, root_category, GENERATED_DIR_NAME)
            os.makedirs(noise_save_dir, exist_ok=True)
        else:
            noise_save_dir = None

        for img_file in images:
            # 원본 이미지 경로 -> 절대경로
            src_path = os.path.join(folder_path, img_file).replace("\\", "/")            

            if degradation_type == "denoise":
                # 1. Clean 원본 데이터 추가
                add_entry(final_data, src_path, "Clean")
                
                # 2. Gaussian Noise 생성 및 데이터 추가
                noise_filename = f"noise_{img_file}"
                full_save_path = os.path.join(noise_save_dir, noise_filename)
                
                sigma = 25
                if add_gaussian_noise(src_path, full_save_path, sigma=sigma):
                    noise_path = full_save_path.replace("\\", "/")
                    add_entry(final_data, noise_path, "Gaussian Noise")
                    
            # denoise 외
            else:
                add_entry(final_data, src_path, degradation_type)

    # JSON 파일 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 완료! 총 {len(final_data)}개의 데이터 쌍이 '{OUTPUT_FILE}'에 저장되었습니다.")

def add_entry(data_list, image_path, type_label):
    """JSON 리스트에 데이터를 추가하는 헬퍼 함수"""
    is_clean = type_label == "Clean"
    
    if is_clean:
        response = (
            "- Degradation Detected: No\n"
            "- Type: None\n"
            "- Severity: None\n"
            "- Description: The image is clear without degradation."
        )
    else:
        # type_label을 LABELING_MAP을 이용해 매핑하거나 그대로 사용
        display_label = type_label
        if type_label == "Gaussian Noise":
            display_label = "Noised" # or keep "Gaussian Noise" based on preference
            
        response = (
            f"- Degradation Detected: Yes\n"
            f"- Type: {display_label}\n"
            f"- Severity: Medium\n"
            f"- Description: Detected {display_label} artifacts in the image."
        )

    entry = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": FIXED_PROMPT}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            }
        ]
    }
    data_list.append(entry)

if __name__ == "__main__":
    create_dataset()