import os
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

def apply_rain_effect(base_img, mask_img, alpha_strength=0.7):
    base_np = np.asarray(base_img, dtype=np.float32) / 255.0
    mask_np = np.asarray(mask_img.resize(base_img.size), dtype=np.float32) / 255.0

    # 배경색 기준 알파 생성
    bg_color = np.median(mask_np.reshape(-1, 3), axis=0)
    diff     = np.linalg.norm(mask_np - bg_color, axis=2)
    alpha    = np.clip((diff - 0.05) * 4.0, 0, 1)**0.7
    alpha    = np.clip(alpha * alpha_strength, 0, 1)

    # 비 색상 (회색 계열)
    gray     = mask_np.mean(axis=2, keepdims=True)
    rain_rgb = np.concatenate([gray, gray, gray], axis=2)
    rain_rgb = np.asarray(
        Image.fromarray((rain_rgb*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.5)),
        dtype=np.float32
    ) / 255.0

    # Screen 블렌딩
    blend = 1 - (1 - base_np) * (1 - rain_rgb * alpha[..., None])
    return (blend * 255).astype(np.uint8)

def process_rain_synthesis(orig_dir, mask_dir, save_dir, alpha_strength=0.7):
    os.makedirs(save_dir, exist_ok=True)

    files = sorted(os.listdir(orig_dir))
    for filename in files:
        orig_path = os.path.join(orig_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        if not os.path.exists(mask_path):
            print(f"[!] 마스크 없음: {filename} → 건너뜀")
            continue

        base_img = Image.open(orig_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("RGB")
        result_np = apply_rain_effect(base_img, mask_img, alpha_strength)
        
        save_path = os.path.join(save_dir, filename)
        Image.fromarray(result_np).save(save_path)
        print(f"[✔] 저장됨: {save_path}")

# 🔧 경로 설정 예시
orig_dir = "/local_datasets/KITTI/object/training/origin_2"
mask_dir = "/local_datasets/KITTI/object/training/rain_mask"
save_dir = "/local_datasets/KITTI/object/training/rainy"

# 실행
process_rain_synthesis(orig_dir, mask_dir, save_dir, alpha_strength=0.7)
