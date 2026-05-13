import os
import numpy as np
from PIL import Image
import random

def compute_mean_std(root_dir, max_images=2000):
    # 收集所有图像路径
    img_paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif")):
                img_paths.append(os.path.join(root, f))

    print(f"Found {len(img_paths)} images, shuffling and taking first {max_images}")

    # 打乱
    random.shuffle(img_paths)

    # 只取前 max_images 张
    img_paths = img_paths[:max_images]

    sum_rgb = np.zeros(3, dtype=np.float64)
    sum_sq_rgb = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for path in img_paths:
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

        img_flat = img.reshape(-1, 3)

        sum_rgb += img_flat.sum(axis=0)
        sum_sq_rgb += (img_flat ** 2).sum(axis=0)
        pixel_count += img_flat.shape[0]

    mean = sum_rgb / pixel_count
    std = np.sqrt(sum_sq_rgb / pixel_count - mean ** 2)

    return mean, std


if __name__ == "__main__":
    root = "/data/Site_JPG_xray_training"
    raw_mean, raw_std = compute_mean_std(root, max_images=2000)

    print("raw_mean:", raw_mean)
    print("raw_std:", raw_std)
# Found 48784 images, shuffling and taking first 2000
# raw_mean: [0.55234206 0.55234206 0.55234206]
# raw_std: [0.19113504 0.19113504 0.19113504]

