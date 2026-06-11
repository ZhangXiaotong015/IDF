import os
import shutil
import random

# ===== 路径配置 =====
cardiac_dir = "/data/filtered_cardiac_neuro_data_jpg_training/Cardiac_JPG"
neuro_dir = "/data/filtered_cardiac_neuro_data_jpg_training/Neuro_JPG"
output_dir = "/data/filtered_cardiac_neuro_pseudo_gt_subset"

os.makedirs(output_dir, exist_ok=True)

# ===== 目标数量 =====
cardiac_target = 300
neuro_target = 200

# ===== 是否加随机扰动（推荐开启）=====
use_random_offset = True
random.seed(42)  # 可复现


def uniform_sample(src_dir, target_num, prefix):
    # 获取并排序文件（按文件名 → 时间序列）
    files = sorted([
        f for f in os.listdir(src_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    total = len(files)
    step = total / target_num

    selected = []

    for i in range(target_num):
        idx = int(i * step)

        # 加随机偏移（避免固定采样点）
        if use_random_offset:
            offset = random.randint(0, max(0, int(step) - 1))
            idx = min(idx + offset, total - 1)

        selected.append(files[idx])

    # 去重（防止极少数重复）
    selected = list(dict.fromkeys(selected))

    # 复制到新目录
    for i, fname in enumerate(selected):
        src_path = os.path.join(src_dir, fname)
        dst_name = f"{prefix}_{i:04d}.jpg"
        dst_path = os.path.join(output_dir, dst_name)
        shutil.copy(src_path, dst_path)

    print(f"{prefix}: selected {len(selected)} images")


# ===== 执行 =====
uniform_sample(cardiac_dir, cardiac_target, "cardiac")
uniform_sample(neuro_dir, neuro_target, "neuro")

print("✅ Done! ~500 images saved in:", output_dir)
