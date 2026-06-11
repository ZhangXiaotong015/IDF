import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pydicom.multival import MultiValue
import argparse

def apply_window(image, window_center, window_width):
    """应用窗宽窗位到像素值"""
    img = image.astype(np.float32)
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min) * 255.0
    return img.astype(np.uint8)

def normalize(image):
    """自动归一化到 0-255"""
    img = image.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return img.astype(np.uint8)

def get_window_value(value):
    """兼容 DSfloat 和 MultiValue"""
    if value is None:
        return None
    if isinstance(value, MultiValue):
        return float(value[0])
    return float(value)

def convert_dicom(input_path, output_dir, base_name):
    try:
        dicom_data = pydicom.dcmread(input_path)
        pixel_array = dicom_data.pixel_array

        wc = get_window_value(getattr(dicom_data, "WindowCenter", None))
        ww = get_window_value(getattr(dicom_data, "WindowWidth", None))

        def process_frame(frame, suffix=""):
            # 如果是 RGB 图像 (H, W, 3)，先转灰度
            if frame.ndim == 3 and frame.shape[-1] == 3:
                frame = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])  # RGB 转灰度

            if wc is not None and ww is not None and frame.ndim == 2:
                frame = apply_window(frame, wc, ww)
            elif frame.ndim == 2:
                frame = normalize(frame)

            output_path = os.path.join(output_dir, f"{base_name}{suffix}.jpg")
            plt.imsave(output_path, frame, cmap='gray')

        # 多帧情况 (N, H, W)
        if pixel_array.ndim == 3 and pixel_array.shape[-1] != 3:
            num_frames = pixel_array.shape[0]
            for i in range(num_frames):
                process_frame(pixel_array[i, :, :], suffix=f"_frame{i}")
            print(f"Converted {input_path} -> {num_frames} frames saved")

        # 单帧灰度 (H, W)
        elif pixel_array.ndim == 2:
            process_frame(pixel_array)
            print(f"Converted {input_path} -> single frame saved")

        # 单帧 RGB (H, W, 3)
        elif pixel_array.ndim == 3 and pixel_array.shape[-1] == 3:
            process_frame(pixel_array)
            print(f"Converted {input_path} -> single RGB frame saved")

        else:
            print(f"Unexpected shape {pixel_array.shape} for {input_path}")

    except Exception as e:
        print(f"Failed to convert {input_path}: {e}")

def process_directory(root_dir, output_root):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if '.' not in filename:  # 没有后缀的文件
                input_path = os.path.join(dirpath, filename)

                # 构造输出路径，保持相同的目录结构
                relative_path = os.path.relpath(dirpath, root_dir)
                output_dir = os.path.join(output_root, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                convert_dicom(input_path, output_dir, filename)

if __name__ == "__main__":
    # 1. 创建参数解析器
    parser = argparse.ArgumentParser(description="将 DICOM 转换为 JPG 格式的脚本")

    # 2. 添加参数（设置了默认值，如果你不传参数，就会使用默认值）
    parser.add_argument('--input_dir', type=str, default="/data/test_set_round2", help='输入文件夹路径')
    parser.add_argument('--output_dir', type=str, default="/data/test_set_jpg_round2", help='输出文件夹路径')

    # 3. 解析参数
    args = parser.parse_args()

    # 4. 传入函数执行
    process_directory(args.input_dir, args.output_dir)

