import numpy as np
import cv2
import os

def calculate_variance(image):
    """
    计算图像的方差（Variance）
    """
    img_flat = image.flatten()
    variance = np.var(img_flat)
    return variance

def calculate_psnr_from_variance(variance, max_pixel=255.0):
    """
    根据方差计算 PSNR
    PSNR = 10 * log10( (max_pixel^2) / variance )
    """
    if variance == 0:
        return float('inf')  # 完全无噪声图像
    psnr = 10 * np.log10((max_pixel ** 2) / variance)
    return psnr

# 示例使用
if __name__ == '__main__':
    # 读取图像（灰度图）
    # image = cv2.imread(r'D:\denoise\BDCNN\TestData\Thorax\0001_20240530171448.png', 0)  # 替换为你的图像路径
    # image = cv2.imread('/scratch/IDF/logs/LocalImageLogger/test_set_jpg/050_T1.jpg', 0)
    # image = cv2.imread('/data/test_set_50/cfd_25/spine/050.jpg', 0)

    predPath = '/scratch/IDF/logs/LocalImageLogger/Team1_Results'
    # predPath = '/data/test_set_50'
    psnr_list = []
    for root, dirs, files in os.walk(predPath):
        for f in files:
            print(os.path.basename(f))
            image = cv2.imread(os.path.join(root,f), 0)

            if image is None:
                print("图像读取失败，请检查路径是否正确。")
            else:
                # 计算图像的方差
                variance = calculate_variance(image)
                print(f"图像的方差（Variance）: {variance:.4f}")

                # 使用方差作为 MSE，计算 PSNR
                max_pixel_value = 255.0  # 8-bit 图像
                psnr = calculate_psnr_from_variance(variance, max_pixel_value)
                psnr_list.append(psnr)
                print(f"基于方差的 PSNR: {psnr:.4f} dB")

    final_psnr = np.array(psnr_list).mean()
    print(f"Final PSNR is {final_psnr} dB!!!")
