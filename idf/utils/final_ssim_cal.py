import cv2
import os
import numpy as np



def ssim(img1, img2, window_size=11, sigma=1.5, K1=0.01, K2=0.03, L=255):

    """

    计算两幅图像的 SSIM（Structural Similarity Index Measure）

    img1, img2: 输入图像（numpy array），必须是相同尺寸的灰度图

    window_size: 高斯窗口大小

    sigma: 高斯核标准差

    K1, K2: 稳定常数

    L: 像素值范围（对于8位图像为255）

    """

    # 创建高斯核

    kernel_x = cv2.getGaussianKernel(window_size, sigma)

    kernel = kernel_x @ kernel_x.T


    # 提取图像通道（假设是灰度图）

    img1 = img1.astype(np.float64)

    img2 = img2.astype(np.float64)


    # 计算均值

    mu1 = cv2.filter2D(img1, -1, kernel)

    mu2 = cv2.filter2D(img2, -1, kernel)


    # 计算方差和协方差

    mu1_sq = mu1 ** 2

    mu2_sq = mu2 ** 2

    mu1_mu2 = mu1 * mu2



    sigma1_sq = cv2.filter2D(img1 ** 2, -1, kernel) - mu1_sq

    sigma2_sq = cv2.filter2D(img2 ** 2, -1, kernel) - mu2_sq

    sigma12 = cv2.filter2D(img1 * img2, -1, kernel) - mu1_mu2


    # SSIM 计算公式

    C1 = (K1 * L) ** 2

    C2 = (K2 * L) ** 2



    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)

    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)



    ssim_map = numerator / denominator

    return np.mean(ssim_map)



# 示例：读取图像并计算 SSIM

if __name__ == "__main__":

    # 读取图像（请替换为你自己的图像路径）

    # img_1 = cv2.imread('original_image.png', 0)   # 原始图像
    #
    # img_2 = cv2.imread('denoised_image.png', 0)   # 降噪后图像

    predPath = '/scratch/IDF/logs/LocalImageLogger/Team1_Results'
    oriPath = '/data/test_set_50'

    # 建立预测结果字典，key 用去掉 "_T1" 的文件名
    pred_dict = {}
    for root, _, files in os.walk(predPath):
        for f in files:
            # 去掉 "_T1" 部分，只保留原始文件名
            base_name = f.replace("_T1", "")
            pred_dict[base_name] = os.path.join(root, f)

    # 遍历原始文件夹，按文件名匹配
    ssim_list = []
    for root, _, files in os.walk(oriPath):
        for f in files:
            if f in pred_dict:  # 这里 f 是 "001.jpg"
                ori_file = os.path.join(root, f)
                pred_file = pred_dict[f]
                img_1 = cv2.imread(ori_file, 0)  # 原始图像
                img_2 = cv2.imread(pred_file, 0)  # 降噪后图像

                # 调用 SSIM 函数

                similarity = ssim(img_1, img_2)
                ssim_list.append(similarity)

                print(f"SSIM: {similarity:.4f}")
    final_ssim = np.array(ssim_list).mean()
    print(f"Final ssim is {final_ssim}!!!")
