from typing import Mapping, Any
import torch
from idf.utils.common import instantiate_from_config
from idf.utils.metrics import calculate_psnr_pt, calculate_ssim_pt
# from torchvision.transforms.functional import center_crop
from idf.models.lit_denoising import LitDenoising
from idf.utils.misc import const_like
import numpy as np
import cv2
import torchvision.utils as vutils
import os
import time
from torchvision.utils import make_grid

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

class LitADenoising(LitDenoising):
    def __init__(
        self,
        data_config: Mapping[str, Any],
        denoiser_config: Mapping[str, Any],
        loss_config: Mapping[str, Any],
        optimizer_config: Mapping[str, Any],
        scheduler_config: Mapping[str, Any] = None,
        misc_config: Mapping[str, Any] = None,
    ):
        super().__init__(data_config, denoiser_config, loss_config, 
                         optimizer_config, scheduler_config,misc_config,)

        self.model = instantiate_from_config(denoiser_config)
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"model size: {num_params/1e6:.2f} M")
        self.misc_config = misc_config
        if self.misc_config.compile:
            self.model = torch.compile(self.model)

        self.loss = instantiate_from_config(loss_config)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.data_config = data_config

        self.val_dataset_names = [k for k in self.data_config.validate.keys()]
        
        # data normalization
        self.data_scale = np.float32(data_config.norm.sigma_data) / np.float32(data_config.norm.raw_std)
        self.data_bias = np.float32(data_config.norm.mu_data) - np.float32(data_config.norm.raw_mean) * self.data_scale
        
        self.save_hyperparameters()

    def forward(self, noisy, adaptive_iter=False, max_iter=None, alpha_schedule=None):
        x = self.normalize(noisy)

        print(f"Allocated before inference: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

        pred = self.model(x, adaptive_iter=adaptive_iter, max_iter=max_iter, alpha_schedule=alpha_schedule)

        print(f"Allocated after inference: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

        pred = self.normalize(pred, reverse=True)

        # -------------------------
        # Post-processing Enhancement V1 (CLAHE + Unsharp Mask)
        # -------------------------
        enhanced_list = []
        for i in range(pred.size(0)):  # 遍历 batch
            img = pred[i].detach().cpu().numpy()  # (3, H, W)
            img = np.transpose(img, (1, 2, 0))  # 转成 (H, W, 3)
            img = (img * 255).astype(np.uint8)

            # 取灰度通道
            pred_gray = img[:, :, 0]

            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(pred_gray)

            # 转回RGB三通道
            enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)

            # Unsharp Mask
            # blurred = cv2.GaussianBlur(enhanced_rgb, (5, 5), 1.0)
            blurred = cv2.GaussianBlur(cv2.copyMakeBorder(enhanced_rgb, 5, 5, 5, 5, cv2.BORDER_REFLECT), (5, 5), 1.0)
            blurred = blurred[5:-5, 5:-5]  # 去掉填充区域
            alpha = 1.5
            sharpened = cv2.addWeighted(enhanced_rgb, 1 + alpha, blurred, -alpha, 0)

            # 转回 torch.Tensor (C, H, W)
            sharpened_tensor = torch.from_numpy(sharpened).float().permute(2, 0, 1) / 255.0
            enhanced_list.append(sharpened_tensor)

        # 拼回 batch
        enhanced_pred = torch.stack(enhanced_list, dim=0)

        # -------------------------
        # Post-processing Enhancement V2 (CLAHE + Controlled Unsharp Mask + Noise Suppression + Color Balance)
        # -------------------------
        # enhanced_list = []
        # for i in range(pred.size(0)):
        #     img = pred[i].detach().cpu().numpy()  # (3, H, W)
        #     img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
        #     img = (img * 255).astype(np.uint8)
        #
        #     # 灰度通道
        #     pred_gray = img[:, :, 0]
        #
        #     # CLAHE（更温和的局部对比度）
        #     clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(24, 24))
        #     enhanced_gray = clahe.apply(pred_gray)
        #
        #     # 防止黑块：亮度保护 + 归一化
        #     enhanced_gray = np.clip(enhanced_gray, 20, 235)
        #     enhanced_gray = cv2.normalize(enhanced_gray, None, 0, 255, cv2.NORM_MINMAX)
        #
        #     # 转回RGB
        #     enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        #
        #     # Controlled Unsharp Mask（降低锐化强度）
        #     blurred = cv2.GaussianBlur(enhanced_rgb, (5, 5), 1.2)
        #     alpha = 0.7
        #     sharpened = cv2.addWeighted(enhanced_rgb, 1 + alpha, blurred, -alpha, 0)
        #
        #     # 噪声抑制掩膜：边缘增强 + 平滑区域保持去噪
        #     edges = cv2.Canny(enhanced_gray, 30, 90)
        #     mask = cv2.GaussianBlur(edges, (7, 7), 2).astype(np.float32) / 255.0
        #
        #     # 分区融合：边缘用锐化，平滑区域用原始denoised
        #     layered = sharpened * mask[..., None] + enhanced_rgb * (1 - mask[..., None])
        #
        #     # 类型统一为 float32，避免 addWeighted 报错
        #     layered = layered.astype(np.float32)
        #     img = img.astype(np.float32)
        #
        #     # 通道平衡：防止单色块或偏色
        #     for c in range(3):
        #         mean_layer = layered[..., c].mean()
        #         mean_img = img[..., c].mean()
        #         scale = mean_img / (mean_layer + 1e-6)
        #         layered[..., c] = np.clip(layered[..., c] * scale, 0, 255)
        #
        #     # 最后再和 denoised 原图轻度融合，保持层次同时抑制噪声
        #     beta = 0.3  # 融合比例，可调
        #     final = cv2.addWeighted(layered, 1 - beta, img, beta, 0)
        #
        #     # 限制像素范围并转回 uint8
        #     final = np.clip(final, 0, 255).astype(np.uint8)
        #
        #     # 转回 torch.Tensor
        #     final_tensor = torch.from_numpy(final).float().permute(2, 0, 1) / 255.0
        #     enhanced_list.append(final_tensor)
        #
        # enhanced_pred = torch.stack(enhanced_list, dim=0)

        # -------------------------
        # Post-processing Enhancement V3 (CLAHE + Controlled Unsharp Mask + Noise Suppression + Color Balance + Block Detection Repair)
        # -------------------------
        # enhanced_list = []
        # for i in range(pred.size(0)):
        #     img = pred[i].detach().cpu().numpy()  # (3, H, W)
        #     img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
        #     img = (img * 255).astype(np.uint8)
        #
        #     # 灰度通道
        #     pred_gray = img[:, :, 0]
        #
        #     # CLAHE（更温和的局部对比度）
        #     clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(32, 32))  # 降低对比度增强强度
        #     enhanced_gray = clahe.apply(pred_gray)
        #
        #     # 防止黑块：亮度保护 + 归一化
        #     enhanced_gray = np.clip(enhanced_gray, 25, 230)
        #     enhanced_gray = cv2.normalize(enhanced_gray, None, 0, 255, cv2.NORM_MINMAX)
        #
        #     # 转回RGB
        #     enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        #
        #     # Controlled Unsharp Mask（降低锐化强度）
        #     blurred = cv2.GaussianBlur(enhanced_rgb, (5, 5), 1.5)
        #     alpha = 0.5  # 降低锐化系数
        #     sharpened = cv2.addWeighted(enhanced_rgb, 1 + alpha, blurred, -alpha, 0)
        #
        #     # 噪声抑制掩膜：边缘增强 + 平滑区域保持去噪
        #     edges = cv2.Canny(enhanced_gray, 40, 100)  # 提高阈值，减少误检
        #     mask = cv2.GaussianBlur(edges, (7, 7), 2).astype(np.float32) / 255.0
        #
        #     # 分区融合：边缘用锐化，平滑区域用原始denoised
        #     layered = sharpened * mask[..., None] + enhanced_rgb * (1 - mask[..., None])
        #
        #     # 类型统一为 float32
        #     layered = layered.astype(np.float32)
        #     img = img.astype(np.float32)
        #
        #     # 通道平衡：防止单色块或偏色
        #     for c in range(3):
        #         mean_layer = layered[..., c].mean()
        #         mean_img = img[..., c].mean()
        #         scale = mean_img / (mean_layer + 1e-6)
        #         layered[..., c] = np.clip(layered[..., c] * scale, 0, 255)
        #
        #     # 最后再和 denoised 原图轻度融合
        #     beta = 0.4  # 稍微提高融合比例，柔化过渡
        #     final = cv2.addWeighted(layered, 1 - beta, img, beta, 0)
        #
        #     # 限制像素范围并转回 uint8
        #     final = np.clip(final, 0, 255).astype(np.uint8)
        #
        #     # -------------------------
        #     # 保险机制：检测并修复单一色块
        #     # -------------------------
        #     # lap = cv2.Laplacian(cv2.cvtColor(final, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
        #     # mask_blocks = (np.abs(lap) < 5e-4).astype(np.uint8) * 255  # 提高阈值，减少误修复
        #     #
        #     # # 仅修复小面积色块
        #     # if mask_blocks.sum() < 0.05 * mask_blocks.size:
        #     #     final = cv2.inpaint(final, mask_blocks, 3, cv2.INPAINT_TELEA)
        #
        #     # 转回 torch.Tensor
        #     final_tensor = torch.from_numpy(final).float().permute(2, 0, 1) / 255.0
        #     enhanced_list.append(final_tensor)
        #
        # enhanced_pred = torch.stack(enhanced_list, dim=0)

        return pred, enhanced_pred
        
    def normalize(self, x, reverse=False):
        if not reverse:
            if self.data_scale is not None:
                x = x * const_like(x, self.data_scale).reshape(1, -1, 1, 1)
            if self.data_bias is not None:
                x = x + const_like(x, self.data_bias).reshape(1, -1, 1, 1)
        else:
            if self.data_scale is not None:
                x = x - const_like(x, self.data_bias).reshape(1, -1, 1, 1)
            if self.data_bias is not None:    
                x = x / const_like(x, self.data_scale).reshape(1, -1, 1, 1)

        return x
        
    @torch.no_grad()
    def get_input(self, batch, config, norm_data=True):
        x = batch[config.input_key]
        y = batch[config.target_key]
        if norm_data:
            x = self.normalize(x)
            y = self.normalize(y)
        return x, y
   
    def training_step(self, batch, batch_idx):
        x, y = self.get_input(batch, self.data_config.train)
        self.log("bs", self.global_batch_size, prog_bar=True, logger=False)
        self.log('lr', self.get_lr(), prog_bar=True, logger=False)

        losses = dict()

        pred = self.model(x)

        # losses['train/loss'] = self.loss(pred, y)
        # losses['train/loss'] = self.loss(pred, y, x)
        losses['train/loss'] = self.loss(pred, y, x, batch_idx)
        losses['train/total'] = sum(losses.values())
        self.log_dict(losses, prog_bar=True)
        return losses['train/total']
    
    def on_validation_start(self):
        self.sampled_images = []
        self.sample_steps_val = 50
        print(f"[Inference Settings] {self.misc_config.adaptive_iteration=}, {self.misc_config.max_iteration=}")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        val_name = self.val_dataset_names[dataloader_idx]
        val_config = self.data_config.validate[val_name]
        # self._validation_step(batch, batch_idx, val_config, suffix=f"_{val_name}")
        self._test_step(batch, batch_idx, val_config, suffix=f"_{val_name}")

    def on_validation_end(self):
        pass

    def _validation_step(self, batch, batch_idx, val_config, suffix=""):
        x, y = self.get_input(batch, val_config, norm_data=False)
        # assert x.shape[0] == 1

        pred, enhanced_pred = self(x, adaptive_iter=self.misc_config.adaptive_iteration,
                                    max_iter=self.misc_config.max_iteration,
                                    alpha_schedule=self.misc_config.get('alpha_schedule'))
        
        pred = torch.clamp(pred, 0.0, 1.0)
        enhanced_pred = torch.clamp(enhanced_pred, 0.0, 1.0)

        # Evaluate metrics.
        losses = {}
        losses[f'val{suffix}/psnr'] = calculate_psnr_pt(y, pred, 0, test_y_channel=False).mean()
        losses[f'val{suffix}/ssim'] = calculate_ssim_pt(y, pred, 0, test_y_channel=False).mean()

        self.log_dict(losses, sync_dist=True, prog_bar=True, add_dataloader_idx=False)
        
        # if batch_idx % 500 == 0:
        #     self.sampled_images.append(x[0].cpu())
        #     self.sampled_images.append(y[0].cpu())
        #     self.sampled_images.append(pred[0].cpu())
        #     if len(self.sampled_images) == 0:
        #         return
        #     rows = []
        #     # 每 3 张图为一组 (x, y, pred)
        #     for i in range(0, len(self.sampled_images), 3):
        #         triplet = self.sampled_images[i:i + 3]
        #         if len(triplet) == 3:
        #             row = make_grid(triplet, nrow=3)
        #             rows.append(row)
        #
        #     # 垂直拼接所有行
        #     final_grid = torch.cat(rows, dim=1)
        #
        #     self.log_image("sampled_images", final_grid, batch_idx)
        #
        #     self.sampled_images.clear()

        if batch_idx % 500 == 0:
            # 遍历 batch 内的所有样本
            for i in range(x.shape[0]):
                # 每个样本的三张图加入 sampled_images
                self.sampled_images.append(x[i].cpu())
                # self.sampled_images.append(y[i].cpu())
                self.sampled_images.append(pred[i].cpu())
                self.sampled_images.append(enhanced_pred[i].cpu())

                row = make_grid(self.sampled_images, nrow=len(self.sampled_images))
                self.log_image("sampled_images", row, batch_idx, sample_id=i)
                # 清空 sampled_images，为下一个样本准备
                self.sampled_images.clear()

    @torch.no_grad()
    def get_input_test(self, batch, config, norm_data=True):
        x = batch[config.input_key]
        y = batch[config.target_key]
        file_name = batch[config.filename_key]
        file_name = os.path.basename(file_name[0])
        if norm_data:
            x = self.normalize(x)
            y = self.normalize(y)
        return x, y, file_name

    def _test_step(self, batch, batch_idx, val_config, suffix=""):
        x, y, file_name = self.get_input_test(batch, val_config, norm_data=False)
        #assert x.shape[0] == 1
        start = time.perf_counter()
        pred = self(x, adaptive_iter=self.misc_config.adaptive_iteration,
                    max_iter=self.misc_config.max_iteration,
                    alpha_schedule=self.misc_config.get('alpha_schedule'))[1]
        end = time.perf_counter()
        print(f"Inference time: {end-start:.4f} seconds")
        pred = torch.clamp(pred, 0.0, 1.0) # (1,3,H,W)

        save_dir = "/scratch/IDF/logs/LocalImageLogger/Team1_Results_Round2"
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, file_name.split('_')[0], file_name.replace('.jpg','_T1.jpg'))

        vutils.save_image(pred, save_path)

        # # Evaluate metrics.
        # losses = {}
        # # variance = calculate_variance(pred)
        # # losses[f'val{suffix}/psnr'] =
        # losses[f'val{suffix}/psnr'] = calculate_psnr_pt(y, pred, 0, test_y_channel=False).mean()
        # losses[f'val{suffix}/ssim'] = calculate_ssim_pt(y, pred, 0, test_y_channel=False).mean()
        #
        # self.log_dict(losses, sync_dist=True, prog_bar=True, add_dataloader_idx=False)
        #
        # # self.sampled_images.append(x[0].cpu())
        # self.sampled_images.append(y[0].cpu())
        # self.sampled_images.append(pred[0].cpu())

