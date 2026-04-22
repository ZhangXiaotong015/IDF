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
        pred = self.model(x, adaptive_iter=adaptive_iter, max_iter=max_iter, alpha_schedule=alpha_schedule)
        pred = self.normalize(pred, reverse=True)
        return pred
        
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

        losses['train/loss'] = self.loss(pred, y)
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
        assert x.shape[0] == 1

        pred = self(x, adaptive_iter=self.misc_config.adaptive_iteration, 
                    max_iter=self.misc_config.max_iteration,
                    alpha_schedule=self.misc_config.get('alpha_schedule'))
        
        pred = torch.clamp(pred, 0.0, 1.0)

        # Evaluate metrics.
        losses = {}
        losses[f'val{suffix}/psnr'] = calculate_psnr_pt(y, pred, 0, test_y_channel=False).mean()
        losses[f'val{suffix}/ssim'] = calculate_ssim_pt(y, pred, 0, test_y_channel=False).mean()

        self.log_dict(losses, sync_dist=True, prog_bar=True, add_dataloader_idx=False)
        
        if batch_idx % 500 == 0:
            self.sampled_images.append(x[0].cpu())
            self.sampled_images.append(y[0].cpu())
            self.sampled_images.append(pred[0].cpu())

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
        assert x.shape[0] == 1

        pred = self(x, adaptive_iter=self.misc_config.adaptive_iteration,
                    max_iter=self.misc_config.max_iteration,
                    alpha_schedule=self.misc_config.get('alpha_schedule'))

        pred = torch.clamp(pred, 0.0, 1.0) # (1,3,H,W)

        save_dir = "/scratch/IDF/logs/LocalImageLogger/Team1_Results"
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, file_name.replace('.jpg','_T1.jpg'))

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

