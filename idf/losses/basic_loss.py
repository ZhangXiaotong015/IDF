# https://github.com/XPixelGroup/BasicSR

import torch
from torch import nn as nn
from torch.nn import functional as F

from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)



class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)



class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)



class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)



class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss

# ------------------------------------------------
#  Gradient Loss（保结构，不让边缘变糊）
# ------------------------------------------------
class GradientLoss(nn.Module):
    def __init__(self, loss_weight=0.1):
        super().__init__()
        self.loss_weight = loss_weight

    def gradient(self, x):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return gx, gy

    def forward(self, pred, target):
        gx_p, gy_p = self.gradient(pred)
        gx_t, gy_t = self.gradient(target)
        loss = torch.mean(torch.abs(gx_p - gx_t)) + torch.mean(torch.abs(gy_p - gy_t))
        return self.loss_weight * loss


# ------------------------------------------------
# Mean-Std Loss（保持亮度/对比度稳定）
# ------------------------------------------------
class MeanStdLoss(nn.Module):
    def __init__(self, loss_weight=0.01):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        mean_p, mean_t = pred.mean(), target.mean()
        std_p, std_t = pred.std(), target.std()
        loss = torch.abs(mean_p - mean_t) + torch.abs(std_p - std_t)
        return self.loss_weight * loss


# ------------------------------------------------
# Frangi Loss（轻量版血管增强）
# ------------------------------------------------
class FrangiLoss(nn.Module):
    def __init__(self, loss_weight=1e-3, sigmas=[1.0, 2.0, 3.0]):
        super().__init__()
        self.loss_weight = loss_weight
        self.sigmas = sigmas

    def to_gray(self, x):
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        return x

    def gaussian_blur(self, x, sigma):
        k = int(3 * sigma)
        device = x.device

        coords = torch.arange(-k, k+1, device=device).float()
        g = torch.exp(-(coords**2) / (2 * sigma * sigma))
        g = g / g.sum()

        g_h = g.view(1, 1, 1, -1)
        g_v = g.view(1, 1, -1, 1)

        C = x.shape[1]
        g_h = g_h.repeat(C, 1, 1, 1)
        g_v = g_v.repeat(C, 1, 1, 1)

        x = F.conv2d(x, g_h, padding=(0, k), groups=C)
        x = F.conv2d(x, g_v, padding=(k, 0), groups=C)
        return x

    def hessian(self, x, sigma):
        x = self.gaussian_blur(x, sigma)

        dxx = x[:, :, :, 2:] - 2 * x[:, :, :, 1:-1] + x[:, :, :, :-2]   # (N,1,H,W-2)
        dyy = x[:, :, 2:, :] - 2 * x[:, :, 1:-1, :] + x[:, :, :-2, :]   # (N,1,H-2,W)
        dxy = (x[:, :, 1:, 1:] - x[:, :, :-1, 1:]
               - x[:, :, 1:, :-1] + x[:, :, :-1, :-1])                  # (N,1,H-1,W-1)

        # 统一裁到 (H-2, W-2)
        dxx_c = dxx[:, :, 1:-1, :]          # (N,1,H-2,W-2)
        dyy_c = dyy[:, :, :, 1:-1]          # (N,1,H-2,W-2)
        dxy_c = dxy[:, :, :-1, :-1]         # (N,1,H-2,W-2)

        return dxx_c, dyy_c, dxy_c

    def frangi_response(self, x):
        x = self.to_gray(x)
        responses = []
        for sigma in self.sigmas:
            dxx, dyy, dxy = self.hessian(x, sigma)

            tmp = torch.sqrt((dxx - dyy)**2 + 4 * dxy**2 + 1e-12)
            lambda1 = 0.5 * (dxx + dyy + tmp)
            lambda2 = 0.5 * (dxx + dyy - tmp)

            Rb = (lambda1 / (lambda2 + 1e-12))**2
            S2 = lambda1**2 + lambda2**2

            response = torch.exp(-Rb / 0.5) * (1 - torch.exp(-S2 / 0.5))
            responses.append(response)

        return torch.max(torch.stack(responses, dim=0), dim=0)[0]

    def forward(self, pred, target):
        Fp = self.frangi_response(pred)
        Ft = self.frangi_response(target)
        return self.loss_weight * torch.mean((Fp - Ft)**2)


# ---------------------------
#  Gaussian blur for smoothing pseudo-GT
# ---------------------------
def gaussian_blur(x, sigma=1.0):
    k = int(3 * sigma)
    device = x.device

    coords = torch.arange(-k, k+1, device=device).float()
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()

    g_h = g.view(1, 1, 1, -1)
    g_v = g.view(1, 1, -1, 1)

    C = x.shape[1]
    g_h = g_h.repeat(C, 1, 1, 1)
    g_v = g_v.repeat(C, 1, 1, 1)

    x = F.conv2d(x, g_h, padding=(0, k), groups=C)
    x = F.conv2d(x, g_v, padding=(k, 0), groups=C)
    return x

# ------------------------------------------------
# Final Combined Loss
# ------------------------------------------------
class FinalLoss(nn.Module):
    def __init__(
        self,
        grad_weight=0.1,
        tv_weight=1e-5,
        frangi_weight=1e-3,
        brightness_weight=0.01
    ):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.tv = WeightedTVLoss()

        self.grad_loss = GradientLoss(loss_weight=grad_weight)
        self.frangi_loss = FrangiLoss(loss_weight=frangi_weight)
        self.brightness_loss = MeanStdLoss(loss_weight=brightness_weight)

        self.tv_weight = tv_weight

    def forward(self, pred, pseudo_gt, noisy):
        residual = pred - noisy

        L_main = self.charbonnier(pred, pseudo_gt)
        L_grad = self.grad_loss(pred, pseudo_gt)
        L_tv = self.tv(residual) * self.tv_weight
        L_frangi = self.frangi_loss(pred, pseudo_gt)
        L_brightness = self.brightness_loss(pred, pseudo_gt)

        loss = L_main + L_grad + L_tv + L_frangi + L_brightness
        return loss

class FinalLossV2(nn.Module):
    def __init__(
        self,
        grad_weight=0.2,
        tv_weight=3e-5,
        frangi_weight=1e-3,
        brightness_weight=0.01,
        pseudo_sigma=1.0,
        decay_tau=3000
    ):
        super().__init__()

        self.charbonnier = CharbonnierLoss()
        self.grad_loss = GradientLoss(loss_weight=grad_weight)
        self.frangi_loss = FrangiLoss(loss_weight=frangi_weight)
        self.brightness_loss = MeanStdLoss(loss_weight=brightness_weight)
        self.tv = WeightedTVLoss()

        self.tv_weight = tv_weight
        self.pseudo_sigma = pseudo_sigma
        self.decay_tau = decay_tau

    def forward(self, pred, pseudo_gt, noisy, step):
        # 1. Smooth pseudo-GT to remove its high-frequency noise
        pseudo_gt_smooth = gaussian_blur(pseudo_gt, sigma=self.pseudo_sigma)

        # 2. Dynamic weight: trust pseudo-GT less as training goes on
        w = torch.exp(torch.tensor(-step / self.decay_tau, device=pred.device))

        # 3. Main loss (with decaying weight)
        L_main = w * self.charbonnier(pred, pseudo_gt_smooth)

        # 4. Structural priors
        L_grad = self.grad_loss(pred, pseudo_gt_smooth)
        L_frangi = self.frangi_loss(pred, pseudo_gt_smooth)
        L_brightness = self.brightness_loss(pred, pseudo_gt_smooth)

        # 5. TV on residual

        loss = L_main + L_grad + L_frangi + L_brightness + L_tv
        return loss
        residual = pred - noisy
        L_tv = self.tv(residual) * self.tv_weight

class FinalLossV2_VesselBoostX(nn.Module):
    def __init__(
        self,
        grad_weight=0.25,
        tv_weight=3e-5,
        frangi_weight=1e-3,
        brightness_weight=0.01,
        pseudo_sigma=1.0,
        decay_tau=3000,
        vessel_dark_weight=0.15,      # 血管更黑
        vessel_edge_weight=0.15,      # 血管边缘更锐利
        vessel_contrast_weight=0.10   # 血管与背景对比度更高
    ):
        super().__init__()

        # 原始 V2 组件
        self.charbonnier = CharbonnierLoss()
        self.grad_loss = GradientLoss(loss_weight=grad_weight)
        self.frangi_loss = FrangiLoss(loss_weight=frangi_weight)
        self.brightness_loss = MeanStdLoss(loss_weight=brightness_weight)
        self.tv = WeightedTVLoss()

        self.tv_weight = tv_weight
        self.pseudo_sigma = pseudo_sigma
        self.decay_tau = decay_tau

        # 血管增强权重
        self.vessel_dark_weight = vessel_dark_weight
        self.vessel_edge_weight = vessel_edge_weight
        self.vessel_contrast_weight = vessel_contrast_weight

        # ⭐ 使用你的 FrangiLoss 作为血管检测器
        self.frangi_detector = FrangiLoss(loss_weight=0.0)


    def forward(self, pred, pseudo_gt, noisy, step):
        # 1. Smooth pseudo-GT
        pseudo_gt_smooth = gaussian_blur(pseudo_gt, sigma=self.pseudo_sigma)

        # 2. Dynamic weight
        w = torch.exp(torch.tensor(-step / self.decay_tau, device=pred.device))

        # 3. Main loss
        L_main = w * self.charbonnier(pred, pseudo_gt_smooth)

        # 4. Structural priors
        L_grad = self.grad_loss(pred, pseudo_gt_smooth)
        L_frangi = self.frangi_loss(pred, pseudo_gt_smooth)
        L_brightness = self.brightness_loss(pred, pseudo_gt_smooth)

        # 5. TV on residual
        residual = pred - noisy
        L_tv = self.tv(residual) * self.tv_weight

        # ------------------------------------------------
        # 6. 血管增强（核心）
        # ------------------------------------------------

        # 6.1 计算 Frangi 响应（血管响应图）
        vessel_response = self.frangi_detector.frangi_response(noisy)
        vessel_response = vessel_response / (vessel_response.max() + 1e-6)

        # 6.2 阈值生成血管 mask（注意：尺寸比 pred 小）
        vessel_mask = (vessel_response > 0.2).float()

        # ⭐ 修复尺寸 mismatch：resize 回 pred 尺寸
        vessel_mask = F.interpolate(
            vessel_mask,
            size=pred.shape[-2:],
            mode='nearest'
        )

        # 6.3 血管更黑（强力压暗）
        L_vessel_dark = (pred * vessel_mask).mean() * self.vessel_dark_weight

        # 6.4 血管边缘更锐利（梯度强化）
        L_vessel_edge = self.grad_loss(
            pred * vessel_mask,
            pseudo_gt_smooth * vessel_mask
        ) * self.vessel_edge_weight

        # 6.5 血管 vs 背景对比度增强
        vessel_region = pred * vessel_mask
        background_region = pred * (1 - vessel_mask)
        L_vessel_contrast = (
            vessel_region.mean() - background_region.mean()
        ) * self.vessel_contrast_weight

        # ------------------------------------------------
        # 总 loss
        # ------------------------------------------------
        loss = (
            L_main + L_grad + L_frangi + L_brightness + L_tv +
            L_vessel_dark + L_vessel_edge + L_vessel_contrast
        )

        return loss


def gaussian_kernel_1d(sigma, order, device):
    radius = int(3 * sigma)
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)

    if order == 0:
        g = torch.exp(-(x**2) / (2 * sigma**2))
        g = g / g.sum()
        return g

    elif order == 1:
        g = -x * torch.exp(-(x**2) / (2 * sigma**2))
        g = g / torch.sum(x * g * (-1))  # normalize
        return g

    elif order == 2:
        g = (x**2 - sigma**2) * torch.exp(-(x**2) / (2 * sigma**2))
        g = g - g.mean()
        g = g / g.abs().sum()
        return g

def conv2d_separable(img, kx, ky):
    # img: [B,1,H,W]
    B, C, H, W = img.shape
    kx = kx.view(1, 1, 1, -1)
    ky = ky.view(1, 1, -1, 1)

    out = F.conv2d(img, kx, padding=(0, kx.shape[-1]//2))
    out = F.conv2d(out, ky, padding=(ky.shape[-2]//2, 0))
    return out

# ---------------------------
# 二阶高斯导数
# ---------------------------
def gaussian_second_derivative_xx(img, sigma):
    device = img.device
    g2 = gaussian_kernel_1d(sigma, order=2, device=device)
    g0 = gaussian_kernel_1d(sigma, order=0, device=device)
    return conv2d_separable(img, g2, g0)

def gaussian_second_derivative_yy(img, sigma):
    device = img.device
    g2 = gaussian_kernel_1d(sigma, order=2, device=device)
    g0 = gaussian_kernel_1d(sigma, order=0, device=device)
    return conv2d_separable(img, g0, g2)

def gaussian_second_derivative_xy(img, sigma):
    device = img.device
    g1 = gaussian_kernel_1d(sigma, order=1, device=device)
    return conv2d_separable(img, g1, g1)

def hessian_line_filter(img, sigmas=[0.8, 1.2]):
    responses = []
    for sigma in sigmas:
        Hxx = gaussian_second_derivative_xx(img, sigma)
        Hxy = gaussian_second_derivative_xy(img, sigma)
        Hyy = gaussian_second_derivative_yy(img, sigma)

        tmp = torch.sqrt((Hxx - Hyy)**2 + 4 * Hxy**2)
        l1 = 0.5 * (Hxx + Hyy + tmp)
        l2 = 0.5 * (Hxx + Hyy - tmp)

        # 细线响应：|l2| 大，|l1| 小
        R = torch.exp(-(l1**2) / (2 * (0.5 * sigma)**2)) * \
            (1 - torch.exp(-(l2**2) / (2 * (1.5 * sigma)**2)))

        responses.append(R.abs())

    return torch.stack(responses).max(dim=0)[0]

def orientation_consistency(img):
    # img: [B,1,H,W]

    # Sobel kernels
    sobel_x = torch.tensor([[-1,0,1],
                            [-2,0,2],
                            [-1,0,1]], dtype=torch.float32, device=img.device).view(1,1,3,3)

    sobel_y = torch.tensor([[-1,-2,-1],
                            [ 0, 0, 0],
                            [ 1, 2, 1]], dtype=torch.float32, device=img.device).view(1,1,3,3)

    # 保证输出尺寸 = 输入尺寸
    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)

    angle = torch.atan2(gy, gx)  # [B,1,H,W]

    # 局部方向一致性（窗口内方差越小越一致）
    # 使用 avg_pool2d 代替复杂操作
    mean = F.avg_pool2d(angle, kernel_size=5, stride=1, padding=2)
    mean2 = F.avg_pool2d(angle * angle, kernel_size=5, stride=1, padding=2)
    var = mean2 - mean * mean

    # 一致性 = 1 - 方差
    return 1 - var

class FinalLossV3(nn.Module):  # Guidewire Version
    def __init__(
        self,
        grad_weight=0.2,
        tv_weight=3e-5,
        frangi_weight=1e-3,
        brightness_weight=0.01,
        pseudo_sigma=1.0,
        decay_tau=3000,
        guidewire_dark_weight=0.05,
    ):
        super().__init__()

        self.charbonnier = CharbonnierLoss()
        self.grad_loss = GradientLoss(loss_weight=grad_weight)
        self.frangi_loss = FrangiLoss(loss_weight=frangi_weight)
        self.brightness_loss = MeanStdLoss(loss_weight=brightness_weight)
        self.tv = WeightedTVLoss()

        self.tv_weight = tv_weight
        self.pseudo_sigma = pseudo_sigma
        self.decay_tau = decay_tau
        self.guidewire_dark_weight = guidewire_dark_weight


    # ---------------------------
    # 精准导丝检测（V2）
    # ---------------------------
    def detect_guidewire(self, noisy):
        # noisy: [B,3,H,W] → 转灰度
        if noisy.shape[1] == 3:
            img = noisy.mean(dim=1, keepdim=True).detach()
        else:
            img = noisy.detach()

        # 细线滤波
        R_line = hessian_line_filter(img)

        # 方向一致性（修复版）
        R_ori = orientation_consistency(img)

        # 综合响应
        R = R_line * R_ori
        R = R / (R.max() + 1e-6)

        mask = (R > 0.25).float()
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)

        return mask

    def forward(self, pred, pseudo_gt, noisy, step):
        # 1. Smooth pseudo-GT
        pseudo_gt_smooth = gaussian_blur(pseudo_gt, sigma=self.pseudo_sigma)

        # 2. Dynamic weight
        w = torch.exp(torch.tensor(-step / self.decay_tau, device=pred.device))

        # 3. Main loss
        L_main = w * self.charbonnier(pred, pseudo_gt_smooth)

        # 4. Structural priors
        L_grad = self.grad_loss(pred, pseudo_gt_smooth)
        L_frangi = self.frangi_loss(pred, pseudo_gt_smooth)
        L_brightness = self.brightness_loss(pred, pseudo_gt_smooth)

        # 5. TV on residual
        residual = pred - noisy
        L_tv = self.tv(residual) * self.tv_weight

        # ---------------------------
        # 6. 导丝变黑（关键）
        # ---------------------------
        guidewire_mask = self.detect_guidewire(noisy)

        # pred * mask 越小越好 → 导丝越黑
        L_dark = (pred * guidewire_mask).mean() * self.guidewire_dark_weight

        # ---------------------------
        # 总 loss
        # ---------------------------
        loss = L_main + L_grad + L_frangi + L_brightness + L_tv + L_dark
        return loss

class FinalLossV3Plus(nn.Module):  # Guidewire Contrast Version
    def __init__(
        self,
        grad_weight=0.25,
        tv_weight=3e-5,
        frangi_weight=1.5e-3,
        brightness_weight=0.02,
        pseudo_sigma=1.0,
        decay_tau=3000,
        guidewire_dark_weight=0.15,   # 提高权重
        guidewire_consistency_weight=0.08,  # 新增结构一致性项
    ):
        super().__init__()

        self.charbonnier = CharbonnierLoss()
        self.grad_loss = GradientLoss(loss_weight=grad_weight)
        self.frangi_loss = FrangiLoss(loss_weight=frangi_weight)
        self.brightness_loss = MeanStdLoss(loss_weight=brightness_weight)
        self.tv = WeightedTVLoss()

        self.tv_weight = tv_weight
        self.pseudo_sigma = pseudo_sigma
        self.decay_tau = decay_tau
        self.guidewire_dark_weight = guidewire_dark_weight
        self.guidewire_consistency_weight = guidewire_consistency_weight


    def detect_guidewire_old(self, noisy):
        # noisy: [B,3,H,W] → 转灰度
        if noisy.shape[1] == 3:
            img = noisy.mean(dim=1, keepdim=True).detach()
        else:
            img = noisy.detach()

        R_line = hessian_line_filter(img)
        R_ori = orientation_consistency(img)

        R = R_line * R_ori
        R = R / (R.max() + 1e-6)

        mask = (R > 0.18).float()  # 阈值略低，mask更密
        mask = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)  # 扩展导丝区域
        return mask

    def detect_guidewire(self, noisy):
        if noisy.shape[1] == 3:
            img = noisy.mean(dim=1, keepdim=True).detach()
        else:
            img = noisy.detach()

        R_line = hessian_line_filter(img, sigmas=[0.8, 1.2, 1.6])
        R_ori = orientation_consistency(img)

        R = R_line * R_ori
        R = R / (R.max() + 1e-6)

        threshold = R.mean() + 0.5 * R.std()  # 动态阈值
        mask = (R > threshold).float()
        mask = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
        return mask

    def forward(self, pred, pseudo_gt, noisy, step):
        pseudo_gt_smooth = gaussian_blur(pseudo_gt, sigma=self.pseudo_sigma)
        w = torch.exp(torch.tensor(-step / self.decay_tau, device=pred.device))

        L_main = w * self.charbonnier(pred, pseudo_gt_smooth)
        L_grad = self.grad_loss(pred, pseudo_gt_smooth)
        L_frangi = self.frangi_loss(pred, pseudo_gt_smooth)
        L_brightness = self.brightness_loss(pred, pseudo_gt_smooth)

        residual = pred - noisy
        L_tv = self.tv(residual) * self.tv_weight

        # ---------------------------
        # 导丝增强部分
        # ---------------------------
        guidewire_mask = self.detect_guidewire(noisy)

        # 局部对比度：导丝区域比背景更暗
        L_dark = ((pred * guidewire_mask).mean() - (pred * (1 - guidewire_mask)).mean()) * self.guidewire_dark_weight

        # 结构一致性：导丝区域保持与 pseudo-GT 的形态一致
        L_consistency = F.l1_loss(pred * guidewire_mask, pseudo_gt_smooth * guidewire_mask) * self.guidewire_consistency_weight

        # ---------------------------
        # 总 loss
        # ---------------------------
        loss = L_main + L_grad + L_frangi + L_brightness + L_tv + L_dark + L_consistency
        return loss

class FinalLossV3PlusPlus(nn.Module):  # Guidewire Contrast Version
    def __init__(
        self,
        grad_weight=0.25,
        tv_weight=3e-5,
        frangi_weight=1.5e-3,
        brightness_weight=0.02,
        pseudo_sigma=1.0,
        decay_tau=3000,
        guidewire_dark_weight=0.15,   # 提高权重
        guidewire_consistency_weight=0.08,  # 新增结构一致性项
        tissue_bright_weight=0.02
    ):
        super().__init__()

        self.charbonnier = CharbonnierLoss()
        self.grad_loss = GradientLoss(loss_weight=grad_weight)
        self.frangi_loss = FrangiLoss(loss_weight=frangi_weight)
        self.brightness_loss = MeanStdLoss(loss_weight=brightness_weight)
        self.tv = WeightedTVLoss()

        self.tv_weight = tv_weight
        self.pseudo_sigma = pseudo_sigma
        self.decay_tau = decay_tau
        self.guidewire_dark_weight = guidewire_dark_weight
        self.guidewire_consistency_weight = guidewire_consistency_weight
        self.tissue_bright_weight = tissue_bright_weight

    def detect_guidewire(self, noisy):
        if noisy.shape[1] == 3:
            img = noisy.mean(dim=1, keepdim=True).detach()
        else:
            img = noisy.detach()

        R_line = hessian_line_filter(img, sigmas=[0.8, 1.2, 1.6])
        R_ori = orientation_consistency(img)

        R = R_line * R_ori
        R = R / (R.max() + 1e-6)

        threshold = R.mean() + 0.5 * R.std()  # 动态阈值
        mask = (R > threshold).float()
        mask = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
        return mask

    def forward(self, pred, pseudo_gt, noisy, step):

        # 1. 平滑伪GT，避免高频噪声
        pseudo_gt_smooth = gaussian_blur(pseudo_gt, sigma=self.pseudo_sigma)

        # 2. 动态权重：训练初期更依赖伪GT，后期逐渐减弱
        w = torch.exp(torch.tensor(-step / self.decay_tau, device=pred.device))

        # 3. 基础损失
        L_main = w * self.charbonnier(pred, pseudo_gt_smooth)
        L_grad = self.grad_loss(pred, pseudo_gt_smooth)
        L_frangi = self.frangi_loss(pred, pseudo_gt_smooth)
        L_brightness = self.brightness_loss(pred, pseudo_gt_smooth)

        # 4. TV约束（残差平滑）
        residual = pred - noisy
        L_tv = self.tv(residual) * self.tv_weight

        # ---------------------------
        # 导丝增强部分
        # ---------------------------
        guidewire_mask = self.detect_guidewire(noisy)

        # 导丝区域比背景更暗
        L_dark = ((pred * guidewire_mask).mean() - (
                    pred * (1 - guidewire_mask)).mean()) * self.guidewire_dark_weight

        # 导丝区域保持与伪GT一致
        L_consistency = F.l1_loss(pred * guidewire_mask,
                                  pseudo_gt_smooth * guidewire_mask) * self.guidewire_consistency_weight

        # ---------------------------
        # 软组织亮度抑制部分
        # ---------------------------
        # Frangi 响应 → 血管/骨骼掩膜
        vessel_response = self.frangi_loss.frangi_response(noisy)
        vessel_response = vessel_response / (vessel_response.max() + 1e-6)
        vessel_mask = (vessel_response > 0.2).float()
        vessel_mask = F.interpolate(vessel_mask, size=pred.shape[-2:], mode='nearest')

        # 软组织掩膜 = 全图 - 导丝 - 血管/骨骼
        tissue_mask = 1 - torch.clamp(guidewire_mask + vessel_mask, 0, 1)

        # 亮度约束：软组织亮度接近全局中位数
        mean_tissue = (pred * tissue_mask).mean()
        median_global = pred.median()
        L_tissue_balance = torch.abs(mean_tissue - median_global) * self.tissue_bright_weight

        # ---------------------------
        # 总 loss
        # ---------------------------
        loss = (
                L_main + L_grad + L_frangi + L_brightness + L_tv +
                L_dark + L_consistency + L_tissue_balance
        )
        return loss


class FinalLossV4(nn.Module):
    def __init__(
        self,
        grad_weight=0.2,
        tv_weight=3e-5,
        frangi_weight=1e-3,
        brightness_weight=0.01,
        pseudo_sigma=1.0,
        decay_tau=3000,
        guidewire_dark_weight=0.10,   # 导丝增强更强
        tissue_bright_weight=0.05,    # 组织抑制（变亮）
        bw_contrast_weight=0.05       # 黑白反差增强
    ):
        super().__init__()

        self.charbonnier = CharbonnierLoss()
        self.grad_loss = GradientLoss(loss_weight=grad_weight)
        self.frangi_loss = FrangiLoss(loss_weight=frangi_weight)
        self.brightness_loss = MeanStdLoss(loss_weight=brightness_weight)
        self.tv = WeightedTVLoss()

        self.tv_weight = tv_weight
        self.pseudo_sigma = pseudo_sigma
        self.decay_tau = decay_tau

        self.guidewire_dark_weight = guidewire_dark_weight
        self.tissue_bright_weight = tissue_bright_weight
        self.bw_contrast_weight = bw_contrast_weight


    def detect_guidewire(self, noisy):
        if noisy.shape[1] == 3:
            img = noisy.mean(dim=1, keepdim=True).detach()
        else:
            img = noisy.detach()

        R_line = hessian_line_filter(img)
        R_ori = orientation_consistency(img)

        R = R_line * R_ori
        R = R / (R.max() + 1e-6)

        mask = (R > 0.25).float()
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        return mask


    def forward(self, pred, pseudo_gt, noisy, step):
        pseudo_gt_smooth = gaussian_blur(pseudo_gt, sigma=self.pseudo_sigma)
        w = torch.exp(torch.tensor(-step / self.decay_tau, device=pred.device))

        # 基础项
        L_main = w * self.charbonnier(pred, pseudo_gt_smooth)
        L_grad = self.grad_loss(pred, pseudo_gt_smooth)
        L_frangi = self.frangi_loss(pred, pseudo_gt_smooth)
        L_brightness = self.brightness_loss(pred, pseudo_gt_smooth)

        residual = pred - noisy
        L_tv = self.tv(residual) * self.tv_weight

        # ---------------------------
        # 导丝检测
        # ---------------------------
        guidewire_mask = self.detect_guidewire(noisy)

        # ---------------------------
        # 1) 黑色结构增强（导丝/血管/植入物）
        # ---------------------------
        L_dark = (pred * guidewire_mask).mean() * self.guidewire_dark_weight

        # ---------------------------
        # 2) 组织抑制（让软组织变亮）
        # ---------------------------
        tissue_mask = (1 - guidewire_mask)
        L_tissue_bright = -(pred * tissue_mask).mean() * self.tissue_bright_weight

        # ---------------------------
        # 3) 黑白反差增强（X-ray 特有）
        # ---------------------------
        pred_min = pred.min()   # 黑区（导丝/血管）
        pred_max = pred.max()   # 亮区（组织）
        L_bw_contrast = -(pred_max - pred_min) * self.bw_contrast_weight

        # ---------------------------
        # 总 loss
        # ---------------------------
        loss = (
            L_main + L_grad + L_frangi + L_brightness + L_tv +
            L_dark + L_tissue_bright + L_bw_contrast
        )

        return loss
