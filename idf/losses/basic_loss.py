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
        residual = pred - noisy
        L_tv = self.tv(residual) * self.tv_weight

        loss = L_main + L_grad + L_frangi + L_brightness + L_tv
        return loss
