#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.fft


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def prepare_for_ssim(img1, img2, clamp=True):
    if img1.dim() == 2:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 2:
        img2 = img2.unsqueeze(0)

    if clamp:
        img1 = torch.clamp(img1, 0.0, 1.0)
        img2 = torch.clamp(img2, 0.0, 1.0)

    return img1, img2


def transform_for_ssim(img, domain="linear", eps=1e-6, clamp=True):
    if img.dim() == 2:
        img = img.unsqueeze(0)

    if clamp:
        img = torch.clamp(img, 0.0, 1.0)

    domain = str(domain).lower()
    if domain == "linear":
        return img
    if domain == "log":
        return torch.log(img + eps)

    raise ValueError(f"Unsupported ssim domain: {domain}")


def prepare_ssim_pair(img1, img2, domain="linear", eps=1e-6, clamp=True):
    img1 = transform_for_ssim(img1, domain=domain, eps=eps, clamp=clamp)
    img2 = transform_for_ssim(img2, domain=domain, eps=eps, clamp=clamp)
    return img1, img2


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def compute_ssim(
    img1,
    img2,
    window_size=11,
    size_average=True,
    clamp=True,
    domain="linear",
    eps=1e-6,
):
    img1, img2 = prepare_ssim_pair(img1, img2, domain=domain, eps=eps, clamp=clamp)
    return ssim(img1, img2, window_size=window_size, size_average=size_average)


def compute_ssim_loss(
    img1,
    img2,
    window_size=11,
    size_average=True,
    clamp=True,
    domain="linear",
    eps=1e-6,
):
    return 1.0 - compute_ssim(
        img1,
        img2,
        window_size=window_size,
        size_average=size_average,
        clamp=clamp,
        domain=domain,
        eps=eps,
    )


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# def fourier_loss(pred, gt):
#     """
#     Compute Fourier-domain L2 loss between pred and gt.
#     pred, gt: Tensors of shape [H, W] or [B, H, W]
#     """
#     # Add batch dim if missing
#     if pred.dim() == 2:
#         pred = pred.unsqueeze(0)
#         gt = gt.unsqueeze(0)

#     # Apply 2D FFT
#     pred_fft = torch.fft.fft2(pred)
#     gt_fft = torch.fft.fft2(gt)

#     # Compute L2 difference in frequency domain
#     diff = pred_fft - gt_fft
#     loss = torch.mean(torch.abs(diff) ** 2)

#     return loss


def fourier_loss(pred, gt):
    pred_fft = torch.fft.fft2(pred)
    gt_fft = torch.fft.fft2(gt)

    diff = torch.abs(pred_fft - gt_fft) ** 2
    total_energy = torch.abs(gt_fft) ** 2

    return torch.mean(diff / (total_energy + 1e-8))
