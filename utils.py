import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between two image tensors [0, 1].
    Shape: (B, C, H, W) or (C, H, W)
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100.0
    return 20.0 * np.log10(max_val / np.sqrt(mse))

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two single-channel or RGB tensors.
    """
    # Simple PyTorch-native approximation for SSIM
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    mu1 = torch.nn.functional.avg_pool2d(img1, kernel_size=window_size, stride=1, padding=window_size // 2)
    mu2 = torch.nn.functional.avg_pool2d(img2, kernel_size=window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.avg_pool2d(img1 * img1, kernel_size=window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(img2 * img2, kernel_size=window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(img1 * img2, kernel_size=window_size, stride=1, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()

def pad_image_for_model(img: torch.Tensor, divisor: int = 16):
    """
    Pads an image tensor (1, C, H, W) with reflection padding so H and W are multiples of `divisor`.
    Returns padded tensor, (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)
    """
    _, _, h, w = img.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h > 0 or pad_w > 0:
        padded = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
    else:
        padded = img

    return padded, (pad_top, pad_bottom, pad_left, pad_right)

def unpad_image(img: torch.Tensor, pad_info: tuple):
    """
    Crops away the padding added by pad_image_for_model.
    """
    pad_top, pad_bottom, pad_left, pad_right = pad_info
    h, w = img.shape[2], img.shape[3]
    h_end = h - pad_bottom if pad_bottom > 0 else h
    w_end = w - pad_right if pad_right > 0 else w
    return img[:, :, pad_top:h_end, pad_left:w_end]

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Converts a (C, H, W) or (1, C, H, W) tensor with values in [0, 1] to a PIL Image.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = torch.clamp(tensor, 0.0, 1.0).detach().cpu()
    np_img = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(np_img)

def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """
    Converts a PIL Image to a (1, C, H, W) float32 tensor in range [0, 1].
    """
    np_img = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)
    return tensor
