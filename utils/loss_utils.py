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

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
import numpy as np
from scipy.spatial import cKDTree

def chamfer_distance(p1, p2):
    """
    p1, p2: (N, D) and (M, D) numpy arrays
    Returns: scalar chamfer distance
    """
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)

    # p1 -> p2
    dist1, _ = tree2.query(p1)
    # p2 -> p1
    dist2, _ = tree1.query(p2)

    return np.mean(dist1**2) + np.mean(dist2**2)
    
def line_to_line_distance(p1, d1, p2, d2, eps=1e-8):

    p1 = np.array(p1, dtype=float)
    d1 = np.array(d1, dtype=float)
    p2 = np.array(p2, dtype=float)
    d2 = np.array(d2, dtype=float)

    cross = np.cross(d1, d2)
    denom = np.linalg.norm(cross)

    if denom < eps:  
        diff = p2 - p1
        proj = diff - np.dot(diff, d1) / np.dot(d1, d1) * d1
        return np.linalg.norm(proj)

    num = np.abs(np.dot((p2 - p1), cross))
    return num / denom

def angle_between_vectors(v1, v2, in_degrees=True, eps=1e-8):

    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    dot = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

    if norm_product < eps:
        return np.inf
        # raise ValueError("Zero-length vector provided.")

    cos_theta = np.clip(dot / norm_product, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)

    return np.degrees(angle_rad) if in_degrees else angle_rad
