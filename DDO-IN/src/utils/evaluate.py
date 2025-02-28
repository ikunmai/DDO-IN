"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import csv
import os

import __main__
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from piq import haarpsi, ssim
import cv2
import lpips
import piq

import torch
import torch.nn.functional as F
from torchvision import transforms
import math
from scipy.ndimage import gaussian_filter


def compute_fsim(img1, img2, sigma=1.0):
    """
    计算两个图像的FSIM (Feature Similarity Index)
    :param img1: 第一个图像（B, C, H, W）
    :param img2: 第二个图像（B, C, H, W）
    :param sigma: 高斯滤波的标准差
    :return: FSIM的值
    """
    device = img1.device  # 获取输入张量所在的设备

    # Sobel 卷积核用于计算图像的梯度
    sobel_kernel = torch.tensor([[[[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]]]], dtype=torch.float32, device=device)

    # 为了保证梯度计算，卷积核应该应用在 img1 和 img2 上
    grad_x1 = F.conv2d(img1, sobel_kernel, padding=1)
    grad_y1 = F.conv2d(img1, sobel_kernel.transpose(2, 3), padding=1)
    grad_x2 = F.conv2d(img2, sobel_kernel, padding=1)
    grad_y2 = F.conv2d(img2, sobel_kernel.transpose(2, 3), padding=1)

    # 计算梯度幅值
    grad_magnitude1 = torch.sqrt(grad_x1 ** 2 + grad_y1 ** 2)
    grad_magnitude2 = torch.sqrt(grad_x2 ** 2 + grad_y2 ** 2)

    # 计算梯度相似性
    grad_similarity = torch.exp(-torch.mean((grad_magnitude1 - grad_magnitude2) ** 2, dim=(1, 2, 3)) / (2 * sigma ** 2))

    # 计算相似性
    # 使用高斯滤波进行平滑处理，构造高斯滤波核
    kernel_size = 5  # 高斯核的大小
    gaussian_kernel = torch.exp(
        -torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size).pow(2) / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()  # 归一化

    # 将高斯核扩展为2D卷积核
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, 1) * gaussian_kernel.view(1, 1, 1, kernel_size)
    gaussian_kernel = gaussian_kernel.to(device)

    # 对两个图像应用高斯滤波
    img1_smoothed = F.conv2d(img1, gaussian_kernel, padding=kernel_size // 2)
    img2_smoothed = F.conv2d(img2, gaussian_kernel, padding=kernel_size // 2)

    # 计算图像相似性（对比度）
    contrast_similarity = torch.exp(-torch.mean((img1_smoothed - img2_smoothed) ** 2, dim=(1, 2, 3)) / (2 * sigma ** 2))

    # 计算颜色相似性（亮度）
    img1_mean = img1.mean(dim=(1, 2, 3), keepdim=True)
    img2_mean = img2.mean(dim=(1, 2, 3), keepdim=True)
    color_similarity = torch.exp(-torch.mean((img1_mean - img2_mean) ** 2, dim=(1, 2, 3)) / (2 * sigma ** 2))

    # 综合梯度相似性、对比度相似性和颜色相似性
    fsim = grad_similarity * contrast_similarity * color_similarity

    # 计算最终的FSIM
    fsim_score = fsim.mean()

    return fsim_score


def lpips_score(img1, img2, model='alex'):
    """
    计算L_pips感知相似性度量
    :param img1: 原始图像（PIL或Tensor）
    :param img2: 重建图像（PIL或Tensor）
    :param model: 使用的模型（可选: 'alex'，'vgg'，'squeeze'等）
    :return: LPIPS值
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 图像归一化
    ])
    
    # 转换为tensor
    img1 = transform(img1).unsqueeze(0)
    img2 = transform(img2).unsqueeze(0)
    
    # 加载LPIPS模型
    lpips_model = lpips.LPIPS(net=model)
    
    # 计算LPIPS
    return lpips_model(img1, img2).item()


def vif_score(img1, img2):
    """
    计算VIF（视觉信息保真度，Visual Information Fidelity）
    :param img1: 原始图像
    :param img2: 重建图像
    :return: VIF值
    """
    # 将图像转换为灰度图像
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 对图像进行高斯模糊处理，以模拟视觉系统对不同频率的响应
    img1_blurred = gaussian_filter(img1_gray, sigma=1)
    img2_blurred = gaussian_filter(img2_gray, sigma=1)

    # 计算两幅图像的均方误差（MSE）
    mse = np.mean((img1_blurred - img2_blurred) ** 2)

    # 计算图像的局部标准差，模拟视觉系统的感知能力
    std1 = np.std(img1_blurred)
    std2 = np.std(img2_blurred)

    # 计算相对信噪比（SNR）
    snr = np.mean(img1_blurred) / np.mean(img2_blurred)

    # 计算VIF值（简化公式）
    vif = (mse / (std1 + std2)) * snr

    return vif


def calculate_vif(image1, image2, data_range=1.0, device='cuda'):
    """
    计算两张单通道图像的VIF值。

    参数：
        image1 (torch.Tensor): 第一张图像，形状为 [1, 1, 160, 160]。
        image2 (torch.Tensor): 第二张图像，形状为 [1, 1, 160, 160]。
        data_range (float): 图像数据的动态范围，默认为 1.0。
        device (str): 计算设备，默认为 'cuda'。

    返回：
        float: VIF值。
    """

    # 将图像移动到指定的设备
    img1 = image1.to(device)
    img2 = image2.to(device)

    # 计算VIF值
    vif_value = piq.vif_p(img1, img2, data_range=data_range)

    return vif_value.item()
    

import torch
import torch.nn.functional as F

def vsi_score(img1, img2, sigma=1.0):
    """
    计算VSI（视觉相似性指数）
    :param img1: 第一个图像（B, C, H, W）
    :param img2: 第二个图像（B, C, H, W）
    :param sigma: 高斯滤波的标准差
    :return: VSI的值
    """
    device = img1.device  # 获取输入张量所在的设备

    # Sobel 卷积核用于计算图像的梯度
    sobel_kernel = torch.tensor([[[[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]]]], dtype=torch.float32, device=device)

    # 为了保证梯度计算，卷积核应该应用在 img1 和 img2 上
    grad_x1 = F.conv2d(img1, sobel_kernel, padding=1)
    grad_y1 = F.conv2d(img1, sobel_kernel.transpose(2, 3), padding=1)
    grad_x2 = F.conv2d(img2, sobel_kernel, padding=1)
    grad_y2 = F.conv2d(img2, sobel_kernel.transpose(2, 3), padding=1)

    # 计算梯度幅值
    grad_magnitude1 = torch.sqrt(grad_x1 ** 2 + grad_y1 ** 2)
    grad_magnitude2 = torch.sqrt(grad_x2 ** 2 + grad_y2 ** 2)

    # 计算梯度相似性
    grad_similarity = torch.exp(-torch.mean((grad_magnitude1 - grad_magnitude2) ** 2, dim=(1, 2, 3)) / (2 * sigma ** 2))

    # 计算最终的VSI
    vsi_value = grad_similarity.mean()

    return vsi_value


def gmsd_score(img1, img2, sigma=1.0):
    """
    计算GMSD（梯度相似性指标）
    :param img1: 第一个图像（B, C, H, W）
    :param img2: 第二个图像（B, C, H, W）
    :param sigma: 高斯滤波的标准差
    :return: GMSD值
    """
    device = img1.device  # 获取输入张量所在的设备

    # Sobel 卷积核用于计算图像的梯度
    sobel_kernel = torch.tensor([[[[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]]]], dtype=torch.float32, device=device)

    # 为了保证梯度计算，卷积核应该应用在 img1 和 img2 上
    grad_x1 = F.conv2d(img1, sobel_kernel, padding=1)
    grad_y1 = F.conv2d(img1, sobel_kernel.transpose(2, 3), padding=1)
    grad_x2 = F.conv2d(img2, sobel_kernel, padding=1)
    grad_y2 = F.conv2d(img2, sobel_kernel.transpose(2, 3), padding=1)

    # 计算梯度幅值
    grad_magnitude1 = torch.sqrt(grad_x1 ** 2 + grad_y1 ** 2)
    grad_magnitude2 = torch.sqrt(grad_x2 ** 2 + grad_y2 ** 2)

    # 计算梯度差异
    diff = grad_magnitude1 - grad_magnitude2
    gmsd = torch.sqrt(torch.mean(diff ** 2, dim=(1, 2, 3)))

    return gmsd.mean()



def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize array to [0, 1] range"""
    if x.shape[0] > 1:
        # batchwise normalization
        max_b = x.view(x.shape[0], -1).max(1).values
        min_b = x.view(x.shape[0], -1).min(1).values
        return (x - min_b.view(-1, 1, 1, 1)) / (
            (max_b - min_b).view(-1, 1, 1, 1) + 1e-24
        )
    else:
        return (x - x.min()) / (x.max() - x.min() + 1e-24)


def rmse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute RMSE between two arrays"""
    return torch.sqrt(torch.mean((x - y) ** 2))


def my_psnr(img1, img2, data_range=None, reduction="mean"):
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))
    if data_range is None:
        max_pixel = img2.view(img2.shape[0], -1).max(1).values
    else:
        max_pixel = data_range

    if reduction == "none":
        return 20 * torch.log10(max_pixel / torch.sqrt(mse))
    else:
        return (20 * torch.log10(max_pixel / torch.sqrt(mse))).mean()


# def sharpnes_metric(x: torch.Tensor) -> torch.Tensor:
#     """Compute sharpness metric between two arrays"""
#     import cpbd

#     return cpbd.compute(x.squeeze().numpy() * 255.0, mode="sharpness")


def calmetric2D(pred_recon, gt_recon):
    # check sizes -> (B, C, H, W )
    if not pred_recon.ndim == 4 or not gt_recon.ndim == 4:
        raise ValueError("Input tensors must be 4D")

    # normalize
    pred = normalize(pred_recon)
    gt = normalize(gt_recon)

    ssim_kernel = 11
    haar_scale = 3
    if pred.shape[-1] < ssim_kernel or pred.shape[-2] < ssim_kernel:
        ssim_kernel = min(pred.shape[-1], pred.shape[-2], ssim_kernel) - 1
        haar_kernel = min(pred.shape[-1], pred.shape[-2], haar_kernel) - 1
        haar_scale = int(np.log2(haar_kernel))

    psnr_array = my_psnr(pred, gt, data_range=1.0, reduction="mean")
    ssim_array = ssim(
        pred, gt, data_range=1.0, kernel_size=ssim_kernel, reduction="mean"
    )
    haar_psi_array = haarpsi(pred, gt, scales=haar_scale, reduction="mean")
    rmse_array = rmse(pred, gt)

    # print(pred.shape)
    # print(gt.shape)
    # print(type(pred))
    # print(type(gt))

    # 
    # lpips = lpips_score(pred, gt)
    vif = calculate_vif(pred, gt)
    vsi = vsi_score(pred, gt)
    gmsd = gmsd_score(pred, gt)
    
    print('lpips', lpips)
    print('vif', vif)
    print('vsi', vsi)
    print('gmsd', gmsd)
    print('psnr_array', psnr_array)
    print('ssim_array', ssim_array)
    print('haar_psi_array', haar_psi_array)
    print('rmse_array', rmse_array)

    # Add FSIM
    fsim_array = compute_fsim(pred, gt)
    print('fsim_array', fsim_array)
    # print('fsim_array', fsim_array)

    return psnr_array, ssim_array, haar_psi_array, rmse_array, fsim_array, vif, vsi, gmsd



def calmetric3D(pred_recon, gt_recon):

    batch = pred_recon.shape[0]

    ssim_array = np.zeros(batch)
    psnr_array = np.zeros(batch)
    haar_psi = np.zeros(batch)
    rmse_array = np.zeros(batch)

    for i in range(batch):
        psnr_array[i], ssim_array[i], haar_psi[i], rmse_array[i] = calmetric2D(
            pred_recon[i].unsqueeze(0), gt_recon[i].unsqueeze(0)
        )

    return psnr_array.mean(), ssim_array.mean(), haar_psi.mean(), rmse_array.mean()


def calc_metrics(pred_recon, gt_recon, save_path, name):
    # check sizes -> (B, C, D, H, W )
    if not pred_recon.ndim == 5 or not gt_recon.ndim == 5:
        batch = pred_recon.shape[0]
        if batch == 1:
            psnr_array, ssim_array, haar_psi, rmse_array, fsim_array, vif_array, vsi_array, gmsd_array = calmetric2D(
                pred_recon, gt_recon
            )
            print(
                f"PSNR: {psnr_array} db, SSIM: {ssim_array*100}%, HaarPSI: {haar_psi}, RMSE: {rmse_array}, FSIM: {fsim_array}, VIF: {vif_array}, VSI: {vsi_array}, GMAD: {gmsd_array}"
            )
            with open(os.path.join(save_path, name + ".csv"), "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["PSNR", "SSIM", "HaarPSI", "RMSE", "FSIM", "VIF", "VSI", "GMSD"])
                writer.writerow([psnr_array, ssim_array, haar_psi, rmse_array, fsim_array, vif_array, vsi_array, gmsd_array])
        else:
            psnr_array, ssim_array, haar_psi, rmse_array, fsim_array = calmetric3D(
                pred_recon, gt_recon
            )
            print(
                f"PSNR: {psnr_array} db, SSIM: {ssim_array*100}%, HaarPSI: {haar_psi}, RMSE: {rmse_array}, FSIM: {fsim_array}, VIF: {vif_array}, VSI: {vsi_array}, GMAD: {gmsd_array}"
            )
            with open(os.path.join(save_path, name + ".csv"), "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["PSNR", "SSIM", "HaarPSI", "RMSE", "FSIM", "VIF", "VSI", "GMSD"])
                writer.writerow([psnr_array, ssim_array, haar_psi, rmse_array, fsim_array, vif_array, vsi_array, gmsd_array])
        # save as latex table
        df = pd.read_csv(os.path.join(save_path, name + ".csv"))
        df = df.round(3)
        df.to_latex(os.path.join(save_path, name + ".tex"), index=False, escape=False)
        os.remove(os.path.join(save_path, name + ".csv"))
        print("Saved metrics as csv and latex table in results folder.")



# write a function that creates a violin plot for the metrics of the whole dataset
def create_violin_plot(
    data, method_names, metric_name="SSIM", save_path="./", name="violin_plot"
):
    # chech if data is numpy array and length of method_names is correct
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array")
    if not len(method_names) == data.shape[1]:
        raise ValueError(
            "length of method_names must be equal to the number of methods in data"
        )

    # get the metrics as numpy array and create a dataframe
    df = pd.DataFrame(data, columns=method_names)
    df = df.round(3)
    sns.set_style("darkgrid")
    # create a color pallete suited for scientific plots
    my_palette = sns.color_palette("colorblind", 4)

    sns.set_palette(my_palette)
    violin_plot = sns.violinplot(data=df)
    violin_plot.set_title(metric_name)
    violin_plot.set_ylabel(metric_name)
    violin_plot.set_xlabel("Methods")
    violin_plot.get_figure().savefig(os.path.join(save_path, name + ".png"))

    print("Saved violin plot in results folder.")


def iou_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(torch.abs(y_true * y_pred))
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    iou = torch.mean((intersection + smooth) / (union + smooth), dim=0)
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = torch.mean((2.0 * intersection + smooth) / (union + smooth), dim=0)
    return dice


def sensitivity_score(true_positives: torch.Tensor, y_gt: torch.Tensor):
    sensitivity = true_positives / (y_gt == 1).sum()
    return sensitivity


def presission_score(true_positives: torch.Tensor, false_positives: torch.Tensor):
    presission = true_positives / (true_positives + false_positives)
    return presission


def specificity_score(true_negatives: torch.Tensor, y_gt: torch.Tensor):
    specificity = true_negatives / (y_gt == 0).sum()
    return specificity


def f1_score(presission: torch.Tensor, sensitivity: torch.Tensor):
    f1 = 2 * (presission * sensitivity) / (presission + sensitivity)
    return f1


def metrics_classification(y_pred: torch.Tensor, y_gt: torch.Tensor):
    # calculate scores for statistics
    true_positives = torch.sum(torch.logical_and((y_pred == 1), (y_gt == 1)))
    false_positives = torch.sum(torch.logical_and((y_pred == 1), (y_gt == 0)))
    true_negative = torch.sum(torch.logical_and((y_pred == 0), (y_gt == 0)))
    metrics_dict = {}
    metrics_dict["Sensistivity"] = sensitivity_score(true_positives, y_gt).item()
    metrics_dict["Specificity"] = specificity_score(true_negative, y_gt).item()
    metrics_dict["Presission"] = presission_score(
        true_positives, false_positives
    ).item()
    metrics_dict["F1"] = f1_score(
        presission_score(true_positives, false_positives),
        sensitivity_score(true_positives, y_gt),
    ).item()

    return metrics_dict


if __name__ == "__main__":

    data = np.random.normal(size=(100, 2), loc=0, scale=100)

    method_names = ["method1", "method2"]

    create_violin_plot(data, method_names, save_path="./", name="test")
