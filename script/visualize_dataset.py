"""数据集可视化脚本

可视化内容：
1. 原始图片（未经增强）
2. 增强后的图片
3. 对应的mask标签（着色）
4. 图片与mask的叠加混合

用法：
    python script/visualize_dataset.py --cfg train_cfg/CamVid/LeNet_cfg.yaml
    python script/visualize_dataset.py --cfg train_cfg/CamVid/LeNet_cfg.yaml --num 8 --split val
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml

# 设置中文字体（Windows系统）
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from dataset.CamVid_dataset import CamVid_11_Dataset
from dataset.augment import get_train_transform, get_val_transform


# CamVid 11类调色板（与类别ID对应）
PALETTE = {
    0:  (128, 128, 128),   # Sky
    1:  (128, 0, 0),       # Building
    2:  (192, 192, 128),   # Pole
    3:  (128, 64, 128),    # Road
    4:  (0, 0, 192),       # Sidewalk
    5:  (128, 128, 0),     # Tree
    6:  (192, 128, 128),   # SignSymbol
    7:  (64, 64, 128),     # Fence
    8:  (64, 0, 128),      # Car
    9:  (64, 64, 0),       # Pedestrian
    10: (0, 128, 192),     # Bicyclist
    255: (0, 0, 0),        # ignore
}

CLASS_NAMES = {
    0: "Sky", 1: "Building", 2: "Pole", 3: "Road", 4: "Sidewalk",
    5: "Tree", 6: "SignSymbol", 7: "Fence", 8: "Car", 9: "Pedestrian", 10: "Bicyclist",
}


def mask_to_color(mask):
    """将整数mask转为RGB彩色图"""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in PALETTE.items():
        color[mask == cls_id] = rgb
    return color


def denormalize(image_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """反归一化tensor图片为numpy [H, W, C] uint8"""
    img = image_tensor.clone().cpu().float()
    for c, (m, s) in enumerate(zip(mean, std)):
        img[c] = img[c] * s + m
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def overlay(image, color_mask, alpha=0.5):
    """将彩色mask叠加到图片上"""
    return cv2.addWeighted(image, alpha, color_mask, 1 - alpha, 0)


def load_cfg(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def visualize(cfg_path, num_samples=4, split="train", save_path=None):
    cfg = load_cfg(cfg_path)
    data_root = cfg["data_root"]
    image_size = tuple(cfg["image_size"])
    json_file = os.path.join(data_root, "camvid_11class_mapping.json")
    if not os.path.exists(json_file):
        json_file = None

    img_dir = os.path.join(data_root, split)
    mask_dir = os.path.join(data_root, f"{split}_labels")

    # 原始数据集（无增强）
    raw_dataset = CamVid_11_Dataset(img_dir, mask_dir, json_file, transform=None)
    # 增强后数据集
    train_transform = get_train_transform(image_size)
    aug_dataset = CamVid_11_Dataset(img_dir, mask_dir, json_file, transform=train_transform)

    num_samples = min(num_samples, len(raw_dataset))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["原始图片", "增强后图片", "Mask标签", "叠加混合"]

    for i in range(num_samples):
        # 原始数据（__getitem__返回的是tensor [C,H,W] float [0,1]）
        raw_img, raw_mask = raw_dataset[i]
        if isinstance(raw_img, torch.Tensor):
            raw_img_np = (raw_img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        else:
            raw_img_np = raw_img
        raw_mask_np = raw_mask.numpy() if isinstance(raw_mask, torch.Tensor) else raw_mask
        raw_color_mask = mask_to_color(raw_mask_np)

        # 增强后数据
        aug_img, aug_mask = aug_dataset[i]
        aug_img_np = denormalize(aug_img)
        aug_mask_np = aug_mask.numpy() if isinstance(aug_mask, torch.Tensor) else aug_mask
        aug_color_mask = mask_to_color(aug_mask_np)

        # 原始图的overlay
        raw_overlay = overlay(raw_img_np, raw_color_mask)

        # 绘制
        axes[i, 0].imshow(raw_img_np)
        axes[i, 0].set_title(col_titles[0] if i == 0 else "", fontsize=12)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(aug_img_np)
        axes[i, 1].set_title(col_titles[1] if i == 0 else "", fontsize=12)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(aug_color_mask)
        axes[i, 2].set_title(col_titles[2] if i == 0 else "", fontsize=12)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(raw_overlay)
        axes[i, 3].set_title(col_titles[3] if i == 0 else "", fontsize=12)
        axes[i, 3].axis("off")

    # 添加图例
    patches = [mpatches.Patch(color=np.array(PALETTE[c]) / 255.0, label=CLASS_NAMES[c])
               for c in sorted(CLASS_NAMES.keys()) if c != 255]
    fig.legend(handles=patches, loc="lower center", ncol=6, fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集可视化")
    parser.add_argument("--cfg", type=str, default="train_cfg/CamVid/LeNet_cfg.yaml",
                        help="yaml配置文件路径")
    parser.add_argument("--num", type=int, default=4, help="可视化样本数")
    parser.add_argument("--split", type=str, default="train", help="数据集划分: train/val/test")
    parser.add_argument("--save", type=str, default=None, help="保存路径（不填则直接显示）")
    args = parser.parse_args()

    visualize(args.cfg, num_samples=args.num, split=args.split, save_path=args.save)
