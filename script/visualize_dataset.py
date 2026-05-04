"""数据集可视化脚本

可视化内容：
1. 原始图片（未经增强）
2. 增强后的图片
3. 对应的mask标签（着色）
4. 图片与mask的叠加混合

用法：
    python script/visualize_dataset.py --cfg train_cfg/CamVid/LeNet_cfg.yaml
    python script/visualize_dataset.py --cfg train_cfg/STS_2D/UNet_cfg.yaml --num 8 --split val
    python script/visualize_dataset.py --cfg train_cfg/CamVid/LeNet_cfg.yaml --split val --save custom/path.png

默认保存路径：script/visual_dataset/{dataset_name}/{yaml文件名}_{split}.png
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
from dataset.STS_2d_dataset import STS_2D_Dataset
from dataset.build_dataset import DIR_BUILDERS
from dataset.augment import get_train_transform, get_val_transform


# ---- 数据集类别名称 ----
CLASS_NAMES_MAP = {
    "CamVid_11": {
        0: "Sky", 1: "Building", 2: "Pole", 3: "Road", 4: "Sidewalk",
        5: "Tree", 6: "SignSymbol", 7: "Fence", 8: "Car", 9: "Pedestrian", 10: "Bicyclist",
    },
    "STS_2D": {
        0: "背景", 1: "前景",
    },
}


def build_palette(rgb_values):
    """从 dataset 实例的 rgb_values 构建调色板字典。"""
    palette = {i: rgb for i, rgb in enumerate(rgb_values)}
    palette[255] = (0, 0, 0)  # ignore
    return palette


def mask_to_color(mask, palette):
    """将整数mask转为RGB彩色图"""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in palette.items():
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


def create_dataset(dataset_name, img_dir, mask_dir, cfg, transform):
    """根据数据集名称创建对应的 Dataset 实例（无增强）。"""
    if dataset_name == "CamVid_11":
        json_file = os.path.join(cfg["data_root"], "camvid_11class_mapping.json")
        if not os.path.exists(json_file):
            json_file = None
        return CamVid_11_Dataset(img_dir, mask_dir, json_file, transform=transform)
    elif dataset_name == "STS_2D":
        return STS_2D_Dataset(
            img_dir, mask_dir, transform=transform,
            num_classes=cfg["num_classes"],
            ignore_index=cfg.get("ignore_index", 255),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def visualize(cfg_path, num_samples=4, split="train", save_path=None):
    cfg = load_cfg(cfg_path)
    dataset_name = cfg.get("dataset_name", "CamVid_11")
    image_size = tuple(cfg["image_size"])

    # 根据数据集类型选择目录布局
    dir_builder = DIR_BUILDERS[dataset_name]
    img_dir, mask_dir = dir_builder(cfg["data_root"], split)

    # 原始数据集（无增强）
    raw_dataset = create_dataset(dataset_name, img_dir, mask_dir, cfg, transform=None)
    # 增强后数据集
    train_transform = get_train_transform(image_size)
    aug_dataset = create_dataset(dataset_name, img_dir, mask_dir, cfg, transform=train_transform)

    # 从 dataset 实例获取调色板和类别名称
    palette = build_palette(raw_dataset.rgb_values)
    class_names = CLASS_NAMES_MAP.get(dataset_name, {})

    num_samples = min(num_samples, len(raw_dataset))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["原始图片", "增强后图片", "Mask标签", "叠加混合"]

    for i in range(num_samples):
        # 原始数据
        raw_img, raw_mask = raw_dataset[i]
        if isinstance(raw_img, torch.Tensor):
            raw_img_np = (raw_img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        else:
            raw_img_np = raw_img
        raw_mask_np = raw_mask.numpy() if isinstance(raw_mask, torch.Tensor) else raw_mask
        raw_color_mask = mask_to_color(raw_mask_np, palette)

        # 增强后数据
        aug_img, aug_mask = aug_dataset[i]
        aug_img_np = denormalize(aug_img)
        aug_mask_np = aug_mask.numpy() if isinstance(aug_mask, torch.Tensor) else aug_mask
        aug_color_mask = mask_to_color(aug_mask_np, palette)

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
    patches = [mpatches.Patch(color=np.array(palette[c]) / 255.0, label=class_names.get(c, str(c)))
               for c in sorted(class_names.keys()) if c != 255]
    fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 6), fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def _auto_save_path(cfg_path, split):
    """根据 yaml 文件名和数据集名称自动生成保存路径。

    格式: script/visual_dataset/{dataset_name}/{yaml文件名}_{split}.png
    """
    cfg = load_cfg(cfg_path)
    dataset_name = cfg.get("dataset_name", "CamVid_11")
    yaml_stem = os.path.splitext(os.path.basename(cfg_path))[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "visual_dataset", dataset_name, f"{yaml_stem}_{split}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集可视化")
    parser.add_argument("--cfg", type=str, default="train_cfg/CamVid/LeNet_cfg.yaml",
                        help="yaml配置文件路径")
    parser.add_argument("--num", type=int, default=4, help="可视化样本数")
    parser.add_argument("--split", type=str, default="train", help="数据集划分: train/val/test")
    parser.add_argument("--save", type=str, default=None, help="保存路径（不填则自动生成）")
    parser.add_argument("--show", action="store_true", help="直接显示而不保存")
    args = parser.parse_args()

    if args.show:
        save_path = None
    else:
        save_path = args.save or _auto_save_path(args.cfg, args.split)

    visualize(args.cfg, num_samples=args.num, split=args.split, save_path=save_path)
