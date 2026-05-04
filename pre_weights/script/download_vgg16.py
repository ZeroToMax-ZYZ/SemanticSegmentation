"""下载 UNet 编码器（VGG16）的 ImageNet 预训练权重。

从 torchvision 下载 VGG16 预训练模型，提取 features 层保存为 vgg16_encoder.pth。
用法：python pre_weights/script/download_vgg16.py
"""

import os
import sys
import torch
import torchvision


# 保存路径：pre_weights/vgg16_encoder.pth
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_SAVE_PATH = os.path.join(_PROJECT_ROOT, "pre_weights", "vgg16_encoder.pth")


def download_vgg16_encoder(save_path=None):
    """下载 VGG16 ImageNet 预训练权重，仅保留 features 层。

    Args:
        save_path: 保存路径，默认为 pre_weights/vgg16_encoder.pth

    Returns:
        保存路径
    """
    save_path = save_path or _SAVE_PATH

    if os.path.exists(save_path):
        print(f"[信息] 预训练权重已存在: {save_path}")
        print("       如需重新下载，请先删除该文件。")
        return save_path

    print("[信息] 正在下载 VGG16 ImageNet 预训练权重...")
    try:
        model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    except Exception:
        model = torchvision.models.vgg16(pretrained=True)

    encoder_state_dict = model.features.state_dict()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(encoder_state_dict, save_path)

    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"[信息] 下载完成！已保存到: {save_path}")
    print(f"[信息] 文件大小: {size_mb:.1f} MB")
    print(f"[信息] 包含 {len(encoder_state_dict)} 个参数层")
    return save_path


if __name__ == "__main__":
    download_vgg16_encoder()
