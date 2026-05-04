"""STS-2D 二值分割数据集。

目录结构: split/image/ 存放图片, split/mask/ 存放mask
mask命名: 与图片文件名相同（如 A-1.png -> A-1.png）
mask值: 0=背景, 255=前景，需阈值化为 0/1
"""

import os
import numpy as np

from dataset.base_dataset import BaseSegDataset


class STS_2D_Dataset(BaseSegDataset):
    """STS-2D 二值分割数据集。"""

    def __init__(self, img_dir, mask_dir, transform=None, num_classes=2, ignore_index=255):
        # 二值分割调色板（用于TensorBoard可视化）
        self.rgb_values = [
            (0, 0, 0),        # 0: 背景 (黑色)
            (255, 255, 255),   # 1: 前景 (白色)
        ]
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # 调用基类构造（会触发 collate_data）
        super().__init__(img_dir, mask_dir, transform=transform)

    def _postprocess_mask(self, mask):
        """将 0/255 的 mask 阈值化为 0/1 的类别索引。"""
        mask = (mask > 127).astype(np.uint8)
        return mask

    def collate_data(self, img_dir, mask_dir):
        """收集图片和mask路径，校验一一对应关系。

        mask命名规则: 与图片文件名完全相同
        """
        list_imgs, list_labels = [], []
        img_files = sorted(os.listdir(img_dir))

        for img_name in img_files:
            mask_path = os.path.join(mask_dir, img_name)
            if not os.path.exists(mask_path):
                print(f'[WARNING] mask not found, skip: {img_name}')
                continue
            list_imgs.append(os.path.join(img_dir, img_name))
            list_labels.append(mask_path)

        assert len(list_imgs) > 0, f'No valid pairs found in {img_dir} / {mask_dir}'
        print(f'[STS-2D] Found {len(list_imgs)} valid image-mask pairs')

        return list_imgs, list_labels
