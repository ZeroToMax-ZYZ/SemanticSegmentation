"""语义分割数据集基类，封装通用的 __getitem__ / __len__ / tensor 转换逻辑。

子类只需实现:
    - collate_data(img_dir, mask_dir): 返回 (list_imgs, list_labels)
    - 设置 self.rgb_values / self.num_classes / self.ignore_index
可选覆盖:
    - _postprocess_mask(mask): 对原始 mask 做后处理（如二值化）
"""

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class BaseSegDataset(Dataset):
    """语义分割数据集基类。"""

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # 子类必须在 super().__init__ 之前设置 self.rgb_values / num_classes / ignore_index
        self.list_imgs, self.list_labels = self.collate_data(img_dir, mask_dir)

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, idx):
        img_path = self.list_imgs[idx]
        mask_path = self.list_labels[idx]

        # 读取图片和mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 子类可覆盖的mask后处理（如二值化阈值）
        mask = self._postprocess_mask(mask)

        # albumentations数据增强（同时处理image和mask）
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 转tensor（如果没有transform则手动转换）
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
        if isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask.astype(np.int64))

        return image, mask

    def _postprocess_mask(self, mask):
        """对读取的原始mask做后处理，默认不做任何修改。子类可覆盖。"""
        return mask

    @abstractmethod
    def collate_data(self, img_dir, mask_dir):
        """收集并校验图片-mask路径对。子类必须实现。

        Returns:
            (list_imgs, list_labels): 两个等长的路径列表
        """
        raise NotImplementedError
