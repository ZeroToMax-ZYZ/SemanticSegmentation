"""CamVid 11类语义分割数据集。

mask文件名 = 图片文件名去掉扩展名 + '_L.png'
"""

import os
import json

from dataset.base_dataset import BaseSegDataset


class CamVid_11_Dataset(BaseSegDataset):
    """读取11类的CamVid数据集。

    目录结构: split/ 存放图片, split_labels/ 存放mask
    mask命名规则: stem + '_L.png'（如 0001TP_009210.png -> 0001TP_009210_L.png）
    """

    def __init__(self, img_dir, mask_dir, json_file=None, transform=None):
        # 先设置子类特有属性（基类 __init__ 会调用 collate_data）
        self.ignore_index = 255
        self.num_classes = 11
        # CamVid 11类调色板（用于TensorBoard可视化）
        self.rgb_values = [
            (128, 128, 128),   # 0: Sky
            (128, 0, 0),       # 1: Building
            (192, 192, 128),   # 2: Pole
            (128, 64, 128),    # 3: Road
            (0, 0, 192),       # 4: Sidewalk
            (128, 128, 0),     # 5: Tree
            (192, 128, 128),   # 6: SignSymbol
            (64, 64, 128),     # 7: Fence
            (64, 0, 128),      # 8: Car
            (64, 64, 0),       # 9: Pedestrian
            (0, 128, 192),     # 10: Bicyclist
        ]
        if json_file and os.path.exists(json_file):
            with open(json_file, 'r') as f:
                mapping = json.load(f)
            self.ignore_index = mapping.get('ignore_index', 255)

        # 调用基类构造（会触发 collate_data）
        super().__init__(img_dir, mask_dir, transform=transform)

    def collate_data(self, img_dir, mask_dir):
        """收集图片和mask路径，校验一一对应关系。

        mask命名规则: stem + '_L.png'，如 0001TP_009210.png -> 0001TP_009210_L.png
        """
        list_imgs, list_labels = [], []

        img_files = sorted(os.listdir(img_dir))

        for img_name in img_files:
            stem, _ = os.path.splitext(img_name)
            mask_name = stem + '_L.png'
            mask_path = os.path.join(mask_dir, mask_name)

            if not os.path.exists(mask_path):
                print(f'[WARNING] mask not found, skip: {img_name} -> {mask_name}')
                continue

            list_imgs.append(os.path.join(img_dir, img_name))
            list_labels.append(mask_path)

        assert len(list_imgs) > 0, f'No valid pairs found in {img_dir} / {mask_dir}'
        print(f'[CamVid] Found {len(list_imgs)} valid image-mask pairs')

        return list_imgs, list_labels
