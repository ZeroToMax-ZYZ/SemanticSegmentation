import torch
from torch.utils.data import Dataset

import os
import json
import numpy as np
import cv2


class CamVid_11_Dataset(Dataset):
    '''
    读取11类的camvid数据集，具体名称映射在json文件
    mask文件名 = 图片文件名去掉扩展名 + '_L.png'
    '''
    def __init__(self, img_dir, mask_dir, json_file=None, transform=None):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.transform = transform

        # 加载类别映射
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

        # 收集并校验数据对
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

    def collate_data(self, img_dir, mask_dir):
        '''
        收集图片和mask路径，校验一一对应关系
        mask命名规则: stem + '_L.png'，如 0001TP_009210.png -> 0001TP_009210_L.png
        '''
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