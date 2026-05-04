"""根据配置构建 train / val / test 的 Dataset 和 DataLoader。

支持的数据集通过 cfg['dataset_name'] 指定，默认 'CamVid_11'。
"""

import os
from torch.utils.data import DataLoader

from dataset.CamVid_dataset import CamVid_11_Dataset
from dataset.STS_2d_dataset import STS_2D_Dataset
from dataset.augment import get_train_transform, get_val_transform


# ---- 数据集注册表 ----

DATASET_REGISTRY = {
    'CamVid_11': CamVid_11_Dataset,
    'STS_2D': STS_2D_Dataset,
}


def _build_camvid_dirs(data_root, split):
    """CamVid 目录布局: data_root/split + data_root/split_labels"""
    return os.path.join(data_root, split), os.path.join(data_root, f'{split}_labels')


def _build_sts2d_dirs(data_root, split):
    """STS-2D 目录布局: data_root/split/image + data_root/split/mask"""
    return os.path.join(data_root, split, 'image'), os.path.join(data_root, split, 'mask')


DIR_BUILDERS = {
    'CamVid_11': _build_camvid_dirs,
    'STS_2D': _build_sts2d_dirs,
}


def _create_dataset(dataset_name, img_dir, mask_dir, cfg, transform):
    """根据数据集名称创建对应的 Dataset 实例。"""
    if dataset_name == 'CamVid_11':
        json_file = os.path.join(cfg['data_root'], 'camvid_11class_mapping.json')
        if not os.path.exists(json_file):
            json_file = None
        return CamVid_11_Dataset(
            img_dir=img_dir, mask_dir=mask_dir,
            json_file=json_file, transform=transform,
        )
    elif dataset_name == 'STS_2D':
        return STS_2D_Dataset(
            img_dir=img_dir, mask_dir=mask_dir, transform=transform,
            num_classes=cfg['num_classes'],
            ignore_index=cfg.get('ignore_index', 255),
        )
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')


def build_dataset(cfg):
    """根据配置构建 train / val / test 的 Dataset 和 DataLoader。

    Args:
        cfg: dict, 训练配置（从yaml读取）

    Returns:
        dataloaders: dict, 包含 'train', 'val', 'test' 的 DataLoader
        dataset_sizes: dict, 各子集样本数
    """
    data_root = cfg['data_root']
    image_size = tuple(cfg['image_size'])
    batch_size = cfg['batch_size']
    dataset_name = cfg.get('dataset_name', 'CamVid_11')
    num_workers = cfg.get('num_workers', 0)
    pin_memory = cfg.get('pin_memory', 'auto')
    if pin_memory == 'auto':
        import torch
        pin_memory = torch.cuda.is_available()

    # 数据增强
    train_transform = get_train_transform(image_size)
    val_transform = get_val_transform(image_size)

    # 获取目录布局函数
    dir_builder = DIR_BUILDERS.get(dataset_name)
    if dir_builder is None:
        raise ValueError(f'No dir layout defined for dataset: {dataset_name}')

    # 构建各子集
    subsets = {}
    for split, transform in [('train', train_transform), ('val', val_transform), ('test', val_transform)]:
        img_dir, mask_dir = dir_builder(data_root, split)

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f'[build_dataset] {split} dir not found, skip')
            continue

        subsets[split] = _create_dataset(dataset_name, img_dir, mask_dir, cfg, transform)

    # 构建DataLoader
    shuffle = {'train': True, 'val': False, 'test': False}
    dataloaders = {}
    dataset_sizes = {}

    for split, ds in subsets.items():
        dataloaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle[split],
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train'),
        )
        dataset_sizes[split] = len(ds)

    return dataloaders, dataset_sizes
