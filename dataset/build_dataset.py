import os
from torch.utils.data import DataLoader

from dataset.CamVid_dataset import CamVid_11_Dataset
from dataset.augment import get_train_transform, get_val_transform


def build_dataset(cfg):
    '''
    根据配置构建 train / val / test 的 Dataset 和 DataLoader
    Args:
        cfg: dict, 训练配置（从yaml读取）
    Returns:
        dataloaders: dict, 包含 'train', 'val', 'test' 的DataLoader
        dataset_sizes: dict, 各子集样本数
    '''
    data_root = cfg['data_root']
    image_size = tuple(cfg['image_size'])
    batch_size = cfg['batch_size']
    num_workers = cfg.get('num_workers', 0)
    pin_memory = cfg.get('pin_memory', 'auto')
    if pin_memory == 'auto':
        import torch
        pin_memory = torch.cuda.is_available()

    # json映射文件
    json_file = os.path.join(data_root, 'camvid_11class_mapping.json')
    if not os.path.exists(json_file):
        json_file = None

    # 数据增强
    train_transform = get_train_transform(image_size)
    val_transform = get_val_transform(image_size)

    # 构建各子集
    subsets = {}
    for split, transform in [('train', train_transform), ('val', val_transform), ('test', val_transform)]:
        img_dir = os.path.join(data_root, split)
        mask_dir = os.path.join(data_root, f'{split}_labels')

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f'[build_dataset] {split} dir not found, skip')
            continue

        subsets[split] = CamVid_11_Dataset(
            img_dir=img_dir,
            mask_dir=mask_dir,
            json_file=json_file,
            transform=transform,
        )

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

