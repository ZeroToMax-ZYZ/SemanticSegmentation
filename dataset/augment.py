import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(image_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    '''
    训练集数据增强：随机裁剪、翻转、颜色抖动、归一化
    Args:
        image_size: (H, W) 目标尺寸
        mean: RGB通道均值
        std: RGB通道标准差
    '''
    h, w = image_size
    return A.Compose([
        # 几何变换
        # mask_interpolation=cv2.INTER_NEAREST 确保mask用最近邻插值，保持整数类ID
        A.RandomResizedCrop(height=h, width=w, scale=(0.08, 1.0), ratio=(0.75, 1.33),
                            interpolation=cv2.INTER_LINEAR,
                            mask_interpolation=cv2.INTER_NEAREST, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15,
                           border_mode=0, p=0.5),

        # 颜色抖动（不影响mask）
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.5),

        # 噪声 / 模糊（不影响mask）
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),

        # 归一化 + 转tensor
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transform(image_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    '''
    验证/测试集变换：仅resize + 归一化
    '''
    h, w = image_size
    return A.Compose([
        A.Resize(height=h, width=w,
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_test_transform(image_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    '''与验证集一致'''
    return get_val_transform(image_size, mean, std)

