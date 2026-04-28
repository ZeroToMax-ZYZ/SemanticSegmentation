import albumentations as A
from albumentations.pytorch import ToTensorV2


CAMVID_MEAN = (0.485, 0.456, 0.406)
CAMVID_STD = (0.229, 0.224, 0.225)


def _to_hw(image_size):
    """Normalize an image size argument to ``(height, width)``.

    Args:
        image_size (int | Sequence[int]): Scalar size or two-element size.

    Returns:
        tuple[int, int]: Height and width.
    """
    if isinstance(image_size, int):
        return image_size, image_size
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or a (height, width) tuple")
    return int(image_size[0]), int(image_size[1])


def get_train_transform(
    image_size=(360, 480),
    mean=CAMVID_MEAN,
    std=CAMVID_STD,
    mask_fill_value=30,
):
    """Create the training augmentation pipeline.

    Args:
        image_size (int | tuple[int, int]): Output size as height and width.
        mean (tuple[float, float, float]): RGB normalization mean.
        std (tuple[float, float, float]): RGB normalization std.
        mask_fill_value (int): Label id used for newly created mask borders.

    Returns:
        albumentations.Compose: Transform that returns image and mask tensors.
    """
    height, width = _to_hw(image_size)
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.HorizontalFlip(p=0.5),
            # Geometric augmentation must use a valid class id or ignore_index
            # for mask borders because CrossEntropyLoss expects integer labels.
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=0,
                value=0,
                mask_value=mask_fill_value,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3,
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def get_eval_transform(
    image_size=(360, 480),
    mean=CAMVID_MEAN,
    std=CAMVID_STD,
):
    """Create the validation/test preprocessing pipeline.

    Args:
        image_size (int | tuple[int, int]): Output size as height and width.
        mean (tuple[float, float, float]): RGB normalization mean.
        std (tuple[float, float, float]): RGB normalization std.

    Returns:
        albumentations.Compose: Deterministic transform for evaluation.
    """
    height, width = _to_hw(image_size)
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def get_camvid_transform(
    split="train",
    image_size=(360, 480),
    mean=CAMVID_MEAN,
    std=CAMVID_STD,
    mask_fill_value=30,
):
    """Dispatch the correct CamVid transform by split name.

    Args:
        split (str): ``train``, ``val``, or ``test``.
        image_size (int | tuple[int, int]): Output size as height and width.
        mean (tuple[float, float, float]): RGB normalization mean.
        std (tuple[float, float, float]): RGB normalization std.
        mask_fill_value (int): Label id used for train-time mask borders.

    Returns:
        albumentations.Compose: Split-specific transform.
    """
    if split == "train":
        return get_train_transform(
            image_size=image_size,
            mean=mean,
            std=std,
            mask_fill_value=mask_fill_value,
        )
    if split in {"val", "test"}:
        return get_eval_transform(image_size=image_size, mean=mean, std=std)
    raise ValueError(f"Unsupported split: {split}")
