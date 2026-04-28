from torch.utils.data import DataLoader, Subset

try:
    from .augment import get_camvid_transform
    from .CamVid_dataset import CamVidDataset
except ImportError:
    from augment import get_camvid_transform
    from CamVid_dataset import CamVidDataset


def build_camvid_dataset(
    root="CamVid",
    split="train",
    image_size=(360, 480),
    transform=None,
    ignore_index=255,
    ignore_void=False,
    void_class_index=None,
    return_paths=False,
):
    """Build a CamVid Dataset instance and attach transforms.

    Args:
        root (str): CamVid root directory.
        split (str): Dataset split name.
        image_size (int | tuple[int, int]): Transform output size as
            ``height, width``.
        transform (callable | None): Optional transform. If None, a split-aware
            default transform is created.
        ignore_index (int): Label value ignored by the loss.
        ignore_void (bool): Whether Void pixels are mapped to ignore_index.
        void_class_index (int | None): Optional configured Void class id.
        return_paths (bool): Whether dataset samples include source paths.

    Returns:
        CamVidDataset: Dataset with transform attached.
    """
    # Build the dataset before transforms so the class_dict.csv can be used to
    # infer the Void class id for augmentation padding.
    dataset = CamVidDataset(
        root=root,
        split=split,
        transform=None,
        ignore_index=ignore_index,
        ignore_void=ignore_void,
        return_paths=return_paths,
    )

    if transform is None:
        mask_fill_value = _resolve_mask_fill_value(
            dataset=dataset,
            ignore_index=ignore_index,
            ignore_void=ignore_void,
            void_class_index=void_class_index,
        )
        transform = get_camvid_transform(
            split=split,
            image_size=image_size,
            mask_fill_value=mask_fill_value,
        )
    dataset.transform = transform
    return dataset


def _resolve_mask_fill_value(dataset, ignore_index, ignore_void, void_class_index=None):
    """Resolve the label value used to fill mask borders during augmentation.

    Args:
        dataset (CamVidDataset): Dataset that may expose ``void_class_index``.
        ignore_index (int): Ignored label id.
        ignore_void (bool): Whether Void should be ignored by loss/metrics.
        void_class_index (int | None): Configured Void class index.

    Returns:
        int: Label id used by Albumentations when new mask pixels are created.
    """
    if ignore_void:
        return ignore_index
    if void_class_index is not None:
        return int(void_class_index)
    if getattr(dataset, "void_class_index", None) is not None:
        return int(dataset.void_class_index)
    return ignore_index


def build_camvid_dataloader(
    root="CamVid",
    split="train",
    image_size=(360, 480),
    batch_size=4,
    num_workers=0,
    shuffle=None,
    pin_memory=True,
    drop_last=None,
    ignore_index=255,
    ignore_void=False,
    void_class_index=None,
    persistent_workers=False,
):
    """Build a DataLoader for one CamVid split.

    Args:
        root (str): CamVid root directory.
        split (str): Split name.
        image_size (int | tuple[int, int]): Transform output size.
        batch_size (int): Batch size.
        num_workers (int): Number of DataLoader workers.
        shuffle (bool | None): Shuffle flag. Defaults to True for train only.
        pin_memory (bool): Whether to use pinned memory.
        drop_last (bool | None): Drop incomplete final batch. Defaults to True
            for train only.
        ignore_index (int): Ignored label id.
        ignore_void (bool): Whether to ignore Void pixels.
        void_class_index (int | None): Optional configured Void class index.
        persistent_workers (bool): Keep workers alive between epochs.

    Returns:
        DataLoader: DataLoader for the requested split.
    """
    dataset = build_camvid_dataset(
        root=root,
        split=split,
        image_size=image_size,
        ignore_index=ignore_index,
        ignore_void=ignore_void,
        void_class_index=void_class_index,
    )

    if shuffle is None:
        shuffle = split == "train"
    if drop_last is None:
        drop_last = split == "train"

    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers

    return DataLoader(dataset, **dataloader_kwargs)


def build_camvid_dataloaders(
    root="CamVid",
    image_size=(360, 480),
    batch_size=4,
    num_workers=0,
    pin_memory=True,
    ignore_index=255,
    ignore_void=False,
    void_class_index=None,
    include_test=False,
    persistent_workers=False,
):
    """Build train/val and optional test DataLoaders for CamVid.

    Args:
        root (str): CamVid root directory.
        image_size (int | tuple[int, int]): Transform output size.
        batch_size (int): Batch size shared by all loaders.
        num_workers (int): Number of DataLoader workers.
        pin_memory (bool): Whether to use pinned memory.
        ignore_index (int): Ignored label id.
        ignore_void (bool): Whether to ignore Void pixels.
        void_class_index (int | None): Optional configured Void class index.
        include_test (bool): Whether to include the test loader.
        persistent_workers (bool): Keep workers alive between epochs.

    Returns:
        dict[str, DataLoader]: Loader dictionary keyed by split name.
    """
    loaders = {
        "train": build_camvid_dataloader(
            root=root,
            split="train",
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            ignore_index=ignore_index,
            ignore_void=ignore_void,
            void_class_index=void_class_index,
            persistent_workers=persistent_workers,
        ),
        "val": build_camvid_dataloader(
            root=root,
            split="val",
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            ignore_index=ignore_index,
            ignore_void=ignore_void,
            void_class_index=void_class_index,
            persistent_workers=persistent_workers,
        ),
    }

    if include_test:
        loaders["test"] = build_camvid_dataloader(
            root=root,
            split="test",
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            ignore_index=ignore_index,
            ignore_void=ignore_void,
            void_class_index=void_class_index,
            persistent_workers=persistent_workers,
        )

    return loaders


def _limit_dataset(dataloader, max_samples, shuffle=None, drop_last=None):
    """Wrap an existing DataLoader's dataset with a small Subset.

    Args:
        dataloader (DataLoader): Original loader.
        max_samples (int | None): Maximum sample count. None keeps the loader.
        shuffle (bool | None): Shuffle flag for the new loader.
        drop_last (bool | None): Drop-last flag for the new loader.

    Returns:
        DataLoader: Limited loader, or the original loader when max_samples is
        None.
    """
    if max_samples is None:
        return dataloader

    dataset = dataloader.dataset
    max_samples = min(int(max_samples), len(dataset))
    subset = Subset(dataset, list(range(max_samples)))

    return DataLoader(
        subset,
        batch_size=dataloader.batch_size,
        shuffle=(shuffle if shuffle is not None else False),
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=(drop_last if drop_last is not None else False),
    )


def build_dataset(cfg):
    """Build project DataLoaders from the global training config.

    Args:
        cfg (dict): Resolved training config.

    Returns:
        tuple[DataLoader, DataLoader, Dataset, Dataset] | dict[str, DataLoader]:
        By default returns train loader, val loader, train dataset, and val
        dataset. Returns the loader dict when ``return_loader_dict=True``.
    """
    loaders = build_camvid_dataloaders(
        root=cfg.get("data_root", cfg.get("root", "CamVid")),
        image_size=cfg.get("image_size", (360, 480)),
        batch_size=cfg.get("batch_size", 4),
        num_workers=cfg.get("num_workers", 0),
        pin_memory=cfg.get("pin_memory", True),
        ignore_index=cfg.get("ignore_index", 255),
        ignore_void=cfg.get("ignore_void", False),
        void_class_index=cfg.get("void_class_index"),
        include_test=cfg.get("include_test", False),
        persistent_workers=cfg.get("persistent_workers", False),
    )

    if cfg.get("debug_mode"):
        loaders["train"] = _limit_dataset(
            loaders["train"],
            cfg.get("debug_max_train_samples", 8),
            shuffle=True,
            drop_last=False,
        )
        loaders["val"] = _limit_dataset(
            loaders["val"],
            cfg.get("debug_max_val_samples", 4),
            shuffle=False,
            drop_last=False,
        )

    if cfg.get("return_loader_dict", False):
        return loaders

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    return train_loader, val_loader, train_loader.dataset, val_loader.dataset
