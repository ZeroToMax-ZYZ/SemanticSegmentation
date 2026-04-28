import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def load_camvid_classes(class_dict_path):
    """Read CamVid class names and RGB colors from ``class_dict.csv``.

    Args:
        class_dict_path (str | Path): Path to the CamVid class dictionary CSV.

    Returns:
        tuple[list[str], list[tuple[int, int, int]]]: Class names and RGB values
        in file order. The row index becomes the class id.
    """
    class_dict_path = Path(class_dict_path)
    if not class_dict_path.exists():
        raise FileNotFoundError(f"class_dict.csv not found: {class_dict_path}")

    class_names = []
    rgb_values = []
    with class_dict_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row["name"].strip())
            rgb_values.append(
                (
                    int(row["r"].strip()),
                    int(row["g"].strip()),
                    int(row["b"].strip()),
                )
            )

    if not class_names:
        raise ValueError(f"No classes found in {class_dict_path}")
    return class_names, rgb_values


def _label_stem(label_path):
    """Convert a CamVid label filename stem to the matching image stem.

    Args:
        label_path (Path): Label file path, usually ending with ``_L.png``.

    Returns:
        str: Stem used by the corresponding image file.
    """
    stem = label_path.stem
    return stem[:-2] if stem.endswith("_L") else stem


class CamVidDataset(Dataset):
    """PyTorch Dataset for CamVid semantic segmentation.

    The dataset reads RGB images and RGB color masks with OpenCV, converts mask
    colors to integer class ids using ``class_dict.csv``, applies optional
    Albumentations transforms, and returns tensors suitable for
    ``CrossEntropyLoss``.
    """

    def __init__(
        self,
        root="CamVid",
        split="train",
        transform=None,
        image_dir=None,
        mask_dir=None,
        class_dict_path=None,
        ignore_index=255,
        ignore_void=False,
        return_paths=False,
    ):
        """Create a CamVid dataset split.

        Args:
            root (str | Path): Dataset root containing split folders and
                ``class_dict.csv``.
            split (str): Split name, such as ``train``, ``val``, or ``test``.
            transform (callable | None): Albumentations transform that accepts
                ``image`` and ``mask``.
            image_dir (str | Path | None): Optional explicit image directory.
            mask_dir (str | Path | None): Optional explicit mask directory.
            class_dict_path (str | Path | None): Optional class dictionary path.
            ignore_index (int): Label id reserved for ignored pixels.
            ignore_void (bool): Whether to map the ``Void`` class to
                ``ignore_index``.
            return_paths (bool): Whether to return image and mask paths in
                addition to tensors.

        Returns:
            None.
        """
        self.root = Path(root)
        self.split = split
        self.image_dir = Path(image_dir) if image_dir is not None else self.root / split
        self.mask_dir = Path(mask_dir) if mask_dir is not None else self.root / f"{split}_labels"
        self.class_dict_path = (
            Path(class_dict_path) if class_dict_path is not None else self.root / "class_dict.csv"
        )
        self.transform = transform
        self.ignore_index = ignore_index
        self.ignore_void = ignore_void
        self.return_paths = return_paths

        self.class_names, self.rgb_values = load_camvid_classes(self.class_dict_path)
        self.classes = self.class_names
        self.num_classes = len(self.class_names)
        self.void_class_index = self._find_class_index("void")
        self.color_to_class = {rgb: idx for idx, rgb in enumerate(self.rgb_values)}
        self._color_lut = self._build_color_lut()
        self.image_paths, self.mask_paths = self._collect_pairs()

    def __len__(self):
        """Return the number of paired image/mask samples.

        Args:
            None.

        Returns:
            int: Dataset length.
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """Load and transform one segmentation sample.

        Args:
            index (int): Sample index.

        Returns:
            tuple[Tensor, Tensor] | tuple[Tensor, Tensor, str, str]: Image tensor
            with shape ``[C, H, W]`` and mask tensor with shape ``[H, W]``. Paths
            are appended when ``return_paths=True``.
        """
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # OpenCV reads BGR by default; convert to RGB because augmentations and
        # TensorBoard visualization expect standard RGB ordering.
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Masks are stored as RGB colors. They are converted to class ids before
        # Albumentations so geometric transforms use nearest-neighbor labels.
        mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self._encode_mask(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask)

        if not torch.is_tensor(image):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)
        mask = mask.long()

        if self.return_paths:
            return image, mask, str(image_path), str(mask_path)
        return image, mask

    def _collect_pairs(self):
        """Collect image/mask path pairs for the current split.

        Args:
            None.

        Returns:
            tuple[list[Path], list[Path]]: Sorted image paths and mask paths.
        """
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        image_by_stem = {
            path.stem: path
            for path in sorted(self.image_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
        }

        pairs = []
        for mask_path in sorted(self.mask_dir.iterdir()):
            if not mask_path.is_file() or mask_path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            image_path = image_by_stem.get(_label_stem(mask_path))
            if image_path is not None:
                pairs.append((image_path, mask_path))

        if not pairs:
            raise RuntimeError(
                f"No CamVid image/mask pairs found in {self.image_dir} and {self.mask_dir}"
            )

        image_paths, mask_paths = zip(*pairs)
        return list(image_paths), list(mask_paths)

    def _encode_mask(self, rgb_mask):
        """Convert an RGB color mask to a class-index mask.

        Args:
            rgb_mask (np.ndarray): RGB mask with shape ``[H, W, 3]``.

        Returns:
            np.ndarray: Integer mask with shape ``[H, W]``.
        """
        # Pack RGB values into a single integer key so the lookup is vectorized
        # instead of scanning all classes for every pixel.
        keys = (
            rgb_mask[:, :, 0].astype(np.int32) * 256 * 256
            + rgb_mask[:, :, 1].astype(np.int32) * 256
            + rgb_mask[:, :, 2].astype(np.int32)
        )
        return self._color_lut[keys]

    def _build_color_lut(self):
        """Build a lookup table from packed RGB keys to class ids.

        Args:
            None.

        Returns:
            np.ndarray: Lookup table indexed by ``r * 256^2 + g * 256 + b``.
        """
        lut = np.full(256 * 256 * 256, self.ignore_index, dtype=np.uint8)
        for class_index, rgb in enumerate(self.rgb_values):
            if self.ignore_void and self.class_names[class_index].lower() == "void":
                continue
            key = rgb[0] * 256 * 256 + rgb[1] * 256 + rgb[2]
            lut[key] = class_index
        return lut

    def _find_class_index(self, class_name):
        """Find the class index by name.

        Args:
            class_name (str): Class name to look up, case-insensitive.

        Returns:
            int | None: Class index if found, otherwise None.
        """
        class_name = class_name.lower()
        for index, name in enumerate(self.class_names):
            if name.lower() == class_name:
                return index
        return None
