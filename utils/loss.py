import torch
import torch.nn as nn
import torch.nn.functional as F


def get_main_logits(outputs):
    """Extract the main logits tensor from common model output formats.

    Args:
        outputs (Tensor | dict | tuple | list): Raw model outputs.

    Returns:
        Tensor: Main segmentation logits with shape ``[B, C, H, W]``.
    """
    if isinstance(outputs, dict):
        for key in ("out", "logits", "main"):
            if key in outputs:
                return outputs[key]
        return next(iter(outputs.values()))
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def soft_dice_loss(logits, target, num_classes, ignore_index=255, eps=1e-6):
    """Compute multi-class soft Dice loss.

    Args:
        logits (Tensor): Model logits with shape ``[B, C, H, W]``.
        target (Tensor): Integer labels with shape ``[B, H, W]``.
        num_classes (int): Number of semantic classes.
        ignore_index (int): Target value to exclude from loss.
        eps (float): Numerical stability constant.

    Returns:
        Tensor: Scalar Dice loss.
    """
    valid_mask = target != ignore_index
    if not valid_mask.any():
        return logits.sum() * 0.0

    probs = torch.softmax(logits, dim=1)
    target = target.clamp(min=0, max=num_classes - 1)
    one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_mask = valid_mask.unsqueeze(1)

    probs = probs * valid_mask
    one_hot = one_hot * valid_mask

    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    denominator = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """Composite semantic segmentation loss.

    Supports CrossEntropy-only training by default and optional Dice loss. It
    also accepts auxiliary outputs from tuple/list/dict model outputs.
    """

    def __init__(
        self,
        num_classes,
        ignore_index=255,
        ce_weight=1.0,
        dice_weight=0.0,
        aux_weight=0.4,
    ):
        """Initialize the loss module.

        Args:
            num_classes (int): Number of semantic classes.
            ignore_index (int): Label value ignored by CrossEntropyLoss.
            ce_weight (float): Weight for cross entropy.
            dice_weight (float): Weight for soft Dice loss.
            aux_weight (float): Weight for each auxiliary output loss.

        Returns:
            None.
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, outputs, target):
        """Compute segmentation loss for main and optional auxiliary outputs.

        Args:
            outputs (Tensor | dict | tuple | list): Model outputs.
            target (Tensor): Integer labels with shape ``[B, H, W]``.

        Returns:
            tuple[Tensor, dict[str, float]]: Scalar loss and detached loss items
            for logging.
        """
        main_logits = get_main_logits(outputs)
        loss, loss_items = self._loss_one(main_logits, target, prefix="")

        aux_outputs = []
        if isinstance(outputs, dict):
            aux_outputs = [v for k, v in outputs.items() if k not in {"out", "logits", "main"}]
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 1:
            aux_outputs = outputs[1:]

        for idx, aux_logits in enumerate(aux_outputs):
            aux_loss, aux_items = self._loss_one(aux_logits, target, prefix=f"aux{idx}_")
            loss = loss + self.aux_weight * aux_loss
            loss_items.update(aux_items)

        loss_items["total_loss"] = float(loss.detach().item())
        return loss, loss_items

    def _loss_one(self, logits, target, prefix=""):
        """Compute CE/Dice loss for one logits tensor.

        Args:
            logits (Tensor): Logits with shape ``[B, C, H, W]``.
            target (Tensor): Integer labels with shape ``[B, H, W]``.
            prefix (str): Prefix used for logging auxiliary loss names.

        Returns:
            tuple[Tensor, dict[str, float]]: Scalar loss and logging values.
        """
        if logits.shape[-2:] != target.shape[-2:]:
            # Align logits to label resolution for models that output a lower
            # resolution feature map.
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)

        ce = self.ce_loss(logits, target)
        total = self.ce_weight * ce
        items = {f"{prefix}ce_loss": float(ce.detach().item())}

        if self.dice_weight > 0:
            dice = soft_dice_loss(
                logits,
                target,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
            )
            total = total + self.dice_weight * dice
            items[f"{prefix}dice_loss"] = float(dice.detach().item())

        return total, items


def build_loss_fn(cfg):
    """Build the configured segmentation loss.

    Args:
        cfg (dict): Resolved training config.

    Returns:
        SegmentationLoss: Loss module.
    """
    loss_cfg = cfg.get("loss", {})
    loss_type = loss_cfg.get("type", cfg.get("loss_fn", "CrossEntropyLoss"))
    if loss_type not in {"CrossEntropyLoss", "CrossEntropyDiceLoss", "SegmentationLoss"}:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    dice_weight = loss_cfg.get("dice_weight", 0.0)
    if loss_type == "CrossEntropyDiceLoss":
        dice_weight = loss_cfg.get("dice_weight", 1.0)

    return SegmentationLoss(
        num_classes=cfg["num_classes"],
        ignore_index=cfg.get("ignore_index", 255),
        ce_weight=loss_cfg.get("ce_weight", 1.0),
        dice_weight=dice_weight,
        aux_weight=loss_cfg.get("aux_weight", 0.4),
    )
