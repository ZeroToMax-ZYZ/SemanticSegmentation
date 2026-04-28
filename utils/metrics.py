import torch


class SegmentationMetricMeter:
    """Accumulate semantic segmentation metrics with a confusion matrix."""

    def __init__(self, num_classes, ignore_index=255, ignore_classes=None):
        """Create a metric accumulator.

        Args:
            num_classes (int): Number of semantic classes.
            ignore_index (int): Label value ignored at pixel level.
            ignore_classes (list[int] | None): Class ids excluded from averaged
                metrics such as mIoU and mean accuracy.

        Returns:
            None.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ignore_classes = set(ignore_classes or [])
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    def reset(self):
        """Reset the accumulated confusion matrix.

        Args:
            None.

        Returns:
            None.
        """
        self.confusion_matrix.zero_()

    @torch.no_grad()
    def update(self, logits, target):
        """Update metrics from one batch of logits and labels.

        Args:
            logits (Tensor): Segmentation logits with shape ``[B, C, H, W]``.
            target (Tensor): Integer labels with shape ``[B, H, W]``.

        Returns:
            None.
        """
        if logits.shape[-2:] != target.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        pred = torch.argmax(logits, dim=1).detach().cpu().view(-1)
        target = target.detach().cpu().view(-1)

        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        if not valid.any():
            return

        target = target[valid].long()
        pred = pred[valid].long().clamp(0, self.num_classes - 1)
        # Encode each (target, pred) pair into one integer so bincount can build
        # the confusion matrix efficiently.
        encoded = target * self.num_classes + pred
        hist = torch.bincount(encoded, minlength=self.num_classes ** 2)
        self.confusion_matrix += hist.reshape(self.num_classes, self.num_classes).double()

    def compute(self):
        """Compute scalar metrics from the accumulated confusion matrix.

        Args:
            None.

        Returns:
            dict: Contains ``miou``, ``pixel_acc``, ``mean_acc``, and
            ``class_iou``.
        """
        hist = self.confusion_matrix
        tp = torch.diag(hist)
        gt = hist.sum(dim=1)
        pred = hist.sum(dim=0)
        union = gt + pred - tp

        valid_classes = union > 0
        for class_index in self.ignore_classes:
            if 0 <= class_index < self.num_classes:
                valid_classes[class_index] = False

        iou = torch.zeros(self.num_classes, dtype=torch.float64)
        iou[union > 0] = tp[union > 0] / union[union > 0]
        miou = iou[valid_classes].mean().item() if valid_classes.any() else 0.0

        class_acc = torch.zeros(self.num_classes, dtype=torch.float64)
        class_acc[gt > 0] = tp[gt > 0] / gt[gt > 0]
        valid_acc = gt > 0
        for class_index in self.ignore_classes:
            if 0 <= class_index < self.num_classes:
                valid_acc[class_index] = False
        mean_acc = class_acc[valid_acc].mean().item() if valid_acc.any() else 0.0

        total = hist.sum()
        pixel_acc = tp.sum().item() / total.item() if total > 0 else 0.0

        return {
            "miou": miou,
            "pixel_acc": pixel_acc,
            "mean_acc": mean_acc,
            "class_iou": iou.tolist(),
        }


def cal_miou(pred, label, num_classes, ignore_index=255):
    """Compatibility helper to compute mIoU for a single prediction batch.

    Args:
        pred (Tensor): Logits with shape ``[B, C, H, W]``.
        label (Tensor): Integer labels with shape ``[B, H, W]``.
        num_classes (int): Number of classes.
        ignore_index (int): Ignored label value.

    Returns:
        tuple[float, list[float]]: mIoU and per-class IoU values.
    """
    meter = SegmentationMetricMeter(num_classes=num_classes, ignore_index=ignore_index)
    meter.update(pred, label)
    metrics = meter.compute()
    return metrics["miou"], metrics["class_iou"]
