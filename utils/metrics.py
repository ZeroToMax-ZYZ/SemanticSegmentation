import torch


class SegmentationMetricMeter:
    def __init__(self, num_classes, ignore_index=255, ignore_classes=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ignore_classes = set(ignore_classes or [])
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    def reset(self):
        self.confusion_matrix.zero_()

    @torch.no_grad()
    def update(self, logits, target):
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
        encoded = target * self.num_classes + pred
        hist = torch.bincount(encoded, minlength=self.num_classes ** 2)
        self.confusion_matrix += hist.reshape(self.num_classes, self.num_classes).double()

    def compute(self):
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
    meter = SegmentationMetricMeter(num_classes=num_classes, ignore_index=ignore_index)
    meter.update(pred, label)
    metrics = meter.compute()
    return metrics["miou"], metrics["class_iou"]
