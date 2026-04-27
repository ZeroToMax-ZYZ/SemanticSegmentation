import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.logger_tensorb import histogram_tensorb_logger, iter_tensorb_logger
from utils.loss import get_main_logits
from utils.metrics import SegmentationMetricMeter


@dataclass
class Checkpoint:
    train_miou: float = 0.0
    val_miou: float = 0.0
    train_pixel_acc: float = 0.0
    val_pixel_acc: float = 0.0
    best_val_miou: float = 0.0


def _prepare_logits(outputs, labels):
    logits = get_main_logits(outputs)
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def fit_train_epoch(epoch, cfg, model, train_loader, loss_fn, optimizer, writer):
    model.train()

    train_loss = 0.0
    samples = 0
    last_outputs = None
    meter = SegmentationMetricMeter(
        num_classes=cfg["num_classes"],
        ignore_index=cfg.get("ignore_index", 255),
        ignore_classes=cfg.get("metric_ignore_classes", []),
    )

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']} [Train]")
    max_batches = cfg.get("debug_max_train_batches")

    for batch_idx, (images, labels) in enumerate(train_bar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        bs = images.shape[0]
        samples += bs

        images = images.to(cfg["device"], non_blocking=True)
        labels = labels.to(cfg["device"], non_blocking=True)

        outputs = model(images)
        loss, loss_item = loss_fn(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        logits = _prepare_logits(outputs, labels)
        meter.update(logits.detach(), labels.detach())
        last_outputs = logits.detach()

        iter_num = epoch * len(train_loader) + batch_idx
        iter_tensorb_logger(writer, loss_item, iter_num)

        train_loss += loss.item() * bs
        train_bar.set_postfix(loss=f"{train_loss / max(samples, 1):.4f}")

    if last_outputs is not None:
        histogram_tensorb_logger(writer, model, last_outputs, epoch)

    metrics = meter.compute()
    epoch_loss = train_loss / max(samples, 1)
    return epoch_loss, metrics


def fit_val_epoch(epoch, cfg, model, val_loader, loss_fn):
    model.eval()

    val_loss = 0.0
    samples = 0
    meter = SegmentationMetricMeter(
        num_classes=cfg["num_classes"],
        ignore_index=cfg.get("ignore_index", 255),
        ignore_classes=cfg.get("metric_ignore_classes", []),
    )

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']} [Val]")
    max_batches = cfg.get("debug_max_val_batches")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_bar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            bs = images.shape[0]
            samples += bs

            images = images.to(cfg["device"], non_blocking=True)
            labels = labels.to(cfg["device"], non_blocking=True)

            outputs = model(images)
            loss, _ = loss_fn(outputs, labels)

            logits = _prepare_logits(outputs, labels)
            meter.update(logits.detach(), labels.detach())

            val_loss += loss.item() * bs
            val_bar.set_postfix(loss=f"{val_loss / max(samples, 1):.4f}")

    metrics = meter.compute()
    epoch_loss = val_loss / max(samples, 1)
    return epoch_loss, metrics


def fit_one_epoch(epoch, cfg, model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler, writer, state=None):
    if state is None:
        state = Checkpoint()

    start_time = time.time()
    train_loss, train_metrics = fit_train_epoch(
        epoch, cfg, model, train_loader, loss_fn, optimizer, writer
    )
    val_loss, val_metrics = fit_val_epoch(
        epoch, cfg, model, val_loader, loss_fn
    )

    lr = optimizer.param_groups[0]["lr"]
    if lr_scheduler is not None:
        lr_scheduler.step()

    state.train_miou = train_metrics["miou"]
    state.train_pixel_acc = train_metrics["pixel_acc"]
    state.val_miou = val_metrics["miou"]
    state.val_pixel_acc = val_metrics["pixel_acc"]

    metrics = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_miou": state.train_miou,
        "train_pixel_acc": state.train_pixel_acc,
        "train_mean_acc": train_metrics["mean_acc"],
        "val_loss": val_loss,
        "val_miou": state.val_miou,
        "val_pixel_acc": state.val_pixel_acc,
        "val_mean_acc": val_metrics["mean_acc"],
        "lr": lr,
        "epoch_time": time.time() - start_time,
        "is_best": None,
    }
    return metrics, state
