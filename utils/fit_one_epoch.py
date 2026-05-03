import time
from dataclasses import dataclass

import torch
from tqdm import tqdm

from utils.logger_tensorb import histogram_tensorb_logger, iter_tensorb_logger
from utils.metrics import SegmentationMetricMeter


def _get_logits(outputs):
    """从模型输出中提取logits tensor，兼容dict/tuple/raw tensor"""
    if isinstance(outputs, dict):
        for key in ("out", "logits", "main"):
            if key in outputs:
                return outputs[key]
        return next(iter(outputs.values()))
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


@dataclass
class Checkpoint:
    """保存当前训练过程中最优/最新的指标状态"""
    train_miou: float = 0.0
    val_miou: float = 0.0
    train_pixel_acc: float = 0.0
    val_pixel_acc: float = 0.0
    best_val_miou: float = 0.0


def fit_train_epoch(epoch, cfg, model, train_loader, loss_fn, optimizer, writer):
    """训练一个epoch

    Args:
        epoch: 当前epoch索引（从0开始）
        cfg: 训练配置字典
        model: 语义分割模型
        train_loader: 训练集DataLoader
        loss_fn: 损失函数
        optimizer: 优化器
        writer: TensorBoard writer

    Returns:
        (epoch平均loss, 指标字典)
    """
    model.train()

    train_loss = 0.0
    samples = 0
    last_outputs = None
    meter = SegmentationMetricMeter(
        num_classes=cfg["num_classes"],
        ignore_index=cfg.get("ignore_index", 255),
        ignore_classes=cfg.get("metric_ignore_classes", []),
    )

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']} [训练]")
    max_batches = cfg.get("debug_max_train_batches")

    for batch_idx, (images, labels) in enumerate(train_bar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        bs = images.shape[0]
        samples += bs

        images = images.to(cfg["device"], non_blocking=True)
        labels = labels.to(cfg["device"], non_blocking=True)

        # 前向传播 -> 计算损失 -> 反向传播 -> 更新参数
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # 更新指标（meter内部处理输出格式和插值）
        meter.update(outputs, labels)
        last_outputs = _get_logits(outputs).detach()

        # 记录iter级别loss
        iter_num = epoch * len(train_loader) + batch_idx
        iter_tensorb_logger(writer, {"ce_loss": loss.item()}, iter_num)

        train_loss += loss.item() * bs
        train_bar.set_postfix(loss=f"{train_loss / max(samples, 1):.4f}")

    # 记录权重和梯度的直方图
    if last_outputs is not None:
        histogram_tensorb_logger(writer, model, last_outputs, epoch)

    metrics = meter.compute()
    epoch_loss = train_loss / max(samples, 1)
    return epoch_loss, metrics


def fit_val_epoch(epoch, cfg, model, val_loader, loss_fn):
    """验证一个epoch

    Args:
        epoch: 当前epoch索引（从0开始）
        cfg: 训练配置字典
        model: 语义分割模型
        val_loader: 验证集DataLoader
        loss_fn: 损失函数

    Returns:
        (epoch平均loss, 指标字典)
    """
    model.eval()

    val_loss = 0.0
    samples = 0
    meter = SegmentationMetricMeter(
        num_classes=cfg["num_classes"],
        ignore_index=cfg.get("ignore_index", 255),
        ignore_classes=cfg.get("metric_ignore_classes", []),
    )

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']} [验证]")
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
            loss = loss_fn(outputs, labels)

            meter.update(outputs, labels)

            val_loss += loss.item() * bs
            val_bar.set_postfix(loss=f"{val_loss / max(samples, 1):.4f}")

    metrics = meter.compute()
    epoch_loss = val_loss / max(samples, 1)
    return epoch_loss, metrics


def fit_one_epoch(epoch, cfg, model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler, writer, state=None):
    """执行一个完整的训练+验证epoch

    Args:
        epoch: 当前epoch索引（从0开始）
        cfg: 训练配置字典
        model: 语义分割模型
        train_loader: 训练集DataLoader
        val_loader: 验证集DataLoader
        loss_fn: 损失函数
        optimizer: 优化器
        lr_scheduler: 学习率调度器，None表示不调度
        writer: TensorBoard writer
        state: 上一轮的Checkpoint状态，首次调用传None

    Returns:
        (epoch指标字典, 更新后的Checkpoint状态)
    """
    if state is None:
        state = Checkpoint()

    start_time = time.time()

    # 训练
    train_loss, train_metrics = fit_train_epoch(
        epoch, cfg, model, train_loader, loss_fn, optimizer, writer
    )
    # 验证
    val_loss, val_metrics = fit_val_epoch(
        epoch, cfg, model, val_loader, loss_fn
    )

    # 更新学习率
    lr = optimizer.param_groups[0]["lr"]
    if lr_scheduler is not None:
        lr_scheduler.step()

    # 更新状态
    state.train_miou = train_metrics["miou"]
    state.train_pixel_acc = train_metrics["pixel_acc"]
    state.val_miou = val_metrics["miou"]
    state.val_pixel_acc = val_metrics["pixel_acc"]

    # 汇总本epoch所有指标
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
