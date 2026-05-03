import json
import os

import torch
from matplotlib import pyplot as plt


# CSV中每行记录的字段顺序
CSV_FIELDS = [
    "epoch",
    "train_loss",
    "train_miou",
    "train_pixel_acc",
    "train_mean_acc",
    "val_loss",
    "val_miou",
    "val_pixel_acc",
    "val_mean_acc",
    "lr",
    "epoch_time",
]


def save_csv(metrics, csv_path):
    """将一个epoch的指标追加写入CSV文件

    Args:
        metrics: fit_one_epoch返回的指标字典
        csv_path: CSV文件路径
    """
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(CSV_FIELDS) + "\n")

    with open(csv_path, "a", encoding="utf-8") as f:
        values = []
        for field in CSV_FIELDS:
            data = metrics[field]
            if isinstance(data, int):
                values.append(str(data))
            elif isinstance(data, float):
                values.append(str(data) if field == "lr" else f"{data:.5f}")
            else:
                values.append(str(data))
        f.write(",".join(values) + "\n")


def plot_metrics(cfg, csv_path, plt_path):
    """从CSV读取指标，绘制loss和分割指标曲线并保存

    Args:
        cfg: 训练配置（预留扩展）
        csv_path: 指标CSV路径
        plt_path: 图片保存路径
    """
    plt.rcParams["font.family"] = "serif"

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        header = next(f).strip().split(",")
        for line in f:
            if line.strip():
                rows.append(dict(zip(header, line.strip().split(","))))

    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows]
    train_losses = [float(row["train_loss"]) for row in rows]
    val_losses = [float(row["val_loss"]) for row in rows]
    train_miou = [float(row["train_miou"]) for row in rows]
    val_miou = [float(row["val_miou"]) for row in rows]
    train_pixel_acc = [float(row["train_pixel_acc"]) for row in rows]
    val_pixel_acc = [float(row["val_pixel_acc"]) for row in rows]

    plt.figure(figsize=(14, 6))

    # 左图：loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker="x", markersize=4, linewidth=1)
    plt.plot(epochs, val_losses, label="Val Loss", marker="x", markersize=4, linewidth=1)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # 右图：mIoU和Pixel Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_miou, label="Train mIoU", marker="x", markersize=4, linewidth=1)
    plt.plot(epochs, val_miou, label="Val mIoU", marker="x", markersize=4, linewidth=1)
    plt.plot(epochs, train_pixel_acc, label="Train Pixel Acc", marker="x", markersize=4, linestyle="--")
    plt.plot(epochs, val_pixel_acc, label="Val Pixel Acc", marker="x", markersize=4, linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title(f"Segmentation Metrics (Best Val mIoU: {max(val_miou):.4f})")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plt_path, dpi=300)
    plt.close()


def save_logger(model, metrics, cfg, state):
    """保存每个epoch的日志产物：CSV指标、曲线图、模型权重

    Args:
        model: 模型
        metrics: 本epoch指标字典
        cfg: 训练配置
        state: Checkpoint状态
    """
    base_logs_path = os.path.join("logs", "logs_upload", cfg["exp_name"])
    base_weights_path = os.path.join("logs", "logs_weights", cfg["exp_name"])

    csv_path = os.path.join(base_logs_path, "metrics.csv")
    plt_path = os.path.join(base_logs_path, "metrics.png")
    model_path = os.path.join(base_weights_path, "weights")

    save_csv(metrics, csv_path)
    plot_metrics(cfg, csv_path, plt_path)
    save_model(model, cfg, model_path, metrics)


def save_model(model, cfg, model_path, metrics):
    """保存模型权重：最优模型、定期保存、最新模型

    Args:
        model: 模型
        cfg: 训练配置，包含save_interval
        model_path: 权重保存目录
        metrics: 指标字典，包含is_best和val_miou
    """
    val_miou = metrics["val_miou"]

    # 保存最优模型
    if metrics["is_best"]:
        torch.save(model.state_dict(), os.path.join(model_path, "best_model.pth"))
        print(f"Best model saved with Val mIoU: {val_miou:.4f}")

    # 按间隔定期保存
    if (metrics["epoch"] % cfg["save_interval"]) == 0:
        torch.save(
            model.state_dict(),
            os.path.join(model_path, f"model_epoch_{metrics['epoch']}_valmiou_{val_miou:.4f}.pth"),
        )

    # 始终保存最新模型
    torch.save(model.state_dict(), os.path.join(model_path, "last_model.pth"))


def save_config(cfg):
    """持久化训练配置，并创建日志目录

    Args:
        cfg: 训练配置字典
    """
    base_logs_path = os.path.join("logs", "logs_upload", cfg["exp_name"])
    base_weights_path = os.path.join("logs", "logs_weights", cfg["exp_name"], "weights")

    os.makedirs(base_logs_path, exist_ok=True)
    os.makedirs(base_weights_path, exist_ok=True)

    config_path = os.path.join(base_logs_path, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
