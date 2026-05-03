import argparse
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from dataset.build_dataset import build_dataset
from nets.build_model import build_model
from utils.fit_one_epoch import fit_one_epoch
from utils.logger import save_config, save_logger
from utils.logger_tensorb import base_tensorb_logger, epoch_tensorb_logger, flatten_config, init_tb_layout
from utils.loss import build_loss_fn
from utils.optim_lr_factory import build_lr_scheduler, build_optimizer


def seed_everything(seed):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deep_update(base, updates):
    """递归合并字典，updates中的值覆盖base"""
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path):
    """从yaml文件加载配置"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return deepcopy(cfg) if cfg else {}


def build_cfg(yaml_path):
    """构建最终训练配置：yaml基础 + 代码覆盖 + 运行时字段

    Args:
        yaml_path: yaml配置文件路径

    Returns:
        完整的训练配置字典
    """
    # 1. 从yaml文件加载基础配置
    cfg = load_yaml(yaml_path)

    # 2. 代码中的覆盖字典（方便调试时快速修改）
    code_overrides = {
        # "epochs": 1,
        # "batch_size": 4,
        # "debug_mode": "debug",
    }
    cfg = deep_update(cfg, code_overrides)

    # 3. 填充运行时字段
    exp_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    cfg["exp_time"] = exp_time

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device
    cfg["GPU_model"] = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

    if cfg.get("pin_memory", "auto") == "auto":
        cfg["pin_memory"] = (device == "cuda")

    # 实验名拼接时间戳
    exp_name = cfg.get("exp_name", "segmentation")
    if cfg.get("append_exp_time", True):
        cfg["exp_name"] = f"{exp_name}_{exp_time}"

    cfg["cfg_path"] = str(yaml_path)

    return cfg


def train():
    # 命令行只接收yaml路径
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="train_cfg/CamVid/LeNet_cfg.yaml",
                        help="yaml配置文件路径")
    args = parser.parse_args()

    # 构建配置
    cfg = build_cfg(args.cfg)
    seed_everything(cfg["seed"])

    if cfg["device"] == "cuda":
        torch.backends.cudnn.benchmark = True

    # 构建数据集和DataLoader
    dataloaders, _ = build_dataset(cfg)

    # 保存配置 + 初始化TensorBoard
    save_config(cfg)
    writer = SummaryWriter(log_dir=os.path.join("logs", "logs_tensorboard", cfg["exp_name"]))
    init_tb_layout(writer)

    # 构建模型、优化器、调度器、损失函数
    model = build_model(cfg).to(cfg["device"])
    optimizer = build_optimizer(model, cfg=cfg)
    lr_scheduler = build_lr_scheduler(optimizer, cfg=cfg)
    loss_fn = build_loss_fn(cfg)

    # 记录初始可视化
    base_tensorb_logger(writer, dataloaders["train"].dataset, dataloaders["val"].dataset, model, cfg)

    # 训练循环
    state = None
    best_val_miou = 0.0
    for epoch in range(cfg["epochs"]):
        metrics, state = fit_one_epoch(
            epoch, cfg, model,
            dataloaders["train"], dataloaders["val"],
            loss_fn, optimizer, lr_scheduler, writer, state,
        )

        if metrics["val_miou"] > best_val_miou:
            best_val_miou = metrics["val_miou"]
            state.best_val_miou = best_val_miou
            metrics["is_best"] = True

        save_logger(model, metrics, cfg, state)
        epoch_tensorb_logger(writer, metrics, epoch)

    # 记录超参数和最终指标
    writer.add_hparams(
        hparam_dict=flatten_config(cfg),
        metric_dict={"val_miou": best_val_miou},
        run_name=".",
    )
    writer.close()


if __name__ == "__main__":
    train()
