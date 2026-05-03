import torch.nn as nn


def build_loss_fn(cfg):
    """根据配置构建CrossEntropyLoss

    Args:
        cfg: 训练配置字典

    Returns:
        nn.CrossEntropyLoss实例
    """
    return nn.CrossEntropyLoss(ignore_index=cfg.get("ignore_index", 255))
