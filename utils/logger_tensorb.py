import json

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def denormalize(tensor, mean=None, std=None):
    """反归一化，用于可视化时还原图片原始像素范围

    Args:
        tensor: 图片tensor，shape [C, H, W] 或 [B, C, H, W]
        mean: 各通道均值，默认ImageNet均值
        std: 各通道标准差，默认ImageNet标准差

    Returns:
        反归一化后的tensor
    """
    if mean is None:
        mean = (0.485, 0.456, 0.406)
    if std is None:
        std = (0.229, 0.224, 0.225)

    if tensor.dim() == 3:
        mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
        std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    else:
        mean = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)
        std = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
    return tensor.mul(std).add(mean)


def flatten_config(cfg):
    """将嵌套配置展平为TensorBoard hparams兼容的格式

    Args:
        cfg: 训练配置字典

    Returns:
        展平后的字典，不支持的类型转为字符串
    """
    flat_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool, torch.Tensor)):
            flat_cfg[key] = value
        else:
            flat_cfg[key] = str(value)
    return flat_cfg


def _unwrap_dataset(dataset):
    """获取最内层的数据集（剥除Subset等包装器）

    Args:
        dataset: Dataset或其包装器

    Returns:
        最内层的Dataset
    """
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _palette_from_dataset(dataset, device):
    """从数据集获取调色板，用于mask可视化着色

    Args:
        dataset: Dataset，期望有rgb_values属性
        device: 返回tensor的设备

    Returns:
        shape [num_classes, 3] 的调色板tensor，值范围[0, 1]
    """
    real_dataset = _unwrap_dataset(dataset)
    rgb_values = getattr(real_dataset, "rgb_values", None)
    if rgb_values is None:
        num_classes = getattr(real_dataset, "num_classes", 1)
        rgb_values = [(int(i * 53) % 256, int(i * 97) % 256, int(i * 193) % 256) for i in range(num_classes)]
    return torch.tensor(rgb_values, dtype=torch.float32, device=device) / 255.0


def _mask_to_color(mask, palette, ignore_index=255):
    """将整数mask转为彩色图片tensor

    Args:
        mask: 整数mask，shape [H, W]
        palette: 类别调色板，shape [C, 3]
        ignore_index: 忽略的标签值，渲染为黑色

    Returns:
        shape [3, H, W] 的彩色mask
    """
    mask = mask.to(torch.long)
    safe_mask = mask.clamp(0, palette.shape[0] - 1)
    color = palette[safe_mask].permute(2, 0, 1)
    ignore = mask == ignore_index
    if ignore.any():
        color[:, ignore] = 0.0
    return color


def _sample_overlay_grid(dataset, count, cfg):
    """采样若干图片，生成 [原图 | mask | 叠加] 的网格图

    Args:
        dataset: 数据集
        count: 采样数量
        cfg: 训练配置，包含ignore_index

    Returns:
        网格图tensor，数据集为空时返回None
    """
    palette = _palette_from_dataset(dataset, device=torch.device("cpu"))
    ignore_index = cfg.get("ignore_index", 255)
    images = []

    for idx in range(min(count, len(dataset))):
        image, mask = dataset[idx]
        image = denormalize(image).clamp(0, 1).cpu()
        color_mask = _mask_to_color(mask.cpu(), palette, ignore_index=ignore_index)
        # 半透明叠加，方便直观检查标注对齐
        overlay = (0.6 * image + 0.4 * color_mask).clamp(0, 1)
        images.extend([image, color_mask, overlay])

    if not images:
        return None
    return make_grid(torch.stack(images), nrow=3, padding=2)


def base_tensorb_logger(writer, train_dataset, val_dataset, model, cfg, train_img_count=5, val_img_count=5, epoch=0):
    """训练开始前写入静态TensorBoard内容：模型图、样本可视化、配置

    Args:
        writer: TensorBoard SummaryWriter
        train_dataset: 训练集
        val_dataset: 验证集
        model: 模型（用于记录计算图）
        cfg: 训练配置
        train_img_count: 训练集可视化样本数
        val_img_count: 验证集可视化样本数
        epoch: 起始step
    """
    # 记录模型计算图
    image, _ = train_dataset[0]
    graph_input = image.unsqueeze(0).to(cfg["device"])
    try:
        writer.add_graph(model, graph_input)
    except Exception as exc:
        writer.add_text("Graph/warning", f"add_graph failed: {exc}", epoch)

    # 记录训练集/验证集样本可视化
    train_grid = _sample_overlay_grid(train_dataset, train_img_count, cfg)
    if train_grid is not None:
        writer.add_image("Train/Image_Mask_Overlay", train_grid, global_step=epoch)

    val_grid = _sample_overlay_grid(val_dataset, val_img_count, cfg)
    if val_grid is not None:
        writer.add_image("Val/Image_Mask_Overlay", val_grid, global_step=epoch)

    # 记录超参数配置
    cfg_json_str = json.dumps(flatten_config(cfg), indent=4, ensure_ascii=False)
    writer.add_text("Config/Hyperparameters", f"```json\n{cfg_json_str}\n```", epoch)


def init_tb_layout(writer):
    """注册自定义TensorBoard标量布局，将相关指标归到同一面板

    Args:
        writer: TensorBoard SummaryWriter
    """
    layout = {
        "Segmentation": {
            "Loss": [
                "Multiline",
                ["metrics/train_loss", "metrics/val_loss", "iter_loss/ce_loss"],
            ],
            "IoU": [
                "Multiline",
                ["metrics/train_miou", "metrics/val_miou"],
            ],
            "Pixel_Accuracy": [
                "Multiline",
                ["metrics/train_pixel_acc", "metrics/val_pixel_acc"],
            ],
        },
    }
    writer.add_custom_scalars(layout)


def epoch_tensorb_logger(writer, metrics, epoch):
    """记录epoch级别的标量指标到TensorBoard

    Args:
        writer: TensorBoard SummaryWriter
        metrics: epoch指标字典
        epoch: 当前epoch索引
    """
    writer.add_scalar("metrics/train_loss", metrics["train_loss"], epoch)
    writer.add_scalar("metrics/val_loss", metrics["val_loss"], epoch)

    writer.add_scalar("metrics/train_miou", metrics["train_miou"], epoch)
    writer.add_scalar("metrics/val_miou", metrics["val_miou"], epoch)

    writer.add_scalar("metrics/train_pixel_acc", metrics["train_pixel_acc"], epoch)
    writer.add_scalar("metrics/val_pixel_acc", metrics["val_pixel_acc"], epoch)

    writer.add_scalar("metrics/train_mean_acc", metrics["train_mean_acc"], epoch)
    writer.add_scalar("metrics/val_mean_acc", metrics["val_mean_acc"], epoch)

    writer.add_scalar("System/learning_rate", metrics["lr"], epoch)
    writer.add_scalar("System/epoch_time", metrics["epoch_time"], epoch)


def iter_tensorb_logger(writer, loss_item, iteration):
    """记录iter级别的loss到TensorBoard

    Args:
        writer: TensorBoard SummaryWriter
        loss_item: loss分量字典，如 {"ce_loss": 0.5}
        iteration: 全局iteration计数
    """
    for key, value in loss_item.items():
        if torch.is_tensor(value):
            value = value.detach().item()
        writer.add_scalar(f"iter_loss/{key}", float(value), iteration)


def histogram_tensorb_logger(writer, model, last_outputs, epoch, debug_mode=False):
    """记录权重、梯度和预测分布的直方图到TensorBoard

    Args:
        writer: TensorBoard SummaryWriter
        model: 模型
        last_outputs: 最后一个batch的logits，用于记录预测分布
        epoch: 当前epoch索引
        debug_mode: True时记录所有层，False时只记录首/中/末三层
    """
    # 筛选需要记录的参数（有梯度且维度>1，即跳过bias和norm的1D参数）
    params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and param.dim() > 1
    ]
    if not debug_mode and len(params) > 3:
        # 正常模式只取首/中/末三个，减小event文件体积
        mid = len(params) // 2
        params = [params[0], params[mid], params[-1]]

    for name, param in params:
        writer.add_histogram(f"Weights/{name}", param.data, epoch)
        if param.grad is not None:
            writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

    # 记录预测logits和类别分布
    if last_outputs is not None:
        writer.add_histogram("Predictions/logits", last_outputs.detach(), epoch)
        writer.add_histogram("Predictions/classes", last_outputs.detach().argmax(dim=1), epoch)
