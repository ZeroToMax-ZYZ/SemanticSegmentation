import json

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def denormalize(tensor, mean=None, std=None):
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
    flat_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool, torch.Tensor)):
            flat_cfg[key] = value
        else:
            flat_cfg[key] = str(value)
    return flat_cfg


def _unwrap_dataset(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _palette_from_dataset(dataset, device):
    real_dataset = _unwrap_dataset(dataset)
    rgb_values = getattr(real_dataset, "rgb_values", None)
    if rgb_values is None:
        num_classes = getattr(real_dataset, "num_classes", 1)
        rgb_values = [(int(i * 53) % 256, int(i * 97) % 256, int(i * 193) % 256) for i in range(num_classes)]
    return torch.tensor(rgb_values, dtype=torch.float32, device=device) / 255.0


def _mask_to_color(mask, palette, ignore_index=255):
    mask = mask.to(torch.long)
    safe_mask = mask.clamp(0, palette.shape[0] - 1)
    color = palette[safe_mask].permute(2, 0, 1)
    ignore = mask == ignore_index
    if ignore.any():
        color[:, ignore] = 0.0
    return color


def _sample_overlay_grid(dataset, count, cfg):
    palette = _palette_from_dataset(dataset, device=torch.device("cpu"))
    ignore_index = cfg.get("ignore_index", 255)
    images = []

    for idx in range(min(count, len(dataset))):
        image, mask = dataset[idx]
        image = denormalize(image).clamp(0, 1).cpu()
        color_mask = _mask_to_color(mask.cpu(), palette, ignore_index=ignore_index)
        overlay = (0.6 * image + 0.4 * color_mask).clamp(0, 1)
        images.extend([image, color_mask, overlay])

    if not images:
        return None
    return make_grid(torch.stack(images), nrow=3, padding=2)


def base_tensorb_logger(writer, train_dataset, val_dataset, model, cfg, train_img_count=5, val_img_count=5, epoch=0):
    image, _ = train_dataset[0]
    graph_input = image.unsqueeze(0).to(cfg["device"])
    try:
        writer.add_graph(model, graph_input)
    except Exception as exc:
        writer.add_text("Graph/warning", f"add_graph failed: {exc}", epoch)

    train_grid = _sample_overlay_grid(train_dataset, train_img_count, cfg)
    if train_grid is not None:
        writer.add_image("Train/Image_Mask_Overlay", train_grid, global_step=epoch)

    val_grid = _sample_overlay_grid(val_dataset, val_img_count, cfg)
    if val_grid is not None:
        writer.add_image("Val/Image_Mask_Overlay", val_grid, global_step=epoch)

    cfg_json_str = json.dumps(flatten_config(cfg), indent=4, ensure_ascii=False)
    writer.add_text("Config/Hyperparameters", f"```json\n{cfg_json_str}\n```", epoch)


def init_tb_layout(writer):
    layout = {
        "Segmentation": {
            "Loss": [
                "Multiline",
                ["metrics/train_loss", "metrics/val_loss", "iter_loss/ce_loss", "iter_loss/total_loss"],
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
    for key, value in loss_item.items():
        if torch.is_tensor(value):
            value = value.detach().item()
        writer.add_scalar(f"iter_loss/{key}", float(value), iteration)


def histogram_tensorb_logger(writer, model, last_outputs, epoch, debug_mode=False):
    params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and param.dim() > 1
    ]
    if not debug_mode and len(params) > 3:
        mid = len(params) // 2
        params = [params[0], params[mid], params[-1]]

    for name, param in params:
        writer.add_histogram(f"Weights/{name}", param.data, epoch)
        if param.grad is not None:
            writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

    if last_outputs is not None:
        writer.add_histogram("Predictions/logits", last_outputs.detach(), epoch)
        writer.add_histogram("Predictions/classes", last_outputs.detach().argmax(dim=1), epoch)
