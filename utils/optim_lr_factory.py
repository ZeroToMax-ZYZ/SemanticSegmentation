import torch.optim as optim

from utils.loss import build_loss_fn


def build_optimizer(model, cfg):
    opt_cfg = cfg["optimizer"]
    optimizer_type = opt_cfg.get("type", "SGD")
    lr = opt_cfg.get("lr", 1e-2)
    weight_decay = opt_cfg.get("weight_decay", 5e-4)

    if optimizer_type == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
            nesterov=opt_cfg.get("nesterov", True),
        )

    if optimizer_type == "Adam":
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_type == "AdamW":
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def build_lr_scheduler(optimizer, cfg):
    sch_cfg = cfg["optimizer"].get("lr_scheduler")
    if not sch_cfg:
        return None

    scheduler_type = sch_cfg.get("type", "PolyLR")

    if scheduler_type == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sch_cfg["step_size"],
            gamma=sch_cfg.get("gamma", 0.1),
        )

    if scheduler_type == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sch_cfg.get("T_max", cfg["epochs"]),
            eta_min=sch_cfg.get("eta_min", 0.0),
        )

    if scheduler_type in {"PolyLR", "WarmupPolyLR"}:
        max_epochs = max(int(sch_cfg.get("max_epochs", cfg["epochs"])), 1)
        power = float(sch_cfg.get("power", 0.9))
        min_lr = float(sch_cfg.get("min_lr", 0.0))
        base_lr = float(cfg["optimizer"].get("lr", optimizer.param_groups[0]["lr"]))
        warmup_epochs = int(sch_cfg.get("warmup_epochs", 0))
        warmup_start_factor = float(sch_cfg.get("warmup_start_factor", 0.1))
        min_factor = min_lr / base_lr if base_lr > 0 else 0.0

        def lr_lambda(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                progress = float(epoch + 1) / float(warmup_epochs)
                return warmup_start_factor + (1.0 - warmup_start_factor) * progress

            poly_epoch = min(max(epoch - warmup_epochs, 0), max_epochs)
            poly_total = max(max_epochs - warmup_epochs, 1)
            factor = (1.0 - float(poly_epoch) / float(poly_total)) ** power
            return max(factor, min_factor)

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unsupported lr_scheduler type: {scheduler_type}")
