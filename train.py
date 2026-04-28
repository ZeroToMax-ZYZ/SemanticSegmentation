import argparse
import json
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


DEFAULT_CFG_PATH = Path("train_cfg") / "CamVid" / "cfg.yaml"


def seed_everything(seed):
    """Set the random seed for Python, NumPy, and PyTorch.

    Args:
        seed (int): Seed value used by all supported random generators.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deep_update(base, updates):
    """Recursively merge an override dictionary into a base dictionary.

    Args:
        base (dict): Original config dictionary.
        updates (dict): Values that should override the base config.

    Returns:
        dict: A deep-copied merged config. The input dictionaries are not mutated.
    """
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_cfg_from_file(cfg_path):
    """Load a training config from a YAML or JSON file.

    Args:
        cfg_path (str | Path): Path to the config file.

    Returns:
        dict: Parsed config dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file extension is unsupported.
        TypeError: If the parsed config is not a dictionary.
    """
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    suffix = cfg_path.suffix.lower()
    with cfg_path.open("r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            cfg = yaml.safe_load(f)
        elif suffix == ".json":
            cfg = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {cfg_path}. Use .yaml, .yml, or .json")

    if cfg is None:
        cfg = {}

    if not isinstance(cfg, dict):
        raise TypeError(f"Config from {cfg_path} must be a dict")
    return deepcopy(cfg)


def parse_override_value(value):
    """Parse a command-line override value with YAML semantics.

    Args:
        value (str): Raw string value from ``--opts``.

    Returns:
        Any: Parsed value, such as int, float, bool, list, dict, None, or str.
    """
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def set_by_dotted_key(cfg, dotted_key, value):
    """Set a nested dictionary value using a dotted key.

    Args:
        cfg (dict): Dictionary to mutate.
        dotted_key (str): Key path such as ``optimizer.lr``.
        value (Any): Value to set at the target key.

    Returns:
        None.
    """
    keys = dotted_key.split(".")
    if any(not key for key in keys):
        raise ValueError(f"Invalid override key: {dotted_key}")

    current = cfg
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def parse_cli_overrides(opts):
    """Convert command-line ``--opts`` items into a nested override dict.

    Args:
        opts (list[str] | None): Items in ``key=value`` format.

    Returns:
        dict: Nested overrides that can be merged into the config.
    """
    overrides = {}
    for item in opts or []:
        if "=" not in item:
            raise ValueError(f"Invalid --opts item: {item}. Expected key=value")
        key, value = item.split("=", 1)
        set_by_dotted_key(overrides, key, parse_override_value(value))
    return overrides


def parse_args():
    """Parse training command-line arguments.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed arguments with cfg, opts, and smoke_test.
    """
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument(
        "--cfg",
        default=os.getenv("TRAIN_CFG", str(DEFAULT_CFG_PATH)),
        help="Path to a YAML/JSON training config.",
    )
    parser.add_argument(
        "--opts",
        nargs="*",
        default=None,
        help="Override config values with dotted keys, e.g. --opts epochs=1 optimizer.lr=0.001",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny one-epoch training pass to verify the pipeline.",
    )
    return parser.parse_args()


def manual_overrides():
    """Return local, source-controlled manual overrides.

    Args:
        None.

    Returns:
        dict: Optional overrides. Keep this empty for normal training and use
        ``--opts`` for temporary command-line changes.
    """
    return {
        # Examples:
        # "epochs": 1,
        # "batch_size": 2,
        # "debug_mode": "debug",
        # "optimizer": {"lr": 1e-3},
    }


def finalize_runtime_config(cfg):
    """Fill runtime-dependent config fields.

    Args:
        cfg (dict): Config loaded from file and overrides.

    Returns:
        dict: Config with exp_time, device, GPU_model, pin_memory, and exp_name
        resolved.
    """
    cfg = deepcopy(cfg)
    exp_time = cfg.get("exp_time") or time.strftime("%Y%m%d-%H%M%S", time.localtime())
    cfg["exp_time"] = exp_time

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device

    cfg["GPU_model"] = torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else "CPU"

    if cfg.get("pin_memory", "auto") == "auto":
        cfg["pin_memory"] = device == "cuda"

    exp_name = cfg.get("exp_name", "semantic_segmentation")
    if "{exp_time}" in exp_name:
        cfg["exp_name"] = exp_name.format(exp_time=exp_time)
    elif cfg.get("append_exp_time", True):
        cfg["exp_name"] = f"{exp_name}_{exp_time}"

    return cfg


def base_config(cfg_path=None, overrides=None):
    """Build the final training config from file and overrides.

    Args:
        cfg_path (str | Path | None): Config path. Defaults to TRAIN_CFG env var
            or DEFAULT_CFG_PATH.
        overrides (dict | None): Programmatic overrides, usually parsed from CLI.

    Returns:
        dict: Fully resolved training config.
    """
    cfg_path = cfg_path or os.getenv("TRAIN_CFG", str(DEFAULT_CFG_PATH))
    cfg = load_cfg_from_file(cfg_path)
    cfg = deep_update(cfg, manual_overrides())
    if overrides:
        cfg = deep_update(cfg, overrides)
    cfg = finalize_runtime_config(cfg)
    cfg["cfg_path"] = str(cfg_path)
    return cfg


def apply_runtime_overrides(cfg, smoke_test=False):
    """Apply quick runtime overrides that should not live in the config file.

    Args:
        cfg (dict): Config dictionary to mutate.
        smoke_test (bool): Whether to force a tiny one-epoch smoke test run.

    Returns:
        None.
    """
    if smoke_test or os.getenv("SEMSEG_SMOKE_TEST", "0") == "1":
        cfg["debug_mode"] = "smoke"
        cfg["epochs"] = 1
        cfg["batch_size"] = 2
        cfg["num_workers"] = 0
        cfg["persistent_workers"] = False
        cfg["pin_memory"] = False
        cfg["debug_max_train_samples"] = 4
        cfg["debug_max_val_samples"] = 2
        cfg["exp_name"] = f"{cfg['exp_name']}_smoke"


def _unwrap_dataset(dataset):
    """Get the underlying dataset from nested torch Subset wrappers.

    Args:
        dataset (Dataset): Dataset or wrapper dataset.

    Returns:
        Dataset: The innermost dataset.
    """
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def train(args=None):
    """Run the full training workflow.

    Args:
        args (argparse.Namespace | None): Parsed command-line args. If None,
            arguments are parsed inside the function.

    Returns:
        None.
    """
    args = args or parse_args()
    cfg = base_config(cfg_path=args.cfg, overrides=parse_cli_overrides(args.opts))
    apply_runtime_overrides(cfg, smoke_test=args.smoke_test)
    seed_everything(cfg["seed"])

    if cfg["device"] == "cuda":
        torch.backends.cudnn.benchmark = True

    # Build data before the model so the class count can be read from class_dict.csv.
    train_loader, val_loader, train_dataset, val_dataset = build_dataset(cfg)
    real_dataset = _unwrap_dataset(train_dataset)
    cfg["num_classes"] = getattr(real_dataset, "num_classes", cfg["num_classes"])

    save_config(cfg)
    tb_path = os.path.join("logs", "logs_tensorboard", cfg["exp_name"])
    writer = SummaryWriter(log_dir=tb_path)
    init_tb_layout(writer)

    # Build all training components from the resolved config.
    model = build_model(cfg).to(cfg["device"])
    optimizer = build_optimizer(model, cfg=cfg)
    lr_scheduler = build_lr_scheduler(optimizer, cfg=cfg)
    loss_fn = build_loss_fn(cfg)

    base_tensorb_logger(writer, train_dataset, val_dataset, model, cfg)

    state = None
    best_val_miou = 0.0
    for epoch in range(cfg["epochs"]):
        # One epoch returns scalar metrics and a lightweight in-memory state.
        metrics, state = fit_one_epoch(
            epoch,
            cfg,
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            lr_scheduler,
            writer,
            state,
        )

        if metrics["val_miou"] > best_val_miou:
            # Mark the current epoch as best before save_logger writes weights.
            best_val_miou = metrics["val_miou"]
            state.best_val_miou = best_val_miou
            metrics["is_best"] = True

        save_logger(model, metrics, cfg, state)
        epoch_tensorb_logger(writer, metrics, epoch)

    flat_cfg = flatten_config(cfg)
    writer.add_hparams(
        hparam_dict=flat_cfg,
        metric_dict={"val_miou": best_val_miou},
        run_name=".",
    )
    writer.close()


if __name__ == "__main__":
    train()
