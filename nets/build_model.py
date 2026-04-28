from nets.base_FCN.alexnet_fcn import AlexNet
from nets.base_FCN.lenet_fcn import LeNet
from nets.base_FCN.vgg_fcn import MyVgg


MODEL_REGISTRY = {
    "LeNet": LeNet,
    "AlexNet": AlexNet,
    "VGGFCN": MyVgg,
}


def auto_model(model_name, in_channels=3, out_channels=32):
    """Instantiate a segmentation model by registry name.

    Args:
        model_name (str): Name in ``MODEL_REGISTRY``.
        in_channels (int): Number of input image channels.
        out_channels (int): Number of output semantic classes.

    Returns:
        nn.Module: Instantiated segmentation model.
    """
    model_class = MODEL_REGISTRY.get(model_name)
    if model_class is None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model_name={model_name}. Available: {available}")

    if model_class is MyVgg:
        model = model_class(nums_output=out_channels)
    else:
        model = model_class(in_channels=in_channels, out_channels=out_channels)

    print(f"[info] build model {model_name} successfully")
    return model


def build_model(cfg):
    """Build the model defined by the training config.

    Args:
        cfg (dict): Resolved training config with ``model_name``,
            ``in_channels``, and ``num_classes``.

    Returns:
        nn.Module: Configured segmentation model.
    """
    return auto_model(
        model_name=cfg.get("model_name", "LeNet"),
        in_channels=cfg.get("in_channels", 3),
        out_channels=cfg["num_classes"],
    )


if __name__ == "__main__":
    model = auto_model("LeNet", in_channels=3, out_channels=32)
    print(model.__class__.__name__)
