from nets.base_FCN.alexnet_fcn import AlexNet
from nets.base_FCN.lenet_fcn import LeNet
from nets.base_FCN.vgg_fcn import MyVgg
from nets.UNet.unet_vgg16 import UNet_vgg16


MODEL_REGISTRY = {
    "LeNet": LeNet,
    "AlexNet": AlexNet,
    "VGGFCN": MyVgg,
    "UNet": UNet_vgg16,
}


def auto_model(model_name, in_channels=3, out_channels=32, **kwargs):
    """通过注册表名称实例化分割模型。

    Args:
        model_name: MODEL_REGISTRY 中的模型名称
        in_channels: 输入图像通道数
        out_channels: 输出类别数
        **kwargs: 传递给模型构造函数的额外参数（如 pretrained）

    Returns:
        nn.Module: 实例化的分割模型
    """
    model_class = MODEL_REGISTRY.get(model_name)
    if model_class is None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model_name={model_name}. Available: {available}")

    if model_class is MyVgg:
        model = model_class(nums_output=out_channels)
    else:
        model = model_class(in_channels=in_channels, out_channels=out_channels, **kwargs)

    print(f"[信息] 构建模型 {model_name} 成功")
    return model


def build_model(cfg):
    """根据训练配置构建模型。

    Args:
        cfg: 训练配置字典，包含 model_name, in_channels, num_classes 等

    Returns:
        nn.Module: 配置好的分割模型
    """
    extra_kwargs = {}
    if "pretrained" in cfg:
        extra_kwargs["pretrained"] = cfg["pretrained"]
    if "pretrained_weights_path" in cfg:
        extra_kwargs["pretrained_weights_path"] = cfg["pretrained_weights_path"]

    return auto_model(
        model_name=cfg.get("model_name", "LeNet"),
        in_channels=cfg.get("in_channels", 3),
        out_channels=cfg["num_classes"],
        **extra_kwargs,
    )


if __name__ == "__main__":
    model = auto_model("LeNet", in_channels=3, out_channels=32)
    print(model.__class__.__name__)
