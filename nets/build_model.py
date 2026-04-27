from nets.base_FCN.lenet_fcn import LeNet
from rich import print

# 创建字符串到类的映射字典
unet_mapping = {
    "LeNet": LeNet,
}

def auto_model(model_name, in_channels=3, out_channels=2):
    """
    根据输入的模型名称字符串，返回对应的模型类
    参数:
        model_name: 模型名称字符串

    返回:
        对应的模型类，如果不存在则返回None或抛出异常
    """
    # 尝试从映射字典中获取模型类
    model_class = unet_mapping.get(model_name)
    
    if model_class is None:
        # 可以选择抛出异常或返回None
        raise ValueError(f"未找到名为'{model_name}'的模型，请检查名称是否正确")
    model = model_class(in_channels, out_channels)
    print(f"[green]\[info][/green] build model {model_name} successfully")
    return model

# 使用示例
if __name__ == "__main__":
    # 输入字符串名称
    model_name = "LeNet"
    # 获取对应的模型类
    model_class = auto_model(model_name, in_channels=3, out_channels=2)
    print(f"成功创建模型: {model_name}")

