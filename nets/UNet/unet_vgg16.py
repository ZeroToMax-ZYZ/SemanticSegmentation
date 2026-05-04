import os

import torch
import torch.nn as nn

# 预训练权重默认路径：项目根目录/pre_weights/vgg16_encoder.pth
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_WEIGHTS_PATH = os.path.join(_PROJECT_ROOT, "pre_weights", "vgg16_encoder.pth")


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        # self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        feat1 = self.features[  :4 ](x)
        feat2 = self.features[4 :9 ](feat1)
        feat3 = self.features[9 :16](feat2)
        feat4 = self.features[16:23](feat3)
        feat5 = self.features[23:-1](feat4)
        return [feat1, feat2, feat3, feat4, feat5]


def make_layers(cfg, batch_norm=False, in_channels = 3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
# 512,512,3 -> 512,512,64 -> 256,256,64 -> 256,256,128 -> 128,128,128 -> 128,128,256 -> 64,64,256
# 64,64,512 -> 32,32,512 -> 32,32,512
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def VGG16(in_channels, pretrained, weights_path=None, **kwargs):
    """构建 VGG16 编码器，可选加载 ImageNet 预训练权重。

    Args:
        in_channels: 输入通道数
        pretrained: 是否加载预训练权重
        weights_path: 权重文件路径，None 时使用默认路径 pre_weights/vgg16_encoder.pth

    Returns:
        VGG 模型（仅 features 部分）
    """
    model = VGG(make_layers(cfgs["D"], batch_norm=False, in_channels=in_channels), **kwargs)
    del model.avgpool
    del model.classifier

    if pretrained:
        path = weights_path or _DEFAULT_WEIGHTS_PATH
        if not os.path.exists(path):
            print(f"[信息] 未检测到预训练权重: {path}")
            print("[信息] 正在自动下载，请稍候...")
            from pre_weights.script.download_vgg16 import download_vgg16_encoder
            download_vgg16_encoder(save_path=path)

        if os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            # 兼容 in_channels != 3 的情况：仅加载 shape 匹配的层
            model_state = model.features.state_dict()
            matched = {}
            skipped = []
            for k, v in state_dict.items():
                if k in model_state and v.shape == model_state[k].shape:
                    matched[k] = v
                else:
                    skipped.append(k)
            model.features.load_state_dict(matched, strict=False)
            print(f"[信息] 成功加载预训练权重: {path}")
            print(f"[信息] 已加载 {len(matched)}/{len(state_dict)} 层参数")
            if skipped:
                print(f"[信息] 跳过 {len(skipped)} 层（shape 不匹配或 in_channels 不同）")

    return model


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class UNet_vgg16(nn.Module):
    def __init__(self, in_channels=3, out_channels=21, pretrained=True, backbone='vgg', pretrained_weights_path=None):
        super().__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(in_channels, pretrained=pretrained, weights_path=pretrained_weights_path)
            in_filters  = [192, 384, 768, 1024]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        # vgg的输出层厚度
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        self.final = nn.Conv2d(out_filters[0], out_channels, 1)

        self.backbone = backbone

    def forward(self, inputs):
       
        [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)
        
        return final


if __name__ == '__main__':
    # pretrained=True: 加载预训练权重（自动下载）
    # pretrained=False: 跳过预训练，随机初始化
    model = UNet_vgg16(in_channels=3, out_channels=21, pretrained=True)
    test_input = torch.randn(1, 3, 448, 448)
    test_output = model(test_input)
    print(test_output.shape)