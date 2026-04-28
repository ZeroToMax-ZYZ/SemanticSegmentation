import cv2
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

class MyVgg(nn.Module):
    """A VGG-style FCN segmentation model with one large deconvolution head."""

    def __init__(self, nums_output):
        """Initialize VGG-like convolution blocks and classifier.

        Args:
            nums_output (int): Number of output semantic classes.

        Returns:
            None.
        """

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(2, 2)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.maxpool5 = nn.MaxPool2d(2, 2) # 7x7x512

        self.deconv = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=64, stride=32, padding=16, output_padding=0, bias=False)# 224x224x512

        self.conv14 = nn.Conv2d(in_channels=512, out_channels=nums_output, kernel_size=1, padding=0)# 224x224x2

    
    def forward(self, data):
        """Run a forward pass.

        Args:
            data (Tensor): Input image batch with shape ``[B, 3, H, W]``.

        Returns:
            Tensor: Segmentation logits.
        """

        conv1 = F.relu(self.conv1(data))
        conv2 = F.relu(self.conv2(conv1))
        maxpool1 = self.maxpool1(conv2)
        conv3 = F.relu(self.conv3(maxpool1))
        conv4 = F.relu(self.conv4(conv3))
        maxpool2 = self.maxpool2(conv4)
        conv5 = F.relu(self.conv5(maxpool2))
        conv6 = F.relu(self.conv6(conv5))
        conv7 = F.relu(self.conv7(conv6))
        maxpool3 = self.maxpool3(conv7)
        conv8 = F.relu(self.conv8(maxpool3))
        conv9 = F.relu(self.conv9(conv8))
        conv10 = F.relu(self.conv10(conv9))
        maxpool4 = self.maxpool4(conv10)
        conv11 = F.relu(self.conv11(maxpool4))
        conv12 = F.relu(self.conv12(conv11))
        conv13 = F.relu(self.conv13(conv12))
        maxpool5 = self.maxpool5(conv13)
        
        deconv = F.relu(self.deconv(maxpool5))
        
        conv14 = self.conv14(deconv)

        return conv14


if __name__ == "__main__":
    my_vgg = MyVgg(2)

    data_try = torch.randn((8, 3, 224, 224))

    print(my_vgg(data_try).shape)
    # 参数量
    params = sum(p.numel() for p in my_vgg.parameters())
    print(params)



