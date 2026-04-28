# use alexnet to semantic segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F
# trans conv --> 1*1 conv
class AlexNet(nn.Module):
    """A lightweight AlexNet-style FCN segmentation model."""

    def __init__(self, in_channels=3, out_channels=21):
        """Initialize convolutional encoder and upsampling classifier.

        Args:
            in_channels (int): Number of input image channels.
            out_channels (int): Number of output classes.

        Returns:
            None.
        """
        super().__init__() 
        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=11,stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1) # 13 x 13 x 256
        # transpose convolution
        # 227 * 227 * 256
        self.deconv = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=256,
            kernel_size=36,
            stride=16,
            padding=1,
            output_padding=1,
            bias=False
        )
        
        # 1*1 conv
        self.classifier = nn.Conv2d(256, out_channels, kernel_size=1 )
        
    def forward(self, data):
        """Run a forward pass.

        Args:
            data (Tensor): Input image batch with shape ``[B, C, H, W]``.

        Returns:
            Tensor: Segmentation logits.
        """
        x = F.relu(self.conv1(data))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        # print('before deconv:', x.shape)
        x = self.deconv(x)
        # print('after deconv:', x.shape)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    model = AlexNet(in_channels=3, out_channels=2)
    test_tensor = torch.rand(1, 3, 227, 227)
    output = model(test_tensor)
    print(output.shape) # bs x num_classes x 227 x 227
                
