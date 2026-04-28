import torch
import torch.nn as nn

class LeNet(nn.Module):
    """A small FCN-style segmentation model based on LeNet/AlexNet blocks."""

    def __init__(self, in_channels=3, out_channels=2):
        """Initialize encoder and transpose-convolution decoder.

        Args:
            in_channels (int): Number of input image channels.
            out_channels (int): Number of output classes.

        Returns:
            None.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Conv2d(96, 256, kernel_size=5, padding=2), # 27*27*256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 13*13*256
            nn.Conv2d(256, 384, kernel_size=3, padding=1), # 13*13*384
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), # 13*13*384
            nn.ReLU(),  
        )
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(384, 128, kernel_size=4, stride=4, padding=0, output_padding=3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, out_channels, kernel_size=11, stride=4, padding=0, output_padding=0),
        )

    def forward(self, x):
        """Run a forward pass.

        Args:
            x (Tensor): Input image batch with shape ``[B, C, H, W]``.

        Returns:
            Tensor: Segmentation logits with shape ``[B, out_channels, H, W]``
            for the configured 227x227 baseline input.
        """
        out = self.encoder(x)
        out = self.deconv(out)
        return out
    
    
if __name__ == '__main__':
    test_data = torch.randn(1, 3, 227, 227)
    model = LeNet()
    result = model(test_data)
    print(result.shape)
