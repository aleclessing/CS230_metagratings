from unet import *

class JNet(nn.Module):

    def __init__(self, im_dim=(64, 256), static_channels=1, dynamic_channels=2):
        super().__init__()

        self.static_channles = static_channels
        self.dynamic_channels = dynamic_channels
        self.in_channels = static_channels + dynamic_channels
        self.im_dim = im_dim

        self.unet = UNet(self.in_channels)
        self.lopside = Up(self.in_channels, self.dynamic_channels)

    def forward(self, x):
        x = self.unet(x)
        x = self.lopside(x)
        return x

        
    