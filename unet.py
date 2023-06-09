
from unet_parts import *

class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, prune_depth=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prune_depth = prune_depth

        self.input_layer = ResBlock(self.in_channels, 16) #image dimension, input channels, output channels
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        if not prune_depth:
            self.down3 = Down(64, 128)

            self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.up3 = Up(32, out_channels)


    def forward(self, x):

        x1 = self.input_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        if not self.prune_depth:
            x4 = self.down3(x3)
            x = self.up1(x4, x3)
        else:
            x = x3
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return x
        