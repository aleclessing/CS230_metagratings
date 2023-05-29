from unet import *

class JNet(nn.Module):

    def __init__(self, im_dim=(64, 256), static_channels=1, dynamic_channels=2):
        super().__init__()

        self.static_channles = static_channels
        self.dynamic_channels = dynamic_channels
        self.in_channels = static_channels + dynamic_channels
        self.im_dim = im_dim

        self.unet = UNet(self.in_channels, 16)
        self.super_up = nn.ConvTranspose2d(self.in_channels + 16, 16, kernel_size=2, stride=2)
        self.super_resblock = ResBlock(16, 16)
        self.final_layer = nn.ConvTranspose2d(16, self.dynamic_channels, kernel_size=1)

    def forward(self, x):
        x_orig = x
        x = self.unet(x)

        x = torch.cat([x, x_orig], dim=1)
        x = self.super_up(x)

        x = self.super_resblock(x)
        x = self.final_layer(x)

        return x

        
    