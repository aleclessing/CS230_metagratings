from unet import *
from cont_decoder import *

class JNet(nn.Module):

    def __init__(self, im_dim=(64, 256), static_channels=1, dynamic_channels=2):
        super().__init__()

        self.static_channles = static_channels
        self.dynamic_channels = dynamic_channels
        self.in_channels = static_channels + dynamic_channels
        self.im_dim = im_dim

        self.unet = UNet(self.in_channels, 32)
        self.cont_decoder = ContDecoder(32, output_channels=dynamic_channels)
        
        
    def forward(self, x):
        x_orig = x
        x = self.unet(x)

        x = torch.cat([x, x_orig], dim=1)
        

        return x

        
    