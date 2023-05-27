from unet import *
from cont_decoder import *

class ContJNet(nn.Module):

    def __init__(self, static_channels=1, dynamic_channels=2):
        super().__init__()

        self.static_channles = static_channels
        self.dynamic_channels = dynamic_channels
        self.in_channels = static_channels + dynamic_channels

        self.unet = UNet(self.in_channels, 32)
        self.cont_decoder = ContDecoder(32, output_channels=dynamic_channels)
        
        
    def forward(self, lr_grid, coords):

        context_grid = self.unet(lr_grid)
        
        x = self.cont_decoder(context_grid, coords)

        

        return x

        
    