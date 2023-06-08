from unet import *
from cont_decoder import *

class ContJNet(nn.Module):

    def __init__(self, static_channels=1, dynamic_channels=2, upsampling_layers=1, context_channels=32):
        super().__init__()

        self.static_channels = static_channels
        self.dynamic_channels = dynamic_channels
        self.upsampling_layers = upsampling_layers
        self.context_channels = context_channels

        self.mlp_in_channels = self.static_channels + self.dynamic_channels + self.context_channels + 2

        self.static_pre_downblocks = []

        for i in range(self.upsampling_layers):
            self.static_pre_downblocks.append(Down(self.static_channels*2**i, min(8, self.static_channels*2**(i+1))))

        self.num_downsamp_static_channels = min(8, self.static_channels*2**self.upsampling_layers)

        #Stop UNet from going down so many levels that it has zero image size
        prune_unet = False
        if upsampling_layers > 3:
            prune_unet = True


        self.unet = UNet(self.dynamic_channels + self.num_downsamp_static_channels, self.context_channels, prune_depth=prune_unet)
        self.cont_decoder = ContDecoder(self.context_channels, output_channels=dynamic_channels)
        
        
    def forward(self, lr_fields, hr_eps, coords):

        eps_encoding = hr_eps.unsqueeze(1)
        for i in range(self.upsampling_layers):
            eps_encoding = self.static_pre_downblocks[i](eps_encoding)

        x = torch.cat([lr_fields, eps_encoding], dim=1)

        context_grid = self.unet(x)
        
        x = self.cont_decoder(lr_fields, context_grid, hr_eps, coords)

        return x

        
    