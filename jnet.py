from unet import *

class TCAJNet(nn.Module):

    def __init__(self, static_channels=1, dynamic_channels=2, upsampling_layers=1):
        super().__init__()

        self.static_channles = static_channels
        self.dynamic_channels = dynamic_channels
        self.in_channels = static_channels + dynamic_channels
        self.upsampling_layers = upsampling_layers

        self.unet = UNet(self.in_channels, 16)

        self.upsamp_convs = []
        self.upsamp_resblocks = []

        for i in range(upsampling_layers):
            self.upsamp_convs.append(nn.ConvTranspose2d(self.in_channels + 16, 16, kernel_size=2, stride=2))
            self.upsamp_resblocks.append(ResBlock(16, 16))
    
        self.final_layer = nn.Conv2d(self.in_channels + 16, self.dynamic_channels, kernel_size=1)

    def forward(self, x):
        x_orig = x
        x = self.unet(x)

        for i in range(self.upsampling_layers):

            if i > 0:
                print(x_orig.shape)
                upscaled_x_orig = torch.nn.Upsample(scale_factor=2**i, mode='bilinear')(x_orig)
                print(upscaled_x_orig.shape)
            else:
                upscaled_x_orig = x_orig

            x = torch.cat([x, upscaled_x_orig], dim=1)

            x = self.upsamp_convs[i](x)
            x = self.upsamp_resblocks[i](x)

        upscaled_x_orig = torch.nn.Upsample(scale_factor=2**self.upsampling_layers, mode='bilinear')(x_orig)
        print(x.shape, upscaled_x_orig.shape)
        x = torch.cat([x, upscaled_x_orig], dim=1)

        x = self.final_layer(x)

        return x

        
    