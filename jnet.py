from unet import *

class TCAJNet(nn.Module):

    def __init__(self, static_channels=1, dynamic_channels=2, upsampling_layers=1):
        super().__init__()

        self.static_channels = static_channels
        self.dynamic_channels = dynamic_channels
        self.upsampling_layers = upsampling_layers

        self.static_pre_downblocks = []

        for i in range(self.upsampling_layers):
            self.static_pre_downblocks.append(Down(self.static_channels*2**i, min(8, self.static_channels*2**(i+1))))

        self.num_downsamp_static_channels = min(8, self.static_channels*2**self.upsampling_layers)

        self.unet = UNet(self.dynamic_channels + self.num_downsamp_static_channels, 16)

        self.upsamp_convs = []
        self.upsamp_resblocks = []

        for i in range(upsampling_layers):
            static_encoding_channels_in_layer = min(8, self.static_channels*2**(self.upsampling_layers-i))
            self.upsamp_convs.append(nn.ConvTranspose2d(self.dynamic_channels + static_encoding_channels_in_layer + 16, 16, kernel_size=2, stride=2))
            self.upsamp_resblocks.append(ResBlock(16, 16))
    
        self.semifinal_layer = ResBlock(self.static_channels + self.dynamic_channels + 16, 16)
        self.final_layer = nn.Conv2d(16, self.dynamic_channels, kernel_size=1, stride=1)

    def forward(self, lr_fields, hr_eps):

        print(lr_fields.shape, hr_eps.shape)

        eps_encodings = [hr_eps.unsqueeze(1)]

        for i in range(self.upsampling_layers):
            eps_encodings.append(self.static_pre_downblocks[i](eps_encodings[-1]))

        x = torch.cat([lr_fields, eps_encodings[-1]], dim=1)

        x = self.unet(x)

        for i in range(self.upsampling_layers):
            if i == 0:
                upscaled_lr_fields_orig = lr_fields
            else:
                upscaled_lr_fields_orig = torch.nn.Upsample(scale_factor=2**i, mode='bilinear')(lr_fields)

            x = torch.cat([x, upscaled_lr_fields_orig, eps_encodings[self.upsampling_layers - i]], dim=1)

            x = self.upsamp_convs[i](x)
            x = self.upsamp_resblocks[i](x)

        upscaled_lr_fields_orig = torch.nn.Upsample(scale_factor=2**self.upsampling_layers, mode='bilinear')(lr_fields)

        #print(x.shape, upscaled_lr_fields_orig.shape, eps_encodings[0].shape)
        x = torch.cat([x, upscaled_lr_fields_orig, eps_encodings[0]], dim=1)

        x = self.semifinal_layer(x)
        x = self.final_layer(x)

        return x

        
    