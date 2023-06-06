import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContDecoder(nn.Module):

    def __init__(self, context_channels=32, coord_channels=2, output_channels=2, hidden_layer_features=[516, 256, 128, 64, 32, 16], static_channels=1):
        super().__init__()

        self.context_channels = context_channels
        self.coord_channels = coord_channels
        self.output_channels = output_channels
        self.static_channels = static_channels
        self.hid_ns = hidden_layer_features
 
        self.input_size = self.context_channels + self.coord_channels + self.static_channels + self.output_channels

        self.layers = []

        self.layers.append(nn.Linear(in_features=self.input_size, out_features=self.hid_ns[0]))

        for i in range(len(self.hid_ns)-1):
            self.layers.append(nn.Linear(in_features=self.hid_ns[i]+self.input_size, out_features=self.hid_ns[i+1]))

        self.layers.append(nn.Linear(in_features=self.hid_ns[-1], out_features=self.output_channels))


    def forward(self, lr_fields, context_grid, hr_eps, coord):

        extra_dim_coord = torch.unsqueeze(coord, 1)

        #necessary to ensure that coord ordering of x and z lines up with grid dimensions
        lr_fields = torch.permute(lr_fields, (0, 1, 3, 2))
        context_grid = torch.permute(context_grid, (0, 1, 3, 2))
        hr_eps = torch.permute(hr_eps.unsqueeze(1), (0, 1, 3, 2))

        context_pts = F.grid_sample(context_grid, extra_dim_coord, align_corners=False)
        lr_field_pts = F.grid_sample(lr_fields, extra_dim_coord, align_corners=False)
        hr_eps_pts = F.grid_sample(hr_eps, extra_dim_coord, align_corners=False)

        #reshape to feed into MLP layers
        context_pts = torch.squeeze(context_pts, dim=2)
        lr_field_pts = torch.squeeze(lr_field_pts, dim=2)
        hr_eps_pts = torch.squeeze(hr_eps_pts, dim=2)
        context_pts = torch.permute(context_pts, (0, 2, 1))
        lr_field_pts = torch.permute(lr_field_pts, (0, 2, 1))
        hr_eps_pts = torch.permute(hr_eps_pts, (0, 2, 1))


        input = torch.cat([context_pts, coord, lr_field_pts, hr_eps_pts], dim=2)

        print(input.shape)

        x = self.layers[0](input)
        x = F.relu(x)
        for i in range(1, len(self.layers)-1):

            print('here' + str(10+i))

            x = torch.cat([x, input], dim=2)
            x = self.layers[i](x)
            x = F.relu(x)

        return self.layers[-1](x)

