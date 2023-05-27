import torch
import torch.nn as nn
import torch.nn.functional as F


class ContDecoder(nn.Module):

    def __init__(self, context_channels=32, coord_channels=2, output_channels=2, hidden_layer_features=[32]):
        super().__init__()

        self.context_channels = context_channels
        self.coord_channels = coord_channels
        self.output_channels = output_channels
        self.hid_ns = hidden_layer_features

        self.layers = []

        self.layers.append(nn.Linear(in_features=self.context_channels + self.coord_channels, out_features=self.hid_ns[0]))

        for i in range(len(self.hid_ns)-1):
            self.layers.append(nn.Linear(in_features=self.hid_ns[i]+self.coord_channels + self.context_channels, out_features=self.hid_ns[i+1]))

        self.layers.append(nn.Linear(in_features=self.hid_ns[-1], out_features=self.output_channels))


    def forward(self, context_grid, coord):

    
        extra_dim_coord = torch.unsqueeze(coord, 1)
        

        context_pts = F.grid_sample(context_grid, extra_dim_coord, align_corners=False)
        context_pts = torch.squeeze(context_pts, dim=2)

        context_pts = torch.permute(context_pts, (0, 2, 1))
        
        input = torch.cat([context_pts, coord], dim=2)

        x = self.layers[0](input)
        x = F.relu(x)
        for i in range(1, len(self.layers)-1):

            x = torch.cat([x, input], dim=2)
            x = self.layers[i](x)
            x = F.relu(x)

        return self.layers[-1](x)

