import torch
import torch.nn as nn


class ContDecoder(nn.Module):

    def __init__(self, context_channels=32, coord_channels=2, output_channels=2, hidden_layer_features=[256, 128, 64, 32]):
        super().__init__()

        self.context_channels = context_channels
        self.coord_channels = coord_channels
        self.output_channels = output_channels
        self.hid_ns = hidden_layer_features

        self.layers = []

        self.layers.append(nn.Linear(in_features=self.context_channels + self.coord_channels, out_features=self.hid_ns[0]))

        for i in range(len(self.hi_ns)-1):
            self.layers.append(nn.Linear(in_features=self.hid_ns[i]+self.coord_channels, out_features=self.hid_ns[i+1]))

        self.layers.append(nn.Linear(in_features=self.hid_ns[-1], out_features=self.output_channels))


    def forward(self, context, coord):

        input = torch.cat([context, coord], dim=1)

        x = self.layers[0](input)
        x = nn.ReLU(x)
        for i in range(1, len(self.layers)-1):
            x = torch.cat([x, input], dim=1)
            x = self.layers[i](x)
            x = nn.ReLU(x)

        return self.layers[-1](x)

