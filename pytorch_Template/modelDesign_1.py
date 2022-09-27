# -*- coding: utf-8 -*-
import torch
from torch import nn


class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv1 = nn.Conv2d(256, 128, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0)
        self.Flatten = nn.Flatten()

        fc_layers = [nn.Linear(64 * 9 * 5, 256), nn.LeakyReLU()]
        fc_layers.extend([nn.Linear(256, 128), nn.LeakyReLU()])
        fc_layers.extend([nn.Linear(128, 2), nn.ReLU()])
        self.fc_layer = nn.Sequential(*fc_layers)

    def forward(self, x, data_format='channels_last'):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.Flatten(x)

        out = self.fc_layer(x)

        out = torch.cat([torch.clamp(out[:, 0:1], 0, 120),
                         torch.clamp(out[:, 1:2], 0, 60)], dim=1)
        return out
