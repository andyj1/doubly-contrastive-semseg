import torch
import torch.nn as nn
import torch.nn.functional as F


class WeatherClassifier(nn.Module):
    def __init__(self, opts, weather_class_num):
        super(WeatherClassifier, self).__init__()

        self.opts = opts
        # GAP to make BxCx1x1
        if self.opts.deeplab:
            num_channels = 2048 # 'out'
            # num_channels = 128 # 'low_level'
        else:
            num_channels = 128


        self.pool_attention = nn.AdaptiveAvgPool2d((1, 1))
        #self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_channels, weather_class_num)


    def forward(self, x):
        x = self.pool_attention(x)      # BxCxHxW --> BxCx1x1
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc(x)      # BxC --> Bxweather

        return x
