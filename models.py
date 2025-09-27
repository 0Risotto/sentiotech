import torch.nn as nn
class TestModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TestModel, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.pool = nn.AdaptiveAvgPool2d(1) #global avgerage pooling
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x