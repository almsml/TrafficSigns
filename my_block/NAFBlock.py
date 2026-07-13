import torch
import torch.nn as nn

class NAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # 使用 BatchNorm2d，适用于标准 NCHW 格式
        self.norm1 = nn.BatchNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
        self.pw2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.norm2 = nn.BatchNorm2d(channels)
        self.pw3 = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)
        self.pw4 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)


    def forward(self, x):
        residual = x
        x1 = self.norm1(x)
        x1 = self.pw1(x1)
        x1 = self.dwconv(x1)
        x1 = self.pw2(x1)
        y = residual + x1 * self.beta

        residual2 = y
        x2 = self.norm2(y)
        x2 = self.pw3(x2)
        x2_1, x2_2 = x2.chunk(2, dim=1)
        x2 = x2_1 * x2_2
        x2 = self.pw4(x2)
        y = residual2 + x2 * self.gamma

        return y
