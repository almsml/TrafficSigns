import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
sys.path.append('.')  # 添加当前目录


class ContrastAwareAttention(nn.Module):
    def __init__(self,in_channels,kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels*2,padding=0,kernel_size=1,bias=False)
        self.conv2 = nn.Conv2d(in_channels=in_channels*2,out_channels=in_channels,padding=0,kernel_size=1,bias=False)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)))

        self.norm = nn.BatchNorm2d(in_channels)
    def forward(self,x):
        # print(x.shape)
        x1 = self.conv1(x) 
        x_mean = F.avg_pool2d(x1,kernel_size=self.kernel_size,stride=1,padding=2)
        x_contrast = x1- x_mean
        x_sigmoid = torch.sigmoid(x_contrast)
        x2 = self.conv2(x_sigmoid*x1)
        out = self.norm(x+self.gamma*x2)
        return out



