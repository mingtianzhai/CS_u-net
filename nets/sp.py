import torch
from torch import nn
import torch.nn.functional as F

from nets.CBMA import CBAMLayer


class StripPooling(nn.Module):
    def __init__(self, in_channels, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))#1*W
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))#H*1
        inter_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1, bias=False),
                                   nn.BatchNorm2d(in_channels))
        self._up_kwargs = up_kwargs
        self.dd = CBAMLayer(in_channels)

    def forward(self, x):
        _, _, h, w = x.size()

        x1 = self.conv1(x)

        x2 = F.interpolate(self.conv2(self.pool1(x1)), (h, w), **self._up_kwargs)#结构图的1*W的部分
        x3 = F.interpolate(self.conv3(self.pool2(x1)), (h, w), **self._up_kwargs)#结构图的H*1的部分
        x4 = self.conv4(F.relu_(x2 + x3))#结合1*W和H*1的特征
        out = self.conv5(x4)
        x1  = self.dd(x)
        out =torch.mul(x1,out)
        return F.relu_(x + out)#将输出的特征与原始输入特征结合