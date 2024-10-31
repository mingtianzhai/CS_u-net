import torch
from torch import nn
import torch.nn.functional as F

from nets.CBMA import CBAMLayer


class BasicBlock1(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(BasicBlock1, self).__init__()
        self.left = nn.Sequential(

            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, padding=0, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),  # inplace=True表示进行原地操作，一般默认为False，表示新建一个变量存储操作
            nn.Conv2d(outchannel, outchannel, kernel_size=1, stride=stride, padding=0, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential(


        )
        self.dd = nn.Conv2d(inchannel,outchannel,1,1,0,bias=True)
        self.cbm = CBAMLayer(outchannel)
        # 论文中模型架构的虚线部分，需要下采样
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out1 = self.left(x)# 这是由于残差块需要保留原始输入
        x1 = self.dd(x)
        out2 = self.cbm(x1)

        out3 = torch.mul(out1,out2)
        out3 = out3+out1
        out = self.shortcut(x)+out3 # 这是ResNet的核心，在输出上叠加了输入x

        return out