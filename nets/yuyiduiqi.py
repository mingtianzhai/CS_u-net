import random
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nets.CBMA import CBAMLayer
from nets.chanca import BasicBlock1


#from nets.gaosfenbu import gaos


class duqi(nn.Module):
    def __init__(self, inputs1, ouput):
        super(duqi, self).__init__()
        self.bn = torch.nn.BatchNorm2d(inputs1,momentum=0.1)
        self.re = torch.nn.Sigmoid()
        self.sc = BasicBlock1(inputs1,inputs1)
        self.sc1= BasicBlock1(inputs1,inputs1)
        #self.norm = torch.norm(inputs1, p=2, dim=0)
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        #self.pool12 = nn.AdaptiveAvgPool2d((16, 16))
        self.sf = torch.nn.Tanh()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))  # 1*W
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))  # H*1
        self.csd = CBAMLayer(inputs1,)









    def forward(self, inputs1,x = random.uniform(1,3)):
        a,b,c,d = inputs1.size()
        globals = self.pool2(inputs1)
        #globals1= self.pool2(inputs1)
        globals1 = torch.norm(inputs1, p=2, dim=2, keepdim=True)
        #x = random.uniform(1,4)
        globals1 = 2*math.pi*np.exp(-((globals1*globals1)/2*x)).cuda()


        globals = self.sc(self.sc1(globals))

        #globals = self.bn(globals)
        #globals = self.re(globals)
        globals = torch.mul(globals,globals1)
        globals = self.sf(globals)
        #globals = F.interpolate(globals,(c,d,),None,'bilinear',True)
        #globals = torch.mul(globals,inputs1)
        globals = self.csd(globals)
        globals = torch.mul(globals,inputs1)


        return globals