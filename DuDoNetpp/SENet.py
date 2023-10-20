import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
import time
from glob import glob
import numpy as np
from dudonet_model.IENet import down, up, double_conv

def SinoPadding(Sino, size=620, dim=2):
    lenth = (size - Sino.size()[dim]) // 2
    CatData = Sino[:, :, 0:lenth, :] if dim==2 else Sino[:, 0:lenth, :]
    Result = torch.cat((Sino, CatData), dim=dim)
    CatData = Sino[:, :, -lenth:, :] if dim==2 else Sino[:, -lenth:, :]
    Result = torch.cat((CatData, Result), dim=dim)
    return Result

def GetOrigSino(Sino, dim=2):
    return Sino[:, :, 10:10+600, :] if dim==2 else Sino[:, 10:10+600, :]

# take place the pooling layer by convolution
class PoolingLayer(nn.Module):
    def __init__(self,channel):
        super(PoolingLayer, self).__init__()
        self.pool = nn.Sequential(
            nn.Conv2d(channel,channel,2,2),
            nn.LeakyReLU()
            )
    def forward(self,x):
        x = self.pool(x)
        return x

def weights_init(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()


class SENet(nn.Module):
    def __init__(self, n_channels=2, o_channels=1):
        super(SENet, self).__init__()
        self.n_channels = n_channels

        n = 32
        filters = [n, n*2, n*4, n*8, n*16]
        self.AvgPool1 = nn.AvgPool2d(2, 2)
        self.AvgPool2 = nn.AvgPool2d(2, 2)
        self.inconv = double_conv(n_channels, filters[0], norm=False)
        self.down_conv1 = down(filters[0], filters[1], norm=False)
        self.down_conv2 = down(filters[1]+1, filters[2], norm=False)
        self.down_conv3 = down(filters[2]+1, filters[3], norm=False)

        self.up_conv1 = up(filters[3]+filters[2], filters[2], norm=False)
        self.up_conv2 = up(filters[2]+filters[1], filters[1], norm=False)
        self.up_conv3 = up(filters[1]+filters[0], filters[0], norm=False)
        self.outc = nn.Conv2d(filters[0], o_channels, 1)

    def forward(self, Sma, Mp):

        # padding sinogram
        Sma = SinoPadding(Sma)
        Mp = SinoPadding(Mp)

        x1 = torch.cat((Sma, Mp), dim=1)
        x1 = self.inconv(x1)    # 2->32
        x2 = self.down_conv1(x1)    # 33->64

        M1 = self.AvgPool1(Mp)
        x3 = torch.cat((x2,M1),dim=1)
        x3 = self.down_conv2(x3)

        M2 = self.AvgPool2(M1)
        x4 = torch.cat((x3,M2),dim=1)
        x4 = self.down_conv3(x4)

        u1 = self.up_conv1(x4,x3)
        u2 = self.up_conv2(u1,x2)
        u3 = self.up_conv3(u2,x1)

        # crop sinogram back
        out = GetOrigSino(self.outc(u3))
        return out




