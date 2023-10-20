import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
import time
from glob import glob
import numpy as np

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, norm=True):
        super(double_conv, self).__init__()
        if norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True) )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.LeakyReLU(inplace=True) )

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, norm)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True,norm=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


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

class PriorNet(nn.Module):
    def __init__(self, n_channels=2, o_channels=1, norm = False):
        super(PriorNet, self).__init__()
        self.n_channels = n_channels

        n = 32
        filters = [n, n*2, n*4, n*8, n*16]
        self.AvgPool1 = nn.AvgPool2d(2, 2)
        self.AvgPool2 = nn.AvgPool2d(2, 2)
        self.inconv = double_conv(n_channels, filters[0], norm=norm)
        self.down_conv1 = down(filters[0], filters[1], norm=norm)
        self.down_conv2 = down(filters[1]+1, filters[2], norm=norm)
        self.down_conv3 = down(filters[2]+1, filters[3], norm=norm)

        self.up_conv1 = up(filters[3]+filters[2], filters[2], norm=norm)
        self.up_conv2 = up(filters[2]+filters[1], filters[1], norm=norm)
        self.up_conv3 = up(filters[1]+filters[0], filters[0], norm=norm)
        self.outc = nn.Conv2d(filters[0], o_channels, 1)


    def forward(self, SLI, Mt):
        x1 = torch.cat((SLI * (1.0-Mt), Mt), dim=1)
        x1 = self.inconv(x1)    # 2->32
        x2 = self.down_conv1(x1)    # 33->64

        M1 = self.AvgPool1(Mt)
        x3 = torch.cat((x2,M1),dim=1)
        x3 = self.down_conv2(x3)

        M2 = self.AvgPool2(M1)
        x4 = torch.cat((x3,M2),dim=1)
        x4 = self.down_conv3(x4)

        u1 = self.up_conv1(x4,x3)
        u2 = self.up_conv2(u1,x2)
        u3 = self.up_conv3(u2,x1)

        out = self.outc(u3)
        return out




