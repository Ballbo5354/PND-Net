
import torch.utils.data
from .modules import *

"""
 the base network for sinogram correction
"""

class nonlocalBHC_Block(nn.Module):
    def __init__(
            self, c_in, c_out=1, ksize_mid=3, act='leaky_relu'):
        super().__init__()
        c_mid = max(c_in // 2, 32)
        c_img = 2
        self.Predict_Art = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1))
        self.BHC_net = nn.Sequential(
            Conv2dSame(c_img, c_mid, 3, 1),
            nn.InstanceNorm2d(c_mid),
            get_activation(act),
            Conv2dSame(c_mid, c_out, 5, 1),
            nn.InstanceNorm2d(c_out),
            get_activation(act),
            Conv2dSame(c_out, c_out, 3, 1),
            nn.Sigmoid())

    def forward(self, x, Mp):
        raw_art = self.Predict_Art(x)
        weight_art = self.BHC_net(torch.cat([Mp, raw_art], dim=1))
        return raw_art*weight_art+raw_art


# non-local sinogram decomposition network
class NSD_Net(nn.Module):
    def __init__(self, n_channels=2):
        super(NSD_Net, self).__init__()
        n = 32
        filters = [n, n * 2, n * 4, n * 8, n * 16]
        self.AvgPool = nn.AvgPool2d(2, 2)
        self.down1 = nn.Sequential(nn.Conv2d(filters[0], filters[0], 3, 2, 1),
                                   nn.LeakyReLU())
        self.down2 = nn.Sequential(nn.Conv2d(filters[1], filters[1], 3, 2, 1),
                                   nn.LeakyReLU())
        self.conv1 = DoubleConv_SE(n_channels, filters[0])
        self.conv2 = DoubleConv_SE(filters[0] + 1, filters[1])
        self.conv3 = DoubleConv_SE(filters[1]+1, filters[2])
        self.up2 = Upconv_SE(filters[2], filters[1])
        self.up3 = Upconv_SE(filters[1], filters[0])
        self.nonBHCNet = nonlocalBHC_Block(filters[0])

    def forward(self, Sma, Mp):
        # Sma = SinoPadding(Sma)
        # Mp = SinoPadding(Mp)

        x1 = torch.cat((Sma, Mp),dim=1)
        x1 = self.conv1(x1)
        x2 = self.down1(x1)
        M2 = self.AvgPool(Mp)
        diffY = x2.size()[2] - M2.size()[2]
        diffX = x2.size()[3] - M2.size()[3]
        M2 = F.pad(M2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x2 = torch.cat((x2, M2),dim=1)
        x2 = self.conv2(x2)
        x3 = self.down2(x2)
        M3 = self.AvgPool(M2)
        diffY = x3.size()[2] - M3.size()[2]
        diffX = x3.size()[3] - M3.size()[3]
        M3 = F.pad(M3, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x3 = torch.cat((x3, M3), dim=1)
        x3 = self.conv3(x3)
        u2 = self.up2(x3, x2)
        u3 = self.up3(u2, x1)
        Sm = self.nonBHCNet(u3, Mp)
        # Sm = GetOrigSino(Sm)
        # Sma = GetOrigSino(Sma)
        Sc = Sma - Sm
        return Sc, Sm
