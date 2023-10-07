from .modules import *


"""
 Fusion network is mask pyramid unet 
"""

# 2023.08.13: 激活函数范围受限Tanh-->[-1,1], 去掉激活函数,并且增加最后的art到模型输出作为残差学习
class FusionNet(nn.Module):
    def __init__(self, n_channels=2):
        super(FusionNet, self).__init__()
        n = 32
        filters = [n, n * 2, n * 4, n * 8, n * 16]
        self.down_1 = nn.Sequential(nn.Conv2d(filters[0], filters[0], 3, 2, 1),
                                   nn.LeakyReLU())
        self.down_2 = nn.Sequential(nn.Conv2d(filters[1], filters[1], 3, 2, 1),
                                    nn.LeakyReLU())
        self.down_3 = nn.Sequential(nn.Conv2d(filters[2], filters[2], 3, 2, 1),
                                    nn.LeakyReLU())

        self.AvgPool = nn.AvgPool2d(2, 2)

        self.conv1 = DoubleConv_SE(n_channels, filters[0], innorm='instance')
        self.conv2 = DoubleConv_SE(filters[0] + 1, filters[1], innorm='instance')
        self.conv3 = DoubleConv_SE(filters[1] + 1, filters[2], innorm='instance')
        self.conv4 = DoubleConv_SE(filters[2], filters[3], innorm='instance')

        self.up1 = Upconv_SE(filters[3], filters[2], innorm='instance')
        self.up2 = Upconv_SE(filters[2], filters[1], innorm='instance')
        self.up3 = Upconv_SE(filters[1], filters[0], innorm='instance')
        self.outc = nn.Conv2d(filters[0], 1, kernel_size=1)

    def forward(self,Ic,art):
        x1 = torch.cat((Ic, art),dim=1)

        x1 = self.conv1(x1)

        x2 = self.down_1(x1)
        M2 = self.AvgPool(art)
        x2 = torch.cat((x2, M2),dim=1)
        x2 = self.conv2(x2)

        x3 = self.down_2(x2)
        M3 = self.AvgPool(M2)
        x3 = torch.cat((x3, M3),dim=1)
        x3 = self.conv3(x3)

        x4 = self.down_3(x3)
        x4 = self.conv4(x4)

        u1 = self.up1(x4, x3)
        u2 = self.up2(u1, x2)
        u3 = self.up3(u2, x1)

        Im = self.outc(u3)+art
        return Im


"""
FusionMap: simple fusion module
"""
class FusionMap(nn.Module):
    def __init__(self, c_in=1, c_out=1):
        super().__init__()
        c_mid = max(c_in // 2, 64)
        ksize_mid = 3
        self.conv_art = nn.Sequential(nn.Conv2d(c_in, c_mid, kernel_size=1),
                                      nn.LeakyReLU())
        self.WNet = nn.Sequential(
            Conv2dSame(c_mid+c_in, c_mid*2, 1, 1),
            nn.InstanceNorm2d(c_mid*2),
            nn.LeakyReLU(),
            Conv2dSame(c_mid*2, c_mid, ksize_mid, 1),
            nn.InstanceNorm2d(c_mid),
            nn.LeakyReLU(),
            Conv2dSame(c_mid, c_out, 1, 1),
            nn.Sigmoid())

    def forward(self, Ic, art):
        art_f = self.conv_art(art)
        weight_art = self.WNet(torch.cat([Ic, art_f], dim=1))
        return weight_art

