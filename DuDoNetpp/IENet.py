import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels=2, n_classes=1, n_filter=64):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, n_filter)
        self.down1 = down(n_filter, n_filter*2)
        self.down2 = down(n_filter*2, n_filter*4)
        self.down3 = down(n_filter*4, n_filter*8)
        self.down4 = down(n_filter*8, n_filter*8)
        self.up1 = up(n_filter*16, n_filter*4)
        self.up2 = up(n_filter*8, n_filter*2)
        self.up3 = up(n_filter*4, n_filter)
        self.up4 = up(n_filter*2, n_filter)
        self.outc = outconv(n_filter, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

# dudonet++ IENet
class IENet(nn.Module):
    def __init__(self, n_channels=2, o_channels=1, n_filter=64):
        super(IENet, self).__init__()
        self.inc1 = inconv(n_channels, n_filter)
        self.inc2 = inconv(n_channels, n_filter)
        self.down1 = down(n_filter*2, n_filter*2)
        self.down2 = down(n_filter*2, n_filter*4)
        self.down3 = down(n_filter*4, n_filter*8)
        self.down4 = down(n_filter*8, n_filter*8)
        self.up1 = up(n_filter*16, n_filter*4)
        self.up2 = up(n_filter*8, n_filter*2)
        self.up3 = up(n_filter*4, n_filter)
        self.up4 = up(n_filter*2, n_filter)
        self.outc = outconv(n_filter, o_channels)

    def forward(self, Xse, Xma, mask):
        x_se = self.inc1(torch.cat([Xse, mask], dim=1)) # 2->64
        x_ma = self.inc2(torch.cat([Xma, mask], dim=1)) # 2->64
        x2 = self.down1(torch.cat([x_ma, x_se], dim=1)) # 128->128
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x_se)   # concat the corrected Xse
        x = self.outc(x)
        return x

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

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, norm)
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


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.conv(x))