import torch

from .modules import *

class ResUnet_L3(nn.Module):
    def __init__(self, channel=1, filters=[64,128,256,512]):
        super(ResUnet_L3, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.InstanceNorm2d(filters[0]),
            nn.LeakyReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample_()
        self.upsample_2 = Upsample_()
        self.upsample_3 = Upsample_()
        # self.upsample_1 = Upsample(filters[4], filters[4])
        # self.upsample_2 = Upsample(filters[3], filters[3])
        # self.upsample_3 = Upsample(filters[2], filters[2])
        # self.upsample_4 = Upsample(filters[1], filters[1])

        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)

        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)
        return output

class ResUnet_L4(nn.Module):
    def __init__(self, channel=1, filters=[32,64,128,256,512]):
        super(ResUnet_L4, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.InstanceNorm2d(filters[0]),
            nn.LeakyReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.bridge = ResidualConv(filters[3], filters[4], 2, 1)

        self.upsample_1 = Upsample_()
        self.upsample_2 = Upsample_()
        self.upsample_3 = Upsample_()
        self.upsample_4 = Upsample_()
        # self.upsample_1 = Upsample(filters[4], filters[4])
        # self.upsample_2 = Upsample(filters[3], filters[3])
        # self.upsample_3 = Upsample(filters[2], filters[2])
        # self.upsample_4 = Upsample(filters[1], filters[1])

        self.up_residual_conv1 = ResidualConv(filters[4] + filters[3], filters[3], 1, 1)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)
        self.up_residual_conv4 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)

        # Bridge
        x5 = self.bridge(x4)

        # Decode
        x5 = self.upsample_1(x5)
        x6 = torch.cat([x5, x4], dim=1)

        x7 = self.up_residual_conv1(x6)

        x7 = self.upsample_2(x7)
        x8 = torch.cat([x7, x3], dim=1)

        x9 = self.up_residual_conv2(x8)

        x9 = self.upsample_3(x9)
        x10 = torch.cat([x9, x2], dim=1)

        x11 = self.up_residual_conv3(x10)

        x11 = self.upsample_4(x11)
        x12 = torch.cat([x11, x1], dim=1)

        x13 = self.up_residual_conv4(x12)

        output = self.output_layer(x13)

        return output



