import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from scipy.ndimage import gaussian_filter
import numpy as np


def get_norm(name, out_channels):
    if name == 'batch':
        norm = nn.BatchNorm2d(out_channels)
    elif name == 'instance':
        norm = nn.InstanceNorm2d(out_channels)
    else:
        norm = None
    return norm


def get_activation(name):
    if name == 'relu':
        activation = nn.ReLU()
    elif name == 'elu':
        activation = nn.ELU()
    elif name == 'leaky_relu':
        activation = nn.LeakyReLU(negative_slope=0.2)
    elif name == 'tanh':
        activation = nn.Tanh()
    elif name == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        activation = None
    return activation

class GaussianFilterLayer(nn.Module):
    def __init__(self, n_layers, filter_size, sigma, groups=1):
        super(GaussianFilterLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(int(filter_size/2)),
            nn.Conv2d(
                n_layers,
                n_layers,
                filter_size,
                stride=1,
                padding=0,
                bias=None,
                groups=n_layers
            )
        )
        self.weights_init(filter_size, sigma)
        # self.requires_grad(False)

    def forward(self, x):
        return self.seq(x)

    def weights_init(self, filter_size, sigma):
        filter_mat = np.zeros((filter_size, filter_size))
        filter_mat[int(filter_size/2), int(filter_size/2)] = 1
        k = gaussian_filter(filter_mat, sigma=sigma)
        for name, param in self.named_parameters():
            param.data.copy_(torch.as_tensor(k))
            param.requires_grad = False


class DownConv(nn.Module):
    def __init__(self, input_dim, output_dim,ksize, stride, padding):
        super(DownConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=ksize, stride=stride, padding=padding
            ),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_block(x)


class DoubleConv_SE(nn.Module):
    def __init__(self,in_c,out_ch,innorm=False):
        super(DoubleConv_SE, self).__init__()
        if innorm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_c, out_ch, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU())
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_c,out_ch,kernel_size=3,padding=1,bias=True),
                nn.LeakyReLU(),
                nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1,bias=True),
                nn.LeakyReLU())

    def forward(self,Sino):
        return self.double_conv(Sino)


class Upconv_SE(nn.Module):
    def __init__(self, in_channels, out_channels,innorm=False):
        super(Upconv_SE, self).__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv_SE(in_channels, out_channels, innorm)
        )

        self.conv = nn.Sequential(
            DoubleConv_SE(in_channels,out_channels,innorm)
        )

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv_SE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_SE, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, n_filter=64, norm=True):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, n_filter)
        self.down1 = down(n_filter, n_filter*2,norm=norm)
        self.down2 = down(n_filter*2, n_filter*4,norm=norm)
        self.down3 = down(n_filter*4, n_filter*8,norm=norm)
        self.down4 = down(n_filter*8, n_filter*8,norm=norm)

        self.up1 = up(n_filter*16, n_filter*4,norm=norm)
        self.up2 = up(n_filter*8, n_filter*2,norm=norm)
        self.up3 = up(n_filter*4, n_filter,norm=norm)
        self.up4 = up(n_filter*2, n_filter,norm=norm)
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

# SNet: U-Net of depth 2
class SNet(nn.Module):
    def __init__(self, n_channels=2):
        super(SNet, self).__init__()
        n = 32
        filters = [n, n * 2, n * 4, n * 8, n * 16]
        self.AvgPool = nn.AvgPool2d(2, 2)
        self.down1 = DownConv(filters[0], filters[0], 3, 2, 1)
        self.down2 = DownConv(filters[1], filters[1], 3, 2, 1)
        self.active = nn.ReLU()
        self.conv1 = DoubleConv_SE(n_channels, filters[0])
        self.conv2 = DoubleConv_SE(filters[0], filters[1])
        self.conv3 = DoubleConv_SE(filters[1], filters[2])
        self.up2 = Upconv_SE(filters[2], filters[1])
        self.up3 = Upconv_SE(filters[1], filters[0])
        self.outconv = nn.Conv2d(filters[0], 1, kernel_size=1)

    def forward(self,Sma, Mp, Mt):
        x1 = torch.cat((Sma,Mp),dim=1)
        x1 = self.conv1(x1)
        x2 = self.down1(x1)
        x2 = self.conv2(x2)
        x3 = self.down2(x2)
        x3 = self.conv3(x3)
        u2 = self.up2(x3,x2)
        u3 = self.up3(u2,x1)
        Sse = self.outconv(u3) * Mt + Sma
        return Sse

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, norm=True):
        super(double_conv, self).__init__()
        if norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.InstanceNorm2d(out_ch),
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
        self.act = nn.Tanh()

    def forward(self, x):
        return self.conv(x)

########################################
""" Discriminator """
########################################
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.

    This class is adopted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

import functools
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.BatchNorm2d
        else:
            use_bias = norm_layer == nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)