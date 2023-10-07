import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from scipy.ndimage import gaussian_filter
import numpy as np


class SobelConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, requires_grad=False):
        assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
        assert out_channels % 4 == 0, 'SobelConv2d\'s out_channels must be a multiple of 4.'
        assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        # Initialize the Sobel kernal
        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2

        # Define the trainable sobel factor
        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)

    def forward(self, x):
        if torch.cuda.is_available():
            self.sobel_factor = self.sobel_factor.cuda()
            if isinstance(self.bias, nn.Parameter):
                self.bias = self.bias.cuda()

        sobel_weight = self.sobel_weight * self.sobel_factor

        if torch.cuda.is_available():
            sobel_weight = sobel_weight.cuda()

        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out

class None_Module(nn.Module):
    def __init__(self, a):
        super().__init__()

    def forward(self,x):
        return x

def get_norm(name, out_channels):
    if name == 'batch':
        norm = nn.BatchNorm2d(out_channels)
    elif name == 'instance':
        norm = nn.InstanceNorm2d(out_channels)
    else:
        norm = None_Module(out_channels)
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


class TotalVariationLoss(nn.Module):
    def __init__(self, c_img=3):
        super().__init__()
        self.c_img = c_img

        kernel = torch.FloatTensor([[0, 1, 0],[1, -2, 0],[0, 0, 0]]).view(1, 1, 3, 3)
        kernel = torch.cat([kernel] * c_img, dim=0)
        self.register_buffer('kernel', kernel)

    def gradient(self, x):
        return nn.functional.conv2d(
            x, self.kernel, stride=1, padding=1, groups=self.c_img)

    def forward(self, results):
        loss = 0.
        for i, res in enumerate(results):
            grad = self.gradient(res.unsqueeze(1))
            loss += torch.mean(torch.abs(grad))
        return loss

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

# residual block without normalize
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.InstanceNorm2d(input_dim),
            nn.LeakyReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.InstanceNorm2d(output_dim),
            nn.LeakyReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.InstanceNorm2d(output_dim)
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)

class ResidualConv2(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv2, self).__init__()

        self.conv_down = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
        )

        self.conv_block = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x1 = self.conv_down(x)
        return self.conv_block(x1) + x1


class DoubleConv_SE(nn.Module):
    def __init__(self, in_c, out_ch, innorm = None):
        super(DoubleConv_SE, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_c, out_ch, kernel_size=3, padding=1, bias=False),
            get_norm(innorm,out_ch),
            nn.LeakyReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            get_norm(innorm,out_ch),
            nn.LeakyReLU())

    def forward(self,Sino):
        return self.double_conv(Sino)


# to match the different between up and down, need to padding
class Upconv_SE(nn.Module):
    def __init__(self, in_channels, out_channels, innorm=None):
        super(Upconv_SE, self).__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels,out_channels,3,1,1),
            nn.LeakyReLU())

        self.conv = nn.Sequential(
            DoubleConv_SE(in_channels, out_channels, innorm))

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # Padding [1,2,3,4]分别对应左右上下padding的大小
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, norm=None):
        super(Upsample, self).__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(input_dim, output_dim,3,1,1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.up_conv(x)


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()
        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class Conv2dSame(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        padding = self.conv_same_pad(kernel_size, stride)
        if type(padding) is not tuple:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = nn.Sequential(
                nn.ConstantPad2d(padding*2, 0),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0)
            )

    def conv_same_pad(self, ksize, stride):
        if (ksize - stride) % 2 == 0:
            return (ksize - stride) // 2
        else:
            left = (ksize - stride) // 2
            right = left + 1
            return left, right

    def forward(self, x):
        return self.conv(x)


"""
 Discriminator 
"""
class Discriminator(nn.Module):
    def __init__(self, input_nc=1):
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

# sobel global Disc
class Discriminator_sobel(nn.Module):
    def __init__(self, input_nc=5):
        super(Discriminator_sobel, self).__init__()
        self.sobel_conv = SobelConv2d(1, 4)

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
        x = torch.cat([x, self.sobel_conv(x)], dim=1)
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



""
import functools
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
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


class NLayerDiscriminator_sobel(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=5, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator_sobel, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.BatchNorm2d
        else:
            use_bias = norm_layer == nn.BatchNorm2d

        kw = 4
        padw = 1

        self.sobel_conv = SobelConv2d(1, 4)

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
        input1 = torch.cat([input, self.sobel_conv(input)],dim=1)
        return self.model(input1)
