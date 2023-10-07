
import torch.nn.functional as F
import torch.utils.data
import itertools
import odl
from odl.contrib import torch as odl_torch
from utils.geometry import build_gemotry
ray_trafo = build_gemotry()
FP = odl_torch.OperatorModule(ray_trafo)
FBP_cpu = odl.tomo.fbp_op(ray_trafo, filter_type='Ram-Lak', frequency_scaling=1.0)
FBP = odl_torch.OperatorModule(FBP_cpu)

from .modules import *
from .FNet import FusionNet,FusionMap
from .IRNet import ResUnet_L3,ResUnet_L4
from .NSDNet import NSD_Net,NSD_NetSimple

"""
2023.8.02
Author: Jinqiu Xia
PND-Net framework re-build
"""


"""
the decomposed network
"""
class DecomposeNet(nn.Module):
    def __init__(self, opt):
        super(DecomposeNet, self).__init__()
        self.FP = FP
        self.FBP = FBP
        self.imPixScale = opt.imPixScale
        self.opt = opt
        self.SNet = NSD_Net()
        self.INet = ResUnet_L3(channel=1, filters=[64,128,256,512])
        self.tiny_beta = 0.001

    def getminmax(self):
        return self.opt.dataMin, self.opt.dataMax

    def normalize(self, img):
        minval, mxaval = self.getminmax()
        torch.clamp(img,minval,mxaval, out=None)
        img = (img-minval)/(mxaval-minval)
        img = img*2.0-1.0
        return img

    def Renormalize(self, img):
        minval, mxaval = self.getminmax()
        img = img*0.5 + 0.5
        img = img*(mxaval-minval)+minval
        return img

    def forward2(self,Xc):
        X_cc = self.INet(self.normalize(Xc))
        Xcc = self.Renormalize(X_cc)
        return Xcc

    def forward(self,Sm, Mask):
        Mp = self.FP(Mask).mul(self.imPixScale)
        # add a tiny value as weight for non-local artifacts
        Mp = Mp + self.tiny_beta

        Ssc, Art_s = self.SNet(Sm, Mp)
        Isc = self.FBP(Ssc).div(self.imPixScale)

        #  need to normalize the data before network propagation
        Imc = self.INet(self.normalize(Isc))
        Imc = self.Renormalize(Imc)
        Art_i = Isc - Imc
        return Ssc, Art_s, Isc, Imc, Art_i

class DecomposeNet_SimpleSNet(nn.Module):
    def __init__(self, opt):
        super(DecomposeNet_SimpleSNet, self).__init__()
        self.FP = FP
        self.FBP = FBP
        self.imPixScale = opt.imPixScale
        self.opt = opt
        self.SNet = NSD_NetSimple()
        self.INet = ResUnet_L3(channel=1, filters=[64,128,256,512])
        self.tiny_beta = 0.001

    def getminmax(self):
        return self.opt.dataMin, self.opt.dataMax

    def normalize(self, img):
        minval, mxaval = self.getminmax()
        torch.clamp(img,minval,mxaval, out=None)
        img = (img-minval)/(mxaval-minval)
        img = img*2.0-1.0
        return img

    def Renormalize(self, img):
        minval, mxaval = self.getminmax()
        img = img*0.5 + 0.5
        img = img*(mxaval-minval)+minval
        return img

    def forward2(self,Xc):
        X_cc = self.INet(self.normalize(Xc))
        Xcc = self.Renormalize(X_cc)
        return Xcc

    def forward(self,Sm, Mask):
        Mp = self.FP(Mask).mul(self.imPixScale)
        # add a tiny value as weight for non-local artifacts
        Mp = Mp + self.tiny_beta

        Ssc, Art_s = self.SNet(Sm, Mp)
        Isc = self.FBP(Ssc).div(self.imPixScale)

        #  need to normalize the data before network propagation
        Imc = self.INet(self.normalize(Isc))
        Imc = self.Renormalize(Imc)
        Art_i = Isc - Imc
        return Ssc, Art_s, Isc, Imc, Art_i


"""
Fusion network with Unet model
"""
# this is the path for synthesize artifact image
class CombineNet_Unet(nn.Module):
    def __init__(self, opt):
        super(CombineNet_Unet, self).__init__()
        self.imPixScale = opt.imPixScale
        self.FNet = FusionNet()
        self.opt = opt
        self.FP = FP
        self.FBP = FBP

    def getminmax(self):
        return self.opt.dataMin, self.opt.dataMax

    def normalize(self, img):
        minval, mxaval = self.getminmax()
        torch.clamp(img,minval,mxaval,out=None)
        img = (img-minval)/(mxaval-minval)
        img = img*2.0-1.0
        return img

    def Renormalize(self, img):
        minval, mxaval = self.getminmax()
        img = img*0.5 + 0.5
        img = img*(mxaval-minval)+minval
        return img

    def forward2(self, Xm, aI):
        Iaa = self.FNet(Xm, aI)+Xm
        return Iaa

    def forward(self,Sc, Art_s, Art_i):
        X_sca = self.FBP(Sc + Art_s).div(self.imPixScale)
        Icm = self.FNet(X_sca, Art_i) + X_sca
        Scm = self.FP(Icm).mul(self.imPixScale)
        return Icm, Scm

"""
Fusion network with simple fusion weight map
"""
# this is the path for synthesize artifact image
class CombineNet_Map(nn.Module):
    def __init__(self, opt):
        super(CombineNet_Map, self).__init__()
        self.imPixScale = opt.imPixScale
        self.FNet = FusionMap()
        self.opt = opt
        self.FP = FP
        self.FBP = FBP

    def getminmax(self):
        return self.opt.dataMin, self.opt.dataMax

    def normalize(self, img):
        minval, mxaval = self.getminmax()
        torch.clamp(img,minval,mxaval,out=None)
        img = (img-minval)/(mxaval-minval)
        img = img*2.0-1.0
        return img

    def Renormalize(self, img):
        minval, mxaval = self.getminmax()
        img = img*0.5 + 0.5
        img = img*(mxaval-minval)+minval
        return img

    def forward2(self,Xm,aI):
        W = self.FNet(Xm, aI)
        Iaa = W*aI+Xm
        return Iaa

    def forward(self,Sc, Art_s, Art_i):
        X_sca = self.FBP(Sc + Art_s).div(self.imPixScale)
        W = self.FNet(X_sca, Art_i)
        Icm = W*Art_i+X_sca
        Scm = self.FP(Icm).mul(self.imPixScale)
        return Icm, Scm

