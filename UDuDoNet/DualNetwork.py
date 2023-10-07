from .baseModel import *
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


"""
2023.7.1
Author: Jinqiu Xia
UDuDoNet-Net framework
"""


# this is the metal artifact reduction path
import numpy as np
import matplotlib.pyplot as plt
class UDuDoNet(nn.Module):
    def __init__(self, opt):
        super(UDuDoNet, self).__init__()
        self.FP = FP
        self.FBP = FBP
        self.imPixScale = opt.imPixScale
        self.opt = opt
        self.SNet = SNet()  #Sma, Mp, Mt
        self.INet = UNet(norm=True)  #Ise

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

    def forward(self,Ia, Sa, Mp, Mt):
        Sse = self.SNet(Sa, Mp, Mt)
        Sm = Sa-Sse
        Isc = self.FBP(Sse).div(self.imPixScale)
        aS = Ia - Isc

        Isc = self.normalize(Isc)
        aI = self.INet(Isc)
        Isc = self.Renormalize(Isc)
        aI = self.Renormalize(aI)
        Imc = Isc-aI
        return Sse, aS, Isc, Imc, aI,Sm


