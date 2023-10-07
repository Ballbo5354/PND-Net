import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from dudonet_model.SENet import SENet,SinoPadding,GetOrigSino
from dudonet_model.IENet import IENet

from utils.geometry import build_gemotry
from odl.contrib import torch as odl_torch
import odl

ray_trafo = build_gemotry()
FP = odl_torch.OperatorModule(ray_trafo)
FBP = odl.tomo.fbp_op(ray_trafo, filter_type='Ram-Lak', frequency_scaling=1.0)
FBP = odl_torch.OperatorModule(FBP)

class DuDoNet(nn.Module):
    def __init__(self, opt):
        super(DuDoNet, self).__init__()
        self.SENet = SENet(norm=False)
        self.IENet = IENet()
        self.imPixScale = opt.imPixScale
        self.minval = opt.dataMin
        self.maxval = opt.dataMax

    def getminmax(self):
        return self.minval, self.maxval

    def Normal_Sino(self, img):
        img = img/4.0
        return img

    def Renormal_Sino(self, img):
        img = img*4.0
        return img

    def normalize(self, img):
        img = torch.clamp(img, self.minval, self.maxval, out=None)
        img = (img-self.minval)/(self.maxval-self.minval)
        return img

    def Renormalize(self, img):
        img = img*(self.maxval-self.minval)+self.minval
        return img

    def forward(self, SLI, Mt, XLI):
        # SLI = self.Normal_Sino(SLI)
        Sino = self.SENet(SLI, Mt)
        # Sino = self.Renormal_Sino(Sino)
        # SLI = self.Renormal_Sino(SLI)

        Sino = SLI*(1.0-Mt)+Sino*Mt
        sino2im = FBP(Sino).div(self.imPixScale)
        sino2im = self.normalize(sino2im)
        ImgCorr = self.IENet(sino2im, XLI)
        return Sino, sino2im, ImgCorr
