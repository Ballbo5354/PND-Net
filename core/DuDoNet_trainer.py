import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from scipy.io import savemat
import os.path as path
from PIL import Image
from utils.visual_tool import *
from DuDoNet.DuDoNetModel import DuDoNet
from utils.geometry import build_gemotry
from utils.io import *
from odl.contrib import torch as odl_torch
import odl


ray_trafo = build_gemotry()
ForwardProj = odl_torch.OperatorModule(ray_trafo)
# FBP = odl_torch.OperatorModule(BeamGemotry)
FBP = odl.tomo.fbp_op(ray_trafo, filter_type='Ram-Lak', frequency_scaling=1.0)
FBP = odl_torch.OperatorModule(FBP)


class DudoNet_Trainer(nn.Module):
    def __init__(self):
        super(DudoNet_Trainer, self).__init__()

    def initial(self, opt, device, train_dataset, train_loder, val_loder=None):
        # dataset loder
        self.train_dataset = train_dataset
        self.train_loder = train_loder
        self.val_loder = val_loder

        self.MAR_Net = DuDoNet(opt).to(device)
        self.FP = ForwardProj
        self.device = device
        self.opt = opt

        self.optimizer = torch.optim.Adam(self.MAR_Net.parameters(),
                                        lr=self.opt.lr,
                                        betas=(0.5, 0.999), eps=1e-06)
        # self.schedual = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
        #                                                   lr_lambda=LambdaLR(opt.Epochs, opt.start_epoch, opt.decay_epoch, last_scale=0.05).step)
        self.schedual = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                          30, 0.5, last_epoch=-1)
        self.criterion = torch.nn.L1Loss(reduction='mean')

        self.logger_valloss = np.zeros([opt.Epochs,1],dtype=np.float32)
        self.logger_trainloss = np.zeros([opt.Epochs, 1], dtype=np.float32)


    def save_model(self, epoch):
        save_ckpt('{:s}/{:s}/weight/epoch{:d}.pth'.format(self.opt.save_dir, self.opt.model_name, epoch),
                  [('model', self.MAR_Net)], None, epoch)

    def load_model(self, pth):
        load_ckpt(pth, [('model', self.MAR_Net)])

    def normalize(self, img):
        minval, mxaval = self.opt.dataMin, self.opt.dataMax
        img = torch.clamp(img,minval,mxaval,out=None)
        img = (img-minval)/(mxaval-minval)
        return img

    def Renormalize(self, img):
        minval, mxaval = self.opt.dataMin, self.opt.dataMax
        img = img*(mxaval-minval)+minval
        return img

    def to_npy(self, *tensors, squeeze=False):
        if len(tensors) == 1:
            if squeeze:
                return tensors[0].detach().cpu().numpy().squeeze()
            else:
                return tensors[0].detach().cpu().numpy()
        else:
            if squeeze:
                return [t.detach().cpu().numpy().squeeze() for t in tensors]
            else:
                return [t.detach().cpu().numpy() for t in tensors]

    def get_visual(self, visual_dir, epoch, images, n_rows=4, normalize=True):
        if normalize: images = (images - 0) / (0.5 - 0)
        visuals = make_grid(images, nrow=images.shape[0] // n_rows, normalize=False)
        visuals = self.to_npy(visuals).transpose(1, 2, 0)
        visuals = (visuals * 255).astype(np.uint8)
        visual_file = path.join(visual_dir,
                                "epoch{}.png".format(epoch))
        Image.fromarray(visuals).convert('RGB').save(visual_file)

    def validation(self, epoch):
        dataType = torch.float32
        self.MAR_Net.eval()
        with torch.no_grad():
            valloss = []
            output_img = []
            i = 0
            # gt, metal, XLI, mask_onlymetal, Sgt, Sma, SLI
            for Igt, Xma, XLI, Mask, Sgt, Sma, SLI in self.val_loder:
                self.MAR_Net.train()
                Igt = Igt.to(device=self.device, dtype=dataType)
                Sgt = Sgt.to(device=self.device, dtype=dataType)
                Xma = Xma.to(device=self.device, dtype=dataType)
                XLI = XLI.to(device=self.device, dtype=dataType)
                SLI = SLI.to(device=self.device, dtype=dataType)
                Mask = Mask.to(device=self.device, dtype=dataType)

                Mp = self.FP(Mask).mul(self.opt.imPixScale)
                Mt = (Mp > 0.0) * 1.0
                Mt = Mt.to(dtype=dataType)

                # Sma, Mp, Xma, mask
                # Sino, sino2im, ImgCorr
                Sse, Ise, Iout = self.MAR_Net(SLI, Mt, XLI)

                loss = self.criterion(Sse, Sgt) + \
                       self.criterion(Iout * (1.0 - Mask), Igt * (1.0 - Mask)) + \
                       self.criterion(Ise * (1.0 - Mask), Igt * (1.0 - Mask))
                valloss.append(loss.item())

                if i==0:
                    output_img = torch.cat([Igt, Xma, Ise, Iout],dim=2)
                    # self.get_visual('{:s}/png/'.format(self.opt.loggerpath), epoch, output_img, n_rows=1)
                    i = i+1
                elif i<=2:
                    img = torch.cat([Igt, Xma, Ise, Iout], dim=2)
                    output_img = torch.cat([output_img,img], dim=0)
                    i = i + 1

            self.get_visual('{:s}/{:s}/png/'.format(self.opt.save_dir, self.opt.model_name),
                            epoch,output_img,n_rows=1,normalize=False)
            self.logger_valloss[epoch,0]=np.mean(valloss)

    def model_fitting(self):
        dataType = torch.float32
        for epoch in range(self.opt.Epochs):
            loop = tqdm(enumerate(self.train_loder), total=len(self.train_dataset)/self.opt.batch_size)
            trainloss = []
            # gt,metal,XLI,mask_onlymetal,Sgt,Sma,SLI
            for step,(Igt, Xma, XLI, Mask, Sgt, Sma,SLI) in loop:
                self.MAR_Net.train()
                Igt = Igt.to(device=self.device,dtype=dataType)
                Sgt = Sgt.to(device=self.device, dtype=dataType)
                Xma = Xma.to(device=self.device, dtype=dataType)
                XLI = XLI.to(device=self.device, dtype=dataType)
                SLI = SLI.to(device=self.device, dtype=dataType)
                Mask = Mask.to(device=self.device, dtype=dataType)

                Mp = self.FP(Mask).mul(self.opt.imPixScale)
                Mt = (Mp>0.0)*1.0
                Mt = Mt.to(dtype=dataType)

                # Sma, Mp, Xma, mask
                # Sino, sino2im, ImgCorr
                Sse, Ise, Iout = self.MAR_Net(SLI, Mt, XLI)

                loss = self.criterion(Sse, Sgt) +\
                       self.criterion(Iout*(1-Mask), Igt*(1-Mask)) + \
                        self.criterion(Ise*(1-Mask), Igt*(1-Mask))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loop.set_description(f'Epoch [{self.opt.Epochs}/{epoch}]')
                loop.set_postfix(total_loss=loss.item())

                trainloss.append(loss.item())

            self.logger_trainloss[epoch,0]=np.mean(trainloss)
            self.schedual.step()
            if (epoch) % self.opt.logger_freq == 0:
                if self.val_loder is not None:
                    self.validation(epoch)
                loss_logger = {'trainLos': self.logger_trainloss,'valLoss':self.logger_valloss}
                savemat('{:s}/{:s}/trainloss.mat'.format(self.opt.save_dir, self.opt.model_name), mdict=loss_logger)

            if (epoch) % self.opt.save_epoch_freq == 0:
                self.save_model(epoch)

