import torch
import torch.nn as nn
import torch.nn.functional as F


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
