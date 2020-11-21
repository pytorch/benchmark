from . import slomo_model as model
import torch
from torchvision.models.vgg import vgg16 as vgg16_m
import torch.nn as nn
import torch.nn.functional as F


L1_lossFn = nn.L1Loss()
MSE_LossFn = nn.MSELoss()


class Model(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.flowComp = model.UNet(6, 4).to(device)
        self.ArbTimeFlowIntrp = model.UNet(20, 5).to(device)
        self.trainFlowBackWarp = model.backWarp(352, 352, device)

        vgg16 = vgg16_m(pretrained=True)
        vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
        vgg16_conv_4_3.to(device)
        for param in vgg16_conv_4_3.parameters():
                param.requires_grad = False
        self.vgg16_conv_4_3 = vgg16_conv_4_3

    def forward(self, trainFrameIndex, I0, I1, IFrame):
        # Calculate flow between reference frames I0 and I1
        flowOut = self.flowComp(torch.cat((I0, I1), dim=1))

        # Extracting flows between I0 and I1 - F_0_1 and F_1_0
        F_0_1 = flowOut[:,:2,:,:]
        F_1_0 = flowOut[:,2:,:,:]

        fCoeff = model.getFlowCoeff(trainFrameIndex, I0.device)

        # Calculate intermediate flows
        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

        # Get intermediate frames from the intermediate flows
        g_I0_F_t_0 = self.trainFlowBackWarp(I0, F_t_0)
        g_I1_F_t_1 = self.trainFlowBackWarp(I1, F_t_1)

        # Calculate optical flow residuals and visibility maps
        intrpOut = self.ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

        # Extract optical flow residuals and visibility maps
        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
        V_t_1   = 1 - V_t_0

        # Get intermediate frames from the intermediate flows
        g_I0_F_t_0_f = self.trainFlowBackWarp(I0, F_t_0_f)
        g_I1_F_t_1_f = self.trainFlowBackWarp(I1, F_t_1_f)

        wCoeff = model.getWarpCoeff(trainFrameIndex, I0.device)

        # Calculate final intermediate frame
        Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)


        # Loss
        recnLoss = L1_lossFn(Ft_p, IFrame)

        prcpLoss = MSE_LossFn(self.vgg16_conv_4_3(Ft_p), self.vgg16_conv_4_3(IFrame))

        warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(self.trainFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(self.trainFlowBackWarp(I1, F_0_1), I0)

        loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
        loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
        loss_smooth = loss_smooth_1_0 + loss_smooth_0_1

        # Total Loss - Coefficients 204 and 102 are used instead of 0.8 and 0.4
        # since the loss in paper is calculated for input pixels in range 0-255
        # and the input to our network is in range 0-1
        loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth

        return Ft_p, loss
