import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np 
import os
from torch_radon import Radon
###########################################################################
class condition_network(nn.Module):
    def __init__(self,LayerNo, num_features):
        super(condition_network, self).__init__()

        self.fc1 = nn.Linear(1, num_features, bias=True)
        self.fc2 = nn.Linear(num_features, num_features, bias=True)
        self.fc3 = nn.Linear(num_features, LayerNo, bias=True)

        self.act12 = nn.ReLU(inplace=True)
        self.act3 = nn.Softplus()
        self.LayerNo = LayerNo

    def forward(self, x):
        x=x[:,0:1]

        x = self.act12(self.fc1(x))
        x = self.act12(self.fc2(x))
        x = self.act3(self.fc3(x))
        return x[0,0:self.LayerNo]
###########################################################################
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        # two 3X3 equal one 5X5
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, input):
        return self.conv(input)
#########################################################################################
class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)
#########################################################################################
class ResUnet(nn.Module):
    def __init__(self, in_ch, out_ch,channel_G):
        super(ResUnet, self).__init__()

        self.head = SingleConv(in_ch, channel_G)
        self.conv_left1 = DoubleConv(channel_G, channel_G)
        # self.pool1 = nn.MaxPool2d(2)
        self.pool1 = nn.Conv2d(channel_G, channel_G, kernel_size=2, stride=2, padding=0)  # down sampling
        self.conv_left2 = DoubleConv(channel_G, channel_G)
        self.pool2 = nn.Conv2d(channel_G, channel_G, kernel_size=2, stride=2, padding=0)
        self.conv_left3 = DoubleConv(channel_G, channel_G)
        self.pool3 = nn.Conv2d(channel_G, channel_G, kernel_size=2, stride=2, padding=0)
        self.conv_left4 = DoubleConv(channel_G, channel_G)
        self.pool4 = nn.Conv2d(channel_G, channel_G, kernel_size=2, stride=2, padding=0)
        self.conv_left5 = DoubleConv(channel_G, channel_G)
        self.pool5 = nn.Conv2d(channel_G, channel_G, kernel_size=2, stride=2, padding=0)
        self.conv_remainder = DoubleConv(channel_G, channel_G)
        self.up5 = nn.ConvTranspose2d(channel_G, channel_G, 2, stride=2)
        self.conv_right5 = DoubleConv(2 * channel_G, channel_G)
        self.up4 = nn.ConvTranspose2d(channel_G, channel_G, 2, stride=2)
        self.conv_right4 = DoubleConv(2 * channel_G, channel_G)
        self.up3 = nn.ConvTranspose2d(channel_G, channel_G, 2, stride=2)
        self.conv_right3 = DoubleConv(2*channel_G, channel_G)
        self.up2 = nn.ConvTranspose2d(channel_G, channel_G, 2, stride=2)
        self.conv_right2 = DoubleConv(2*channel_G, channel_G)
        self.up1 = nn.ConvTranspose2d(channel_G, channel_G, 2, stride=2)
        self.conv_right1 = DoubleConv(2*channel_G, channel_G)
        self.tail = SingleConv(channel_G, out_ch)

    def forward(self, x):
        x_head = self.head(x)
        c1 = self.conv_left1(x_head)
        p1 = self.pool1(c1)
        c2 = self.conv_left2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv_left3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv_left4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv_left5(p4)
        p5 = self.pool5(c5)
        remainder = self.conv_remainder(p5)
        up_5 = self.up5(remainder + p5)  # residual
        merge5 = torch.cat([up_5, c5], dim=1)
        c_r5 = self.conv_right5(merge5)
        up_4 = self.up4(c_r5 + p4) # residual
        merge4 = torch.cat([up_4, c4], dim=1)
        c_r4 = self.conv_right4(merge4)
        up_3 = self.up3(c_r4 + p3) # residual
        merge3 = torch.cat([up_3, c3], dim=1)
        c_r3 = self.conv_right3(merge3)
        up_2 = self.up2(c_r3 + p2) # residual
        merge2 = torch.cat([up_2, c2], dim=1)
        c_r2 = self.conv_right2(merge2)
        up_1 = self.up1(c_r2 + p1)  # residual
        merge1 = torch.cat([up_1, c1], dim=1)
        c_r1 = self.conv_right1(merge1)
        c_r1_2 = self.tail(c_r1 + x_head)
        return c_r1_2     # residual
###########################################################################
# define basic block of FISTA-Net
class  BasicBlock(nn.Module):
    def __init__(self, features):
        super(BasicBlock, self).__init__()

        self.ResUnet = ResUnet(9, 1, features)  # local residual learning

    def forward(self, x, theta, theta_label, sinogram, x_step,radon,radon_label):
        # low/high sparse-view data fuse
        # low-level
        sino_pred = radon.forward(x)
        sino_pred_dg = radon_label.forward(x)   # [1,1,60,512]
        filtered_sinogram = radon.filter_sinogram(sino_pred)
        filtered_sinogram_high = radon_label.filter_sinogram(sino_pred_dg)

        low_to_high = F.interpolate(filtered_sinogram, size=[len(theta_label), filtered_sinogram_high.shape[3]], mode="bilinear")

        # sinogram interp
        low_to_high_label = F.interpolate(sinogram, size=[len(theta_label), filtered_sinogram_high.shape[3]], mode="bilinear")
        high_image = radon_label.backprojection(low_to_high_label)  # Interp high image

        # inpterp image error
        high_image_error = radon_label.backprojection(low_to_high - filtered_sinogram_high)   # Interp erro

        # rk block in the paper
        X_fbp = radon.backprojection(sinogram- filtered_sinogram)   # AT(y- Ax)
        X_fbp_low = x - radon.backprojection(filtered_sinogram)   # (I-ATA)x
        X_fbp_high = x - radon_label.backprojection(filtered_sinogram_high)  # (I-ATA)x

        # mid reconstruction
        r_mid = x + X_fbp - X_fbp_low
        sino_pred_mid = radon.forward(r_mid)
        sino_pred_dg_mid = radon_label.forward(r_mid)
        filtered_sinogram_mid = radon.filter_sinogram(sino_pred_mid)
        filtered_sinogram_high_mid = radon_label.filter_sinogram(sino_pred_dg_mid)
        r_low = r_mid - radon.backprojection(filtered_sinogram_mid)
        r_high = r_mid - radon_label.backprojection(filtered_sinogram_high_mid)

        # Dk block in the paper， FISTA-Net architecture
        sigma = x_step.repeat(x.shape[0], 1, x.shape[2], x.shape[3])

        # error fuse
        error_low = torch.cat((X_fbp, X_fbp_low, r_low), 1)
        error_high = torch.cat((high_image_error, X_fbp_high, r_high), 1)

        x_input_cat = torch.cat((x, high_image, error_high, error_low, sigma), 1)
        x_pred = self.ResUnet(x_input_cat)

        return [x_pred]
###########################################################################
class MVMS_RCN(nn.Module):
    def __init__(self, LayerNo,num_feature):
        super(MVMS_RCN, self).__init__()
        self.LayerNo = LayerNo
        self.num_features  = num_feature

        onelayer = []
        self.bb = BasicBlock(self.num_features)
        for i in range(self.LayerNo):
            onelayer.append(self.bb)

        self.fcs = nn.ModuleList(onelayer)

        self.condition = condition_network(self.LayerNo,self.num_features)

    def forward(self, cond, x0, sinogram,theta,theta_label):
        x_step = self.condition(cond)

        radon = Radon(x0.shape[2], theta, det_count=729)
        radon_label = Radon(x0.shape[2], theta_label, det_count=729)

        x = x0

        for i in range(self.LayerNo):
            [x] = self.fcs[i](x, theta, theta_label,sinogram, x_step[i],
                              radon,radon_label)

        xnew = x
        return [xnew, xnew, x0]
