import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np 
import os
from torch_radon import Radon


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)
###########################################################################
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)
###########################################################################
class ResUnet(nn.Module):
    def __init__(self, in_ch, out_ch,num_feature):
        super(ResUnet, self).__init__()

        self.conv1 = DoubleConv(in_ch, num_feature)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(num_feature, 2*num_feature)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(2*num_feature, 4*num_feature)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(4*num_feature, 8*num_feature)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(8*num_feature, 16*num_feature)
        self.up6 = nn.ConvTranspose2d(16*num_feature, 8*num_feature, 2, stride=2)
        self.conv6 = DoubleConv(16*num_feature, 8*num_feature)
        self.up7 = nn.ConvTranspose2d(8*num_feature, 4*num_feature, 2, stride=2)
        self.conv7 = DoubleConv(8*num_feature, 4*num_feature)
        self.up8 = nn.ConvTranspose2d(4*num_feature, 2*num_feature, 2, stride=2)
        self.conv8 = DoubleConv(4*num_feature, 2*num_feature)
        self.up9 = nn.ConvTranspose2d(2*num_feature, num_feature, 2, stride=2)
        self.conv9 = DoubleConv(2*num_feature, num_feature)
        self.conv10 = nn.Conv2d(num_feature, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        # out = nn.Sigmoid()(c10)
        return c10+x
###########################################################################
# Define resunet
class FBPConNet(torch.nn.Module):
    def __init__(self, num_feature):
        super(FBPConNet, self).__init__()
        self.unet = ResUnet(1, 1, num_feature)
        self.unet.apply(initialize_weights)

    def forward(self, cond, x0, sinogram, theta, theta_label):
        x_pred = self.unet(x0)  # local residual learning
        return [x_pred,x_pred,x_pred]
##################################################################################3