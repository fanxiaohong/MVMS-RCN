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


# define basic block of FISTA-Net
class  BasicBlock(nn.Module):
    """docstring for  BasicBlock"""

    def __init__(self, features):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(features, 1, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(features, features, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(features, features, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(features, features, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(features, features, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, features, 3, 3)))


    def forward(self, x, theta, sinogram):
        """
        x:              image from last stage, (batch_size, channel, height, width)
        theta:          angle vector for radon/iradon transform
        sinogram:       measured signogram of the target image
        lambda_step:    gradient descent step
        soft_thr:       soft-thresholding value
        """
        # rk block in the paper
        radon = Radon(x.shape[2], theta, det_count=729)
        sino_pred = radon.forward(x)
        filtered_sinogram = radon.filter_sinogram(sino_pred)
        X_fbp = radon.backprojection(filtered_sinogram - sinogram)

        x_input = x - self.lambda_step * X_fbp

        x_D = F.conv2d(x_input, self.conv_D, padding=1)

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        x_pred = x_input + x_G

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss, symloss]

class ISTANetplus(nn.Module):
    def __init__(self, LayerNo, num_feature):
        super(ISTANetplus, self).__init__()
        self.LayerNo = LayerNo

        onelayer = []
        for i in range(LayerNo):
            onelayer.append(BasicBlock(num_feature))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, cond, x0, sinogram, theta, theta_label):
        """
        sinogram    : measured signal vector;
        x0          : initialized x with FBP
        """

        # initialize the result
        x = x0
        layers_sym = []     # for computing symmetric loss
        for i in range(self.LayerNo):
            [x, layer_sym, layer_st] = self.fcs[i](x, theta, sinogram)
            layers_sym.append(layer_sym)
        xnew = x
        return [xnew, layers_sym, layers_sym]
