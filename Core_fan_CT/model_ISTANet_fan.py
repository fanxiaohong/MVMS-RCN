import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np 
import os
from torch_radon import RadonFanbeam


# define basic block of FISTA-Net
class  BasicBlock(nn.Module):
    """docstring for  BasicBlock"""

    def __init__(self, features):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(features, 1, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(features, features, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(features, features, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, features, 3, 3)))

        self.det_count = 1024
        self.source_distance = 500
        self.det_spacing = 2

    def forward(self, x, theta, sinogram):
        """
        x:              image from last stage, (batch_size, channel, height, width)
        theta:          angle vector for radon/iradon transform
        sinogram:       measured signogram of the target image
        lambda_step:    gradient descent step
        soft_thr:       soft-thresholding value
        """
        # rk block in the paper
        # radon = Radon(x.shape[2], theta, clip_to_circle=True)
        # radon = RadonFanbeam(x.shape[2], theta, source_distance=512)
        # radon = RadonFanbeam(x.shape[2], theta, source_distance=x.shape[2]*1.5)
        radon = RadonFanbeam(x.shape[2], theta,
                             source_distance=self.source_distance,
                             det_distance=self.source_distance, det_count=self.det_count,
                             det_spacing=self.det_spacing)
        sino_pred = radon.forward(x)
        filtered_sinogram = radon.filter_sinogram(sino_pred)
        X_fbp = radon.backprojection(filtered_sinogram - sinogram)

        x_input = x - self.lambda_step * X_fbp

        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_pred = x_backward

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]

class ISTANet(nn.Module):
    def __init__(self, LayerNo, num_feature):
        super(ISTANet, self).__init__()
        self.LayerNo = LayerNo

        onelayer = []
        for i in range(LayerNo):
            onelayer.append(BasicBlock(num_feature))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, cond, x0, sinogram, theta, theta_label):
        # initialize the result
        x = x0
        layers_sym = []     # for computing symmetric loss
        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, theta, sinogram)
            layers_sym.append(layer_sym)

        xnew = x
        return [xnew, layers_sym, layers_sym]
