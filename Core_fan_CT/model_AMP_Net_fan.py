
import torch 
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np 
import os
# from skimage.transform import radon, iradon
from torch.nn import Module
from torch_radon import RadonFanbeam

class Denoiser(Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 1, 3, padding=1,bias=False))

    def forward(self, inputs):
        output = self.D(inputs)
        return output

class AMP_Net(nn.Module):
    def __init__(self, LayerNo):
        super(AMP_Net, self).__init__()
        self.layer_num = LayerNo
        self.denoisers = []
        self.steps = []
        for n in range(self.layer_num):
            self.denoisers.append(Denoiser())
            self.register_parameter("step_" + str(n + 1), nn.Parameter(torch.tensor(1.0), requires_grad=False))
            self.steps.append(eval("self.step_" + str(n + 1)))
        for n, denoiser in enumerate(self.denoisers):
            self.add_module("denoiser_" + str(n + 1), denoiser)

        self.det_count = 1024
        self.source_distance = 500
        self.det_spacing = 2

    def forward(self, cond, x0, sinogram, theta, theta_label):
        X = x0
        radon = RadonFanbeam(x0.shape[2], theta,
                             source_distance=self.source_distance,
                             det_distance=self.source_distance, det_count=self.det_count,
                             det_spacing=self.det_spacing)


        for n in range(self.layer_num):
            step = self.steps[n]
            denoiser = self.denoisers[n]
            sino_pred = radon.forward(X)
            filtered_sinogram = radon.filter_sinogram(sino_pred)
            outputs = radon.backprojection(sinogram - filtered_sinogram)
            z = X + step * outputs

            noise = denoiser(X)

            # ATA（noise）
            sino_noise =radon.forward(noise)  # A*noise
            filtered_sinogram_noise = radon.filter_sinogram(sino_noise)
            outputs_noise = radon.backprojection(filtered_sinogram_noise)  # AT*A*noise

            X = z + noise - step * outputs_noise

        return X, X , X

