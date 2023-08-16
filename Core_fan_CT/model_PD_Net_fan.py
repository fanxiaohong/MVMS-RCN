import torch
import torch.nn as nn
from torch_radon import RadonFanbeam
from torch.nn import init

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


class DualNet(nn.Module):
    def __init__(self, n_dual):
        super(DualNet, self).__init__()
        
        self.n_dual = n_dual
        self.n_channels = n_dual + 2
        
        layers = [
            nn.Conv2d(self.n_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, self.n_dual, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*layers)
        
    def forward(self, h, Op_f, g):
        x = torch.cat([h, Op_f, g], dim=1)
        x = h + self.block(x)
        return x
    
class PrimalNet(nn.Module):
    def __init__(self, n_primal):
        super(PrimalNet, self).__init__()
        
        self.n_primal = n_primal
        self.n_channels = n_primal + 1
        
        layers = [
            nn.Conv2d(self.n_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, self.n_primal, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*layers)
        
    def forward(self, f, OpAdj_h):
        # x = self.input_concat_layer(f, OpAdj_h)
        x = torch.cat([f, OpAdj_h], dim=1)
        x = f + self.block(x)
        return x
        

class LearnedPrimalDual(nn.Module):
    def __init__(self, LayerNo):
        
        super(LearnedPrimalDual, self).__init__()
        self.n_iter = LayerNo
        self.n_primal = 5
        self.n_dual = 5
        self.primal_nets = nn.ModuleList()
        self.dual_nets = nn.ModuleList()

        for i in range(self.n_iter):
            self.primal_nets.append(
                PrimalNet(self.n_primal)
            )
            self.dual_nets.append(
                DualNet(self.n_dual)
            )


        self.primal_nets.apply(initialize_weights)
        self.dual_nets.apply(initialize_weights)

        self.det_count = 1024
        self.source_distance = 500
        self.det_spacing = 2
        
    def forward(self, cond, x0, g, theta, theta_label):

        h = torch.zeros([1,5,g.shape[2],g.shape[3]], device=g.device)
        f = torch.zeros([1,5,x0.shape[2],x0.shape[3]], device=g.device)

        radon = RadonFanbeam(x0.shape[2], theta,
                             source_distance=self.source_distance,
                             det_distance=self.source_distance, det_count=self.det_count,
                             det_spacing=self.det_spacing)

        for i in range(self.n_iter):
            ## Dual
            # Apply forward operator to f^(2)
            f_2 = f[:,1:2,:,:]
            Op_f = radon.forward(f_2)
            # Apply dual network
            h = self.dual_nets[i](h, Op_f, g)

            ## Primal
            # Apply adjoint operator to h^(1)
            h_1 = h[:,0:1,:,:]
            h_1 = radon.filter_sinogram(h_1)  # 'hann'
            OpAdj_h = radon.backprojection(h_1)
            # Apply primal network
            f = self.primal_nets[i](f, OpAdj_h)
        
        return f[:,0:1],f[:,0:1],f[:,0:1]
        