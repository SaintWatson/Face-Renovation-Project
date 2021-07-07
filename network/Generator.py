import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.nn.utils.spectral_norm as sp_norm


class MLP(nn.Module):

    def __init__(self, in_nc, out_nc):

        super().__init__()
        

        mid_nc = min(in_nc, out_nc)
        self.fc = nn.Sequential(
            nn.Conv2d(in_nc, mid_nc, kernel_size=1 ),
            nn.ReLU(),
            nn.Conv2d(mid_nc, out_nc, kernel_size=1)
        )
    def forward(self, x):
        return self.fc(x)

class SuppressionModule(nn.Module):

    def __init__(self, in_nc, out_nc):

        super().__init__()
        
        self.weight = torch.rand(3, 3, out_nc)
        self.G = MLP(in_nc, out_nc)
        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=3)

    def forward(self, x):

        # phi = nn.ReLU(self.G(x))

        # self.weight *= phi

        return self.conv(x)

class InstanceNorm2d(nn.Module):

    def __init__(self, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def forward(self, x):
        square = torch.mul(x, x)
        mean = torch.mean(square, (2,3), True)
        rsqrt = torch.rsqrt(mean + self.epsilon)
        return x * rsqrt

class SPADE(nn.Module):

    def __init__(self, norm_nc, label_nc):

        super().__init__()
        

        self.normalizer = InstanceNorm2d(norm_nc)

        n_hidden = min(128, norm_nc)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, n_hidden, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.mlp_gamma = nn.Conv2d(n_hidden, norm_nc, kernel_size=3, padding=1, bias=False)
        self.mlp_beta = nn.Conv2d(n_hidden, norm_nc, kernel_size=3, padding=1, bias=False)

    def forward(self, x, semantic):

        N = self.normalizer(x)

        semantic = F.interpolate(semantic, size=x.size()[2:], mode='nearest')
        activate = self.mlp_shared(semantic)
        gamma = self.mlp_gamma(activate)
        beta = self.mlp_beta(activate)

        return N * gamma + beta

class SPADEBlock(nn.Module):
    
    def __init__(self, in_nc, out_nc, semantic_nc=3):
        super().__init__()
        
        self.hasShortcut = (in_nc != out_nc)
        mid_nc = min(in_nc, out_nc)

        # conv layer
        self.conv_0 = nn.Conv2d(in_nc, mid_nc, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(mid_nc, out_nc, kernel_size=3, padding=1)
        self.conv_s = nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=False)

        # spectral norm layer
        self.conv_0 = sp_norm(self.conv_0)
        self.conv_1 = sp_norm(self.conv_1)
        self.conv_s = sp_norm(self.conv_s)

        # norm layer
        self.norm_0 = SPADE(in_nc, semantic_nc)
        self.norm_1 = SPADE(mid_nc, semantic_nc)
        self.norm_s = SPADE(in_nc, semantic_nc)

    def activate(self, x):
        return F.leaky_relu(x, 0.2)

    def short_cut(self, x, semantic_nc):

        if self.hasShortcut:
            return self.conv_s(self.norm_s(x, semantic_nc))
        return x

    def forward(self, x, semantic_nc):

        mid = self.conv_0(self.activate(self.norm_0(x, semantic_nc)))
        out = self.conv_1(self.activate(self.norm_1(mid, semantic_nc)))

        return out + self.short_cut(x, semantic_nc)

class FaceGenerator(nn.Module):

    def __init__(self):

        super().__init__()

        def suppress_down(in_nc, out_nc):
            return nn.Sequential(
                nn.Conv2d(in_nc, out_nc, 5, 2, padding=2, bias=False),
                nn.BatchNorm2d(out_nc),
                nn.ReLU(),
                nn.MaxPool2d(2,2)
            )

        

        def replenish_up(in_nc, out_nc):
            return nn.Sequential(
                SPADEBlock(in_nc, out_nc),
                nn.MaxUnpool2d(2,2)
            )

        self.NN = nn.Sequential(
            suppress_down(3, 16),
            suppress_down(16, 32),
            suppress_down(32, 64),
            suppress_down(64, 128),
            suppress_down(128, 256),
            replenish_up(256, 128),
            replenish_up(128, 64),
            replenish_up(64, 32),
            replenish_up(32, 16),
            replenish_up(16, 3)
        )

        
     
        
    def forward(self, x):
        return self.NN(x)
        

if __name__ == '__main__':
    
    model = FaceGenerator()

    from torchsummary import summary
    print(summary(model, (3, 512, 512)))