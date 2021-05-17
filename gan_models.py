import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np

# G(z)
class Generator_MLP(nn.Module):
    # initializers
    def __init__(self, batch_size=64, image_size=64, z_dim=100, mlp_dim=64, rgb_channel=3):
        self.size = [image_size, rgb_channel]
        super(Generator_MLP, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [SpectralNorm(nn.Linear(in_feat, out_feat))]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(z_dim, 128, normalize=False),
            *block(128, mlp_dim*4),
            *block(mlp_dim*4, mlp_dim*12),
            *block(mlp_dim*12, mlp_dim*48),
            nn.Linear(mlp_dim*48, image_size*image_size*rgb_channel),
            nn.Tanh()
        )

    # forward method
    def forward(self, z):
        # 64 x 128
        x = self.model(z)
        x = x.view(x.size(0), self.size[1], self.size[0], self.size[0])

        return x, None, None

class Discriminator_MLP(nn.Module):
    # initializers
    def __init__(self, batch_size=64, image_size=64, mlp_dim=64, rgb_channel=3):
        super(Discriminator_MLP, self).__init__()
        self.model = nn.Sequential(
            SpectralNorm(nn.Linear(image_size*image_size*rgb_channel, mlp_dim * 48)),
            #nn.BatchNorm1d(mlp_dim * 48),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Linear(mlp_dim*48, mlp_dim*12)),
            #nn.BatchNorm1d(mlp_dim * 12),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Linear(mlp_dim*12, mlp_dim*4)),
            #nn.BatchNorm1d(mlp_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(mlp_dim*4, 1),
        )
        #self.model = nn.Sequential(
        #    nn.Linear(image_size*image_size*rgb_channel, mlp_dim * 48),
        #    #nn.BatchNorm1d(mlp_dim * 48),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    nn.Linear(mlp_dim*48, mlp_dim*12),
        #    #nn.BatchNorm1d(mlp_dim * 12),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    nn.Linear(mlp_dim*12, mlp_dim*4),
        #    #nn.BatchNorm1d(mlp_dim * 4),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    nn.Linear(mlp_dim*4, 1),
        #)


    # forward method
    def forward(self, input):
        # 64 x 3 x 64 x 64
        x = input.view(input.size(0), -1)
        x = self.model(x)
        # 64 x 1
        return x.squeeze(), None, None