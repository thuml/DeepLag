"""
@author: Zongyi Li
modified by Haixu Wu to adapt to this code base
"""
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math


################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        in_channels = args.in_dim * args.in_var
        out_channels = args.out_dim * args.out_var
        self.modes1 = args.num_basis
        self.modes2 = args.num_basis
        self.modes3 = args.num_basis // 2
        self.width = args.d_model
        self.padding = [int(x) for x in args.padding.split(',')]

        # self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.convs = nn.ModuleList([SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3) for _ in range(args.num_layers)])
        # self.w0 = nn.Conv3d(self.width, self.width, 1)
        # self.w1 = nn.Conv3d(self.width, self.width, 1)
        # self.w2 = nn.Conv3d(self.width, self.width, 1)
        # self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.ws = nn.ModuleList([nn.Conv3d(self.width, self.width, 1) for _ in range(args.num_layers)])
        # self.bn0 = torch.nn.BatchNorm3d(self.width)
        # self.bn1 = torch.nn.BatchNorm3d(self.width)
        # self.bn2 = torch.nn.BatchNorm3d(self.width)
        # self.bn3 = torch.nn.BatchNorm3d(self.width)
        self.bns = nn.ModuleList([torch.nn.BatchNorm3d(self.width) for _ in range(args.num_layers)])

        self.fc0 = nn.Linear(in_channels + 3, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        if not all(item == 0 for item in self.padding):
            x = F.pad(x, [0, self.padding[0], 0, self.padding[1], 0, self.padding[2]])

        # x1 = self.conv0(x)
        # x2 = self.w0(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv1(x)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2
        for i in range(self.num_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)

        if not all(item == 0 for item in self.padding):
            x = x[..., :-self.padding[2], :-self.padding[1], :-self.padding[0]]
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
