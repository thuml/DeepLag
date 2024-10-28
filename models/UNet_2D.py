import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math

import matplotlib.pyplot as plt

################################################################
# Multiscale modules 2D
################################################################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Model(nn.Module):
    def __init__(self, args, bilinear=True):
        super(Model, self).__init__()
        in_channels = args.in_dim * args.in_var
        out_channels = args.out_dim * args.out_var
        width = args.d_model
        num_token = args.num_token
        num_basis = args.num_basis
        patch_size = [int(x) for x in args.patch_size.split(',')]
        padding = [int(x) for x in args.padding.split(',')]
        # multiscale modules
        self.inc = DoubleConv(width, width)
        self.down1 = Down(width, width * 2)
        self.down2 = Down(width * 2, width * 4)
        self.down3 = Down(width * 4, width * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(width * 8, width * 16 // factor)
        self.up1 = Up(width * 16, width * 8 // factor, bilinear)
        self.up2 = Up(width * 8, width * 4 // factor, bilinear)
        self.up3 = Up(width * 4, width * 2 // factor, bilinear)
        self.up4 = Up(width * 2, width, bilinear)
        self.outc = OutConv(width, width)
        # projectors
        self.padding = padding
        self.fc0 = nn.Linear(in_channels + 2, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        if not all(item == 0 for item in self.padding):
            x = F.pad(x, [self.padding[1]//2, self.padding[1]-self.padding[1]//2, 
                          self.padding[0]//2, self.padding[0]-self.padding[0]//2])

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        if not all(item == 0 for item in self.padding):
            x = x[..., self.padding[0]//2:-(self.padding[0]-self.padding[0]//2), 
                       self.padding[1]//2:-(self.padding[1]-self.padding[1]//2)]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
