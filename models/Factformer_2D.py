import glob

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import argparse
from tqdm import tqdm
import time
import os
import gc
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from .libs.fact.factorization_module import FABlock2D
from .libs.fact.positional_encoding_module import GaussianFourierFeatureTransform

class FactorizedTransformer(nn.Module):
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 dim_out,
                 depth,
                 **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):

            layer = nn.ModuleList([])
            layer.append(nn.Sequential(
                GaussianFourierFeatureTransform(2, dim // 2, 1),
                nn.Linear(dim, dim)
            ))
            layer.append(FABlock2D(dim, dim_head, dim, heads, dim_out, use_rope=True,
                                   **kwargs))
            self.layers.append(layer)

    def forward(self, u, pos_lst=None):
        b, nx, ny, c = u.shape
        # nx, ny = pos_lst[0].shape[0], pos_lst[1].shape[0]
        # print(f'nx, ny: {nx}, {ny}')
        if pos_lst is None:
            pos, pos_lst = self.get_grid(u.shape, u.device)
            pos = pos.view(-1, 2)
        # print(pos.shape)
        for pos_enc, attn_layer in self.layers:
            tmp = pos_enc(pos).view(1, nx, ny, -1)
            u += tmp
            u = attn_layer(u, pos_lst) + u
        return u
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1)
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1)
        return torch.cat((gridx.repeat([1, 1, size_y, 1]), gridy.repeat([1, size_x, 1, 1])), dim=-1).to(device), \
            [gridx[0,...,0], gridy[0,0]]


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.H = int(((args.h - 1) / args.h_down) + 1)
        self.W = int(((args.w - 1) / args.w_down) + 1)
        # self.resolutions = args.resolutions   # hierachical resolutions, [16, 8, 4]
        # self.out_resolution = args.out_resolution

        self.in_dim = args.in_dim * args.in_var
        self.out_dim = args.out_dim * args.out_var

        self.depth = args.depth           # depth of the encoder transformer
        self.dim = args.d_model                 # dimension of the transformer
        self.heads = args.heads
        self.dim_head = args.dim_head
        # self.reducer = args.reducer
        # self.resolution = args.resolution

        # self.pos_in_dim = args.pos_in_dim
        # self.pos_out_dim = args.pos_out_dim
        self.positional_embedding = 'rotary'
        self.kernel_multiplier = 3

        self.to_in = nn.Linear(self.in_dim, self.dim, bias=True)

        self.encoder = FactorizedTransformer(self.dim, self.dim_head, self.heads, self.dim, self.depth,
                                             kernel_multiplier=self.kernel_multiplier)

        self.down_block = nn.Sequential(
            nn.InstanceNorm2d(self.dim),
            nn.Conv2d(self.dim, self.dim//2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.dim//2, self.dim//2, kernel_size=3, stride=1, padding=1, bias=True))

        self.up_block = nn.Sequential(
            nn.Upsample(size=(self.H, self.W), mode='nearest'),
            nn.Conv2d(self.dim//2, self.dim//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.dim//2, self.dim, kernel_size=3, stride=1, padding=1, bias=True))

        self.simple_to_out = nn.Sequential(
            Rearrange('b nx ny c -> b c (nx ny)'),
            nn.GroupNorm(num_groups=8, num_channels=self.dim*2),
            nn.Conv1d(self.dim*2, self.dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv1d(self.dim, self.out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self,
                u,
                fx=None, T=None
                ):
        pos_lst = None
        # b, _, c = u.shape
        # u = u.view(b, self.H, self.W, c)
        _, nx, ny, _ = u.shape
        u = self.to_in(u)
        u_last = self.encoder(u, pos_lst)
        u = rearrange(u_last, 'b nx ny c -> b c nx ny')
        assert u.shape[1] == self.dim
        u = self.down_block(u)
        u = self.up_block(u)
        u = rearrange(u, 'b c nx ny -> b nx ny c')
        # print(u.shape, u_last.shape)
        u = torch.cat([u, u_last], dim=-1)
        u = self.simple_to_out(u)
        u = rearrange(u, 'b c (nx ny) -> b nx ny c', nx=nx, ny=ny)
        return u