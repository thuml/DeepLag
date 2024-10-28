"""
@author: Qilong Ma
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        padding = kernel_size // 2
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # b c z h w
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        # b c z h w
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, trilinear=True):
        super(Up, self).__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # b c z h w
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2,
                        diffZ // 2, diffZ - diffZ // 2,])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Attention(nn.Module):
    def __init__(self, dim, patch_size=(4,4,4), heads = 8, dim_head = 64, dropout = 0., shape=(1,1), which_q='euler'):
        super(Attention, self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.which_q = which_q

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        z, h, w = shape
        self.z, self.h, self.w = z, h, w
        
        if self.which_q == 'euler':
            self.to_q = nn.Linear(dim, inner_dim, bias = False)
            self.to_kv = nn.Linear(dim+3, inner_dim * 2, bias = False)
        elif self.which_q == 'lagrange':
            self.to_q = nn.Linear(dim+3, inner_dim, bias = False)
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.to_out2 = nn.Sequential(
            nn.Linear(inner_dim, 3),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward_euler(self, x, h, p):
        # x: b (h_p w_p z_p) c, h: b k c, p: b k 3

        q = self.to_q(x) # b n (h d)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads) # b h n d
        kv = self.to_kv(torch.cat([h, p], dim=-1)).chunk(2, dim = -1) # [b k (h d)] * 2
        k, v = map(lambda t: rearrange(t, 'b k (h d) -> b h k d', h = self.heads), kv) # b h k d
        # print('qkv:', q.shape, k.shape, v.shape)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # b h n k
        # print('q*k:', dots.shape)
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v) # b h n d
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out) # b n c

        return x

    def forward_lagrange(self, x, h, p):
        # x: b (h_p w_p z_p) c, h: b k c, p: b k 3

        p = p.to(torch.float32)
        p /= repeat(torch.tensor([self.z, self.h, self.w], dtype=p.dtype, device=p.device), 'd -> b k d', b=p.shape[0], k=p.shape[1])

        h_and_p = torch.cat([h, p], dim=-1) # b k (c+3)

        q = self.to_q(h_and_p) # b k (h d)
        q = rearrange(q, 'b k (h d) -> b h k d', h = self.heads) # b h k d
        kv = self.to_kv(x).chunk(2, dim = -1) # [b n (h d)] * 2
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv) # b h n d
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # b h k n
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v) # b h k d
        out = rearrange(out, 'b h k d -> b k (h d)')
        h = self.to_out(out) # b k c
        p = self.to_out2(out) # b k 3
        p *= repeat(torch.tensor([self.z, self.h, self.w], dtype=p.dtype, device=p.device), 'd -> b k d', b=p.shape[0], k=p.shape[1])

        return h, p
    
    def forward(self, x, h, p):
        if self.which_q == 'euler':
            return self.forward_euler(x, h, p)
        elif self.which_q == 'lagrange':
            return self.forward_lagrange(x, h, p)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., is_coo=False, euler_shape=(1,1,1)):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.is_coo = is_coo
        self.z, self.h, self.w = euler_shape

    def forward(self, x):
        if not self.is_coo:
            return self.net(x)
        else:
            x /= repeat(torch.tensor([self.z, self.h, self.w], dtype=x.dtype, device=x.device), 'd -> b k d', b=x.shape[0], k=x.shape[1])
            x = self.net(x)
            x *= repeat(torch.tensor([self.z, self.h, self.w], dtype=x.dtype, device=x.device), 'd -> b k d', b=x.shape[0], k=x.shape[1])
            return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., patch_size=(1,1,1), shape=(1,1,1), which_q='euler'):
        super(Transformer, self).__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_h = nn.LayerNorm(dim)
        # self.norm_p = nn.LayerNorm(2)
        self.layers = nn.ModuleList([])
        self.which_q = which_q
        if self.which_q == 'euler':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim, patch_size = patch_size, heads = heads, dim_head = dim_head, dropout = dropout, shape=shape, which_q=which_q),
                    FeedForward(dim, mlp_dim, dropout = dropout)
                ]))
        elif self.which_q == 'lagrange':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim, patch_size = patch_size, heads = heads, dim_head = dim_head, dropout = dropout, shape=shape, which_q=which_q),
                    FeedForward(3, mlp_dim, dropout = dropout, is_coo=True, euler_shape=shape),
                    FeedForward(dim, mlp_dim, dropout = dropout),
                ]))
    def forward(self, x, h, p): # x: b n c, h: b k c, p: b k 2
        if self.which_q == 'euler':
            for attn, ff in self.layers:
                x = attn(self.norm_x(x), self.norm_h(h), p) + x
                x = ff(self.norm_x(x)) + x
            return x
        elif self.which_q == 'lagrange':
            for attn, ff, ff2 in self.layers:
                p_prev = p
                h_prev = h
                h, p = attn(self.norm_x(x), self.norm_h(h), p)
                p = p + p_prev
                h = h + h_prev
                # p = ff(self.norm_p(p)) + p
                p = ff(p) + p
                h = ff2(self.norm_h(h)) + h
            return h, p


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        in_channels = args.in_dim * args.in_var
        out_channels = args.out_dim * args.out_var
        self.num_layers = args.num_layers
        self.width = args.d_model
        self.padding = tuple(int(x) for x in args.padding.split(','))
        self.patch_size = tuple(int(x) for x in args.patch_size.split(','))

        self.img_z = int(((args.z - 1) / args.z_down) + 1)
        self.img_h = int(((args.h - 1) / args.h_down) + 1)
        self.img_w = int(((args.w - 1) / args.w_down) + 1)
        num_pixel = self.img_z * self.img_h * self.img_w
        self.img_patch_z = self.img_z // self.patch_size[0]
        self.img_patch_h = self.img_h // self.patch_size[1]
        self.img_patch_w = self.img_w // self.patch_size[2]
        num_patches = self.img_patch_z * self.img_patch_h * self.img_patch_w
        self.pixel_per_patch = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        patch_dim = self.width * self.pixel_per_patch
        self.to_patch_embedding_conv = nn.Sequential(
            Rearrange('b c z h w -> b z h w c'),
            nn.LayerNorm(self.width),
            Rearrange('b z h w c -> b c z h w'),
            nn.Conv3d(self.width, self.width, self.patch_size, stride=self.patch_size),
            Rearrange('b c z_p h_p w_p -> b (z_p h_p w_p) c'),
            nn.LayerNorm(self.width),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_pixel, self.width))
        self.dropout = nn.Dropout(args.emb_dropout)
        self.num_samples = min(args.num_samples, num_pixel)

        self.down = nn.ModuleList([
            Down(self.width*(2**i), self.width*(2**(i+1))) for i in range(self.num_layers-2)
        ])
        self.down.append(Down(self.width*(2**(self.num_layers-2)), self.width*(2**(self.num_layers-2))))
        self.up = nn.ModuleList([Up(self.width*2, self.width)])
        for i in range(self.num_layers-2):
            self.up.append(Up(self.width*(2**(i+2)), self.width*(2**i)))
        self.img_z_layers = [self.img_patch_z//(2**i) for i in range(self.num_layers)]
        self.img_h_layers = [self.img_patch_h//(2**i) for i in range(self.num_layers)]
        self.img_w_layers = [self.img_patch_w//(2**i) for i in range(self.num_layers)]
        self.coo_layers = [torch.stack(torch.meshgrid([
                                torch.arange(self.img_z_layers[i]), torch.arange(self.img_h_layers[i]), torch.arange(self.img_w_layers[i])
                            ]),dim=-1).reshape(-1,3).cuda() 
                           for i in range(self.num_layers)] # (z h w) 3

        self.euler_evo = nn.ModuleList([
            Transformer(self.width*(2**i), depth=1, heads=8, dim_head=64, mlp_dim=512, dropout=args.mlp_dropout, patch_size=self.patch_size, shape=(self.img_z_layers[i], self.img_h_layers[i], self.img_w_layers[i]), which_q='euler') for i in range(self.num_layers-1)
        ])
        self.euler_evo.append(Transformer(self.width*(2**(self.num_layers-2)), depth=1, heads=8, dim_head=64, mlp_dim=512, dropout=args.mlp_dropout, patch_size=self.patch_size, shape=(self.img_z_layers[-1], self.img_h_layers[-1], self.img_w_layers[-1]), which_q='euler'))
        self.lagrange_evo = nn.ModuleList([
            Transformer(self.width*(2**i), depth=1, heads=8, dim_head=64, mlp_dim=512, dropout=args.mlp_dropout, patch_size=self.patch_size, shape=(self.img_z_layers[i], self.img_h_layers[i], self.img_w_layers[i]), which_q='lagrange') for i in range(self.num_layers-1)
        ])
        self.lagrange_evo.append(Transformer(self.width*(2**(self.num_layers-2)), depth=1, heads=8, dim_head=64, mlp_dim=512, dropout=args.mlp_dropout, patch_size=self.patch_size, shape=(self.img_z_layers[-1], self.img_h_layers[-1], self.img_w_layers[-1]), which_q='lagrange'))
        
        self.resample_strategy = args.resample_strategy
        if self.resample_strategy == 'learned':
            self.resample_probs = [nn.Sequential(
                Rearrange('b (z h w) c -> b c z h w', z=self.img_z_layers[i], h=self.img_h_layers[i]), 
                DoubleConv(self.width*(2**i), 1, self.width, kernel_size=9),
                Rearrange('b 1 z h w -> b (z h w) 1'),
                Rearrange('b n 1 -> b n'),
                nn.Softmax(dim=-1),
            ).cuda() for i in range(self.num_layers-1)]
            self.resample_probs.append(nn.Sequential(
                Rearrange('b (z h w) c -> b c z h w', z=self.img_z_layers[-1], h=self.img_h_layers[-1]), 
                DoubleConv(self.width*(2**(self.num_layers-2)), 1, self.width, kernel_size=9),
                Rearrange('b 1 z h w -> b (z h w) 1'),
                Rearrange('b n 1 -> b n'),
                nn.Softmax(dim=-1),
            ).cuda())

        self.mlp_head = nn.Linear(self.width, self.pixel_per_patch*self.width)
        num_chans = []
        for i in range(self.num_layers):
            num_chans.append(args.d_model*(2**i) if i < self.num_layers-1 else args.d_model*(2**(i-1)))
        self.fc0 = nn.Linear(in_channels + 3, self.width)  # input channel is 3: (a(x, y), x, y)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.fc3 = nn.ModuleList([nn.Linear(2*num_chan, num_chan) for num_chan in num_chans])
        self.conv_pooldown = nn.Conv3d(3, 3, 2*self.num_layers//2, stride=2*self.num_layers//2)
    
    def set_bdydom(self, boundary, domain):
        # domain & boundary: z h w (0-1 mask)
        self.boundary_patch = F.max_pool3d(boundary[None, ...], kernel_size=self.patch_size).squeeze(0).cuda() # z/p h/p w/p
        self.domain_patch = F.max_pool3d(domain[None, ...], kernel_size=self.patch_size).squeeze(0).cuda()
        
        self.boundary_ms = []
        self.coo_boundary_ms = []
        for i in range(self.num_layers):
            boundary_ds = F.max_pool3d(self.boundary_patch[None, ...], kernel_size=2**i).squeeze(0)
            self.boundary_ms.append(boundary_ds)
            coo_boundary_ds = torch.nonzero(boundary_ds) # K 3
            self.coo_boundary_ms.append(coo_boundary_ds)
        self.domain_ms = []
        self.coo_domain_ms = []
        for i in range(self.num_layers):
            domain_ds = F.max_pool3d(self.domain_patch[None, ...], kernel_size=2**i).squeeze(0)
            self.domain_ms.append(domain_ds)
            coo_domain_ds = torch.nonzero(domain_ds) # K 3
            self.coo_domain_ms.append(coo_domain_ds)

    def forward(self, x, h_x_q, h_coo_q, h_coo_offset_q):
        # x: b z h w t*v, h_x_q: b k c, h_coo_q: b k 3
        B, Z, H, W, _ = x.shape
        Z_P, H_P, W_P = Z//self.patch_size[0], H//self.patch_size[1], W//self.patch_size[2]
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1) # b z h w c_ori+3
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous() # b c(width) z h w
        if not all(item == 0 for item in self.padding):
            x = F.pad(x, [self.padding[2]//2, self.padding[2]-self.padding[2]//2, 
                          self.padding[1]//2, self.padding[1]-self.padding[1]//2, 
                          self.padding[0]//2, self.padding[0]-self.padding[0]//2])
        
        x = self.to_patch_embedding_conv(x)
        x += self.pos_embedding
        x = self.dropout(x) # b n=(z/p*h/p*w/p) c

        xs = [x]
        x = rearrange(x, 'b (z_p h_p w_p) c -> b c z_p h_p w_p', z_p=Z_P, h_p=H_P)
        for i in range(self.num_layers-1):
            x = self.down[i](x)
            xs.append(rearrange(x, 'b c z_ds h_ds w_ds -> b (z_ds h_ds w_ds) c'))

        coo_offset_zxys = []
        for i in range(self.num_layers):
            if None in h_coo_q:
                num_samples = self.num_samples // (8**i)
                idx_coo_sample = torch.multinomial(self.resample_probs[i](xs[i])*self.domain_ms[i][None,...].reshape(1,-1), num_samples, replacement=False)
                h_coo_q[i] = torch.stack([self.coo_layers[i][idx].type_as(xs[i]) for idx in idx_coo_sample], dim=0) # b(list) k 3
            h_coo_offset_q[i] = -h_coo_q[i]

            xs[i] = self.euler_evo[i](xs[i], h_x_q[i], h_coo_q[i]) # b n c
            h_x_q[i], h_coo_q[i] = self.lagrange_evo[i](xs[i], h_x_q[i], h_coo_q[i]) # b k c, b k 3
            h_coo_q[i] = self.chk_resample(xs[i], (self.img_z_layers[i], self.img_h_layers[i], self.img_w_layers[i]), h_coo_q[i], i)
            h_x_q[i] = torch.cat((h_x_q[i], self.trilinear_interp(xs[i], h_coo_q[i], (self.img_z_layers[i], self.img_h_layers[i], self.img_w_layers[i]))),dim=-1) # b k 2c
            h_x_q[i] = self.fc3[i](h_x_q[i])
            h_coo_offset_q[i] += h_coo_q[i]

        x = xs[-1]
        x = rearrange(x, 'b (z_ds h_ds w_ds) c -> b c z_ds h_ds w_ds', z_ds=self.img_z_layers[-1], h_ds=self.img_h_layers[-1])
        for i in range(self.num_layers-2, -1, -1):
            x_up = rearrange(xs[i], 'b (z_ds h_ds w_ds) c -> b c z_ds h_ds w_ds', z_ds=self.img_z_layers[i], h_ds=self.img_h_layers[i])
            x = self.up[i](x, x_up)
        x = rearrange(x, 'b c z_p h_p w_p -> b (z_p h_p w_p) c')

        x = self.mlp_head(x) # b (z/p h/p w/p) (p p p c)
        x = rearrange(x, 'b (z_p h_p w_p) (pz ph pw c) -> b c (z_p pz) (h_p ph) (w_p pw)', 
                      h_p=H_P, w_p=W_P, pz=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2])

        if not all(item == 0 for item in self.padding):
            x = x[..., self.padding[0]//2:-(self.padding[0]-self.padding[0]//2), 
                  self.padding[1]//2:-(self.padding[1]-self.padding[1]//2), 
                  self.padding[2]//2:-(self.padding[2]-self.padding[2]//2)]
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x, h_x_q, h_coo_q, h_coo_offset_q, coo_offset_zxys

    @staticmethod
    def get_grid(shape, device):
        batchsize, size_z, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        return torch.cat((gridz, gridx, gridy), dim=-1).to(device)
    
    @staticmethod
    def trilinear_interp(x, p, shape):
        # x: b n c, p: b k 3
        D, H, W = shape
        
        def get_x_q(x, q):
            # x: b n c, q: b k 3
            C = x.shape[-1]
            index = q[..., 0]*H*W + q[..., 1]*W + q[..., 2]  # idx = x*h*w + y*w + z, (b, k)
            index = repeat(index, 'b k -> b k c', c=C)
            x_q = torch.gather(x, dim=1, index=index) # b k c
            return x_q
        
        # coord of 8-neig
        q_000 = p.detach().floor() # b k 3
        q_111 = q_000 + 1
        q_000 = torch.cat([torch.clamp(q_000[..., :1], 0, D-1), torch.clamp(q_000[..., 1:2], 0, H-1), torch.clamp(q_000[..., 2:], 0, W-1)], dim=-1).long() # b k 3
        q_111 = torch.cat([torch.clamp(q_111[..., :1], 0, D-1), torch.clamp(q_111[..., 1:2], 0, H-1), torch.clamp(q_111[..., 2:], 0, W-1)], dim=-1).long()
        q_001 = torch.cat([q_000[..., :2], q_111[..., 2:]], dim=-1)
        q_010 = torch.cat([q_000[..., :1], q_111[..., 1:2], q_000[..., 2:]], dim=-1)
        q_011 = torch.cat([q_000[..., :1], q_111[..., 1:]], dim=-1)
        q_100 = torch.cat([q_111[..., :1], q_000[..., 1:]], dim=-1)
        q_101 = torch.cat([q_111[..., :1], q_000[..., 1:2], q_111[..., 2:]], dim=-1)
        q_110 = torch.cat([q_111[..., :2], q_000[..., 2:]], dim=-1)

        
        # trilinear kernel 
        g_000 = (1 + (q_000[..., 0].type_as(p) - p[..., 0])) * (1 + (q_000[..., 1].type_as(p) - p[..., 1])) * (1 + (q_000[..., 2].type_as(p) - p[..., 2]))
        g_001 = (1 + (q_001[..., 0].type_as(p) - p[..., 0])) * (1 + (q_001[..., 1].type_as(p) - p[..., 1])) * (1 - (q_001[..., 2].type_as(p) - p[..., 2]))
        g_010 = (1 + (q_010[..., 0].type_as(p) - p[..., 0])) * (1 - (q_010[..., 1].type_as(p) - p[..., 1])) * (1 + (q_010[..., 2].type_as(p) - p[..., 2]))
        g_011 = (1 + (q_011[..., 0].type_as(p) - p[..., 0])) * (1 - (q_011[..., 1].type_as(p) - p[..., 1])) * (1 - (q_011[..., 2].type_as(p) - p[..., 2]))
        g_100 = (1 - (q_100[..., 0].type_as(p) - p[..., 0])) * (1 + (q_100[..., 1].type_as(p) - p[..., 1])) * (1 + (q_100[..., 2].type_as(p) - p[..., 2]))
        g_101 = (1 - (q_101[..., 0].type_as(p) - p[..., 0])) * (1 + (q_101[..., 1].type_as(p) - p[..., 1])) * (1 - (q_101[..., 2].type_as(p) - p[..., 2]))
        g_110 = (1 - (q_110[..., 0].type_as(p) - p[..., 0])) * (1 - (q_110[..., 1].type_as(p) - p[..., 1])) * (1 + (q_110[..., 2].type_as(p) - p[..., 2]))
        g_111 = (1 - (q_111[..., 0].type_as(p) - p[..., 0])) * (1 - (q_111[..., 1].type_as(p) - p[..., 1])) * (1 - (q_111[..., 2].type_as(p) - p[..., 2]))
        
        # x of 8-neig
        x_q_000 = get_x_q(x, q_000) # b k c
        x_q_001 = get_x_q(x, q_001)
        x_q_010 = get_x_q(x, q_010)
        x_q_011 = get_x_q(x, q_011)
        x_q_100 = get_x_q(x, q_100)
        x_q_101 = get_x_q(x, q_101)
        x_q_110 = get_x_q(x, q_110)
        x_q_111 = get_x_q(x, q_111)
        
        # trilinear interp
        x_p = (
            g_000[..., None] * x_q_000 +
            g_001[..., None] * x_q_001 +
            g_010[..., None] * x_q_010 +
            g_011[..., None] * x_q_011 +
            g_100[..., None] * x_q_100 +
            g_101[..., None] * x_q_101 +
            g_110[..., None] * x_q_110 +
            g_111[..., None] * x_q_111
        ) # b k c
        return x_p
    
    def chk_resample(self, x, shape, coord, layer):
        # coord: b k 3
        B, k, _ = coord.shape
        Z, H, W = shape
        front_out = coord[:,:,0] < 0 # b k
        back_out = coord[:,:,0] >= Z-1 # x=h-1, bottom 2 of 4-neig oor
        top_out = coord[:,:,1] < 0 # b k
        bottom_out = coord[:,:,1] >= H-1 # x=h-1, bottom 2 of 4-neig oor
        left_out = coord[:,:,2] < 0
        right_out = coord[:,:,2] >= W-1 # y=w-1, right 2 of 4-neig oor
        
        p_out = front_out | back_out | top_out | bottom_out | left_out | right_out # b k
        idx_out = torch.nonzero(p_out.flatten()).flatten() # m, ranging from [0, b*k)
        if self.resample_strategy == 'uniform':
            coo_init = torch.cat([torch.randint(0,Z-1,(B*k,1)), torch.randint(0,H-1,(B*k,1)), torch.randint(0,W-1,(B*k,1))], dim=1).type_as(coord) # b*k, 3
            new_coord = torch.scatter(coord.reshape(B*k,3), dim=0, index=idx_out[:,None].repeat(1,3), src=coo_init)
        elif self.resample_strategy == 'boundary':
            idx_coo_sample = torch.multinomial(1./torch.ones(self.coo_boundary_ms[layer].shape[0]), k, replacement=False) # k
            coo_init = self.coo_boundary_ms[layer][idx_coo_sample].type_as(coord) # k 3
            new_coord = torch.scatter(coord.reshape(B*k,3), dim=0, index=idx_out[:,None].repeat(1,3), src=coo_init.repeat(B,1))
        elif self.resample_strategy == 'domain':
            idx_coo_sample = torch.multinomial(1./torch.ones(self.coo_domain_ms[layer].shape[0]), k, replacement=False) # k
            coo_init = self.coo_domain_ms[layer][idx_coo_sample].type_as(coord) # k 3
            new_coord = torch.scatter(coord.reshape(B*k,3), dim=0, index=idx_out[:,None].repeat(1,3), src=coo_init.repeat(B,1))
        elif self.resample_strategy == 'learned':
            idx_coo_sample = torch.multinomial(self.resample_probs[layer](x), k, replacement=False) # b k
            coo_init = torch.cat([self.coo_layers[layer][idx].type_as(coord) for idx in idx_coo_sample], dim=0) # b*k 3
            new_coord = torch.scatter(coord.reshape(B*k,3), dim=0, index=idx_out[:,None].repeat(1,3), src=coo_init)
        new_coord = new_coord.reshape(B, k, 3)
        return new_coord
    