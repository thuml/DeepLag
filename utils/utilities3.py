import os
from pathlib import Path
import h5py
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import operator
from functools import reduce

#################################################
# Utilities
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### x: list of tensors
class MultipleTensors():
    def __init__(self, x):
        self.x = x

    def to(self, device):
        self.x = [x_.to(device) for x_ in self.x]
        return self

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item]


# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = True
        self.h5 = False
        self._load_file()

    def _load_file(self):

        if self.file_path[-3:] == '.h5':
            self.data = h5py.File(self.file_path, 'r')
            self.h5 = True
        else:
            try:
                self.data = scipy.io.loadmat(self.file_path)
            except:
                self.data = h5py.File(self.file_path, 'r')
                self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if self.h5:
            x = x[()]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x


def process_boundary_condition(boundary_path, ds_rate=4, method='sobel'):
    mask = torch.tensor(np.load(boundary_path))[None, None, :, :] # 1 1 512or128 512or128
    if 'rot' in boundary_path:
        boundary_mask = (mask == 1).float()
    else:
        boundary_mask = torch.rot90(mask == 1, dims=(2,3)).float()
    if not ('ds' in boundary_path or '128' in boundary_path):
        boundary_mask = F.max_pool2d(boundary_mask, ds_rate) # 1 1 128 128
    
    # sobel
    if method == 'sobel':
        sobel_x = torch.tensor([[1,0,-1], [2,0,-2], [1,0,-1]], dtype=torch.float)
        sobel_y = torch.tensor([[1,2,1], [0,0,0], [-1,-2,-1]], dtype=torch.float)
        sobel_grad = torch.abs(F.conv2d(boundary_mask, sobel_x[None,None,:,:], padding=1)) + \
                     torch.abs(F.conv2d(boundary_mask, sobel_y[None,None,:,:], padding=1))
        boundary = (sobel_grad > 0).float().squeeze() # 128 128
        domain = (1 - torch.max(boundary_mask.squeeze(), boundary))
    # conv_transpose
    elif method == 'deconv':
        deconv = F.conv_transpose2d(boundary_mask, torch.ones((1,1,3,3), dtype=torch.float))[:,:,1:-1,1:-1]
        boundary = ((deconv > 0).float() - boundary_mask).squeeze() # 128 128
        domain = (1 - (deconv > 0).float()).squeeze() # 128 128
    
    return boundary, domain


def get_land_sea_mask(data_path, mask_value):
    example_data_file = sorted(os.listdir(data_path))[0]
    example_data = np.load(os.path.join(data_path, example_data_file)) # c=5 h=180 w=300
    land = torch.from_numpy(example_data[0] <= mask_value).float()
    sea = 1 - land
    return land, sea


def load_constant_masks(npy_path=Path('./'), shape=(721, 1440), ds_rate=1, device='cpu'):
    npy_path = npy_path / 'constant_masks.npy'
    if not os.path.exists(npy_path):
        raise Exception(
            'Constant masks .npy not exist! Run convert_constant_masks_to_numpy in constant_masks.py first!')
    h_range = int(shape[0] // ds_rate * ds_rate)
    w_range = int(shape[1] // ds_rate * ds_rate)
    land_mask, soil_type, topography = [
        torch.tensor(arr[:h_range:ds_rate, :w_range:ds_rate], dtype=torch.float32, device=device)
        for arr in np.load(npy_path)]
    for mask in [land_mask, soil_type, topography]:
        h_ds, w_ds = mask.shape
        mask = mask.reshape(-1)
        mean_mask = mask.mean(-1, keepdim=True).detach()  # number
        mask -= mean_mask
        stdev_mask = torch.sqrt(torch.var(mask, dim=-1, keepdim=True, unbiased=False) + 1e-5)  # number
        mask /= stdev_mask
        mask = mask.reshape(h_ds, w_ds)
    return land_mask, soil_type, topography


# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, reduction=True, size_average=True, channel_wise=False):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.channel_wise = channel_wise

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1),
                                                          self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms
    
    def rel_channelwise(self, x, y):
        # x,y : b [z] h w c
        num_examples = x.size()[0]
        num_channels = x.size()[-1]

        diff_norms = torch.norm(x.reshape(num_examples, -1, num_channels) - y.reshape(num_examples, -1, num_channels), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1, num_channels), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel_channelwise(x, y) if self.channel_wise else self.rel(x, y)

def correct_acc_loss(x, y, mean, lat_range=(40,25), nlat=180):
    mean = torch.tensor(mean).unsqueeze(0).cuda()  # 1HWC
    lat_north, lat_south = lat_range
    weight_list = [np.cos((e / 180.0) * np.pi) for e in np.linspace(lat_north, lat_south, nlat+1)[:-1]]
    weight_list = torch.tensor(np.array(weight_list)).cuda()
    weight_sum = torch.sum(weight_list)
    weight_mask = (nlat * weight_list/weight_sum)[None, :, None, None]  # BHWC
    a = torch.mean(weight_mask * ((x - mean) * (y - mean)))
    b = torch.sqrt(
        torch.mean(weight_mask * (torch.abs(x - mean) ** 2)) * torch.mean(weight_mask * (torch.abs(y - mean) ** 2)))
    return a / b

class UncertaintyLoss(nn.Module):
    def __init__(self, num_losses, dtype=torch.float, device='cuda'):
        super(UncertaintyLoss, self).__init__()
        self.ln_sigma_2 = nn.Parameter(torch.ones(num_losses, dtype=dtype, device=device))

    def forward(self, losses):
        # tmp_exp = torch.clamp(self.ln_sigma_2, min=-5, max=5)
        # weighted_loss = torch.sum((0.5 * (torch.exp(-(2 * tmp_exp)) * losses) + tmp_exp))
        print(self.ln_sigma_2)
        print(losses, '\n')
        weighted_loss = torch.sum(1./2 * (torch.exp(-self.ln_sigma_2)*losses + self.ln_sigma_2))
        return weighted_loss


# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


class MMDStatistic:
    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.
    The kernel used is equal to:
    .. math ::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j\|x - x'\|^2},
    for the :math:`\alpha_j` proved in :py:meth:`~.MMDStatistic.__call__`.
    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        # The three constants used in the test.
        self.a00 = 1. / (n_1 * (n_1 - 1))
        self.a11 = 1. / (n_2 * (n_2 - 1))
        self.a01 = - 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        r"""Evaluate the statistic.
        The kernel used is
        .. math::
            k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},
        for the provided ``alphas``.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alphas : list of :class:`float`
            The kernel parameters.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.MMDStatistic.pval`.
        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)

        kernels = None
        for alpha in alphas:
            kernels_a = torch.exp(- alpha * distances ** 2)
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        k_1 = kernels[:self.n_1, :self.n_1]
        k_2 = kernels[self.n_1:, self.n_1:]
        k_12 = kernels[:self.n_1, self.n_1:]

        mmd = (2 * self.a01 * k_12.sum() +
               self.a00 * (k_1.sum() - torch.trace(k_1)) +
               self.a11 * (k_2.sum() - torch.trace(k_2)))
        if ret_matrix:
            return mmd, kernels
        else:
            return mmd

def vorticity(u, v):
    # u, v shape: b h w c
    # ret vor shape: b h-1 w-1 c
    pvpx = v[:, 1:, :-1] - v[:, :-1, :-1]
    pupy = u[:, :-1, 1:] - u[:, :-1, :-1]
    vor = pvpx - pupy
    return vor

def vorticity_3d(u, v, w):
    # u, v, w shape: b z h w c
    # ret vor shape: b z-1 h-1 w-1 c 3
    pwpy = w[:, :-1, :-1, 1:] - w[:, :-1, :-1, :-1]
    pvpz = v[:, 1:, :-1, :-1] - v[:, :-1, :-1, :-1]
    pupz = u[:, 1:, :-1, :-1] - u[:, :-1, :-1, :-1]
    pwpx = w[:, :-1, 1:, :-1] - w[:, :-1, :-1, :-1]
    pvpx = v[:, :-1, 1:, :-1] - v[:, :-1, :-1, :-1]
    pupy = u[:, :-1, :-1, 1:] - u[:, :-1, :-1, :-1]
    vorx = pwpy - pvpz
    vory = pupz - pwpx
    vorz = pvpx - pupy
    return torch.stack([vorx, vory, vorz], dim=-1)

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

def visual(grid, name):
    plt.clf()
    h, w = grid.shape
    # x = np.linspace(0, 1, h)
    # y = np.linspace(0, 1, w)
    
    # X, Y = np.meshgrid(x, y)
    
    # plt.contour(X, Y, grid)
    plt.imshow(grid)
    plt.savefig(name)
    plt.close()

def visual_zoy(field, name):
    size_z, size_x, size_y = field.shape
    plt.clf()
    plt.imshow(field[:,size_x//2,:])
    plt.savefig(name)
    plt.close()

def visual_traj_old(x_vis, coo_vis, n_pts=50, save_path='.'):
    # x_vis: num_layer+1 h w, coo_vis:num_layer k 2
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    num_layer, k, _ = coo_vis.shape
    n_pts = min(n_pts, k)
    idx = torch.randperm(k)[:n_pts]
    
    for i in range(num_layer):
        plt.clf()
        plt.imshow(x_vis[i])
        plt.scatter(coo_vis[i,idx,1], coo_vis[i,idx,0], c='r')
        plt.savefig(os.path.join(save_path, f'layer_{i}.png'))
    
    plt.clf()
    plt.imshow(x_vis[-1])
    plt.savefig(os.path.join(save_path, f'out.png'))
    
    plt.clf()
    plt.figure(figsize=(10,10))
    plt.imshow(x_vis[0])
    for i in range(n_pts):
        plt.plot(coo_vis[:,idx[i],1], coo_vis[:,idx[i],0], ls='-', marker='.')
    plt.savefig(os.path.join(save_path, f'traj.png'))

def visual_traj(padding, xs, coo_q, n_pts=100, save_path='.'):
    # xs: T(list) num_layers(list) h w, coo_q: T(list) num_layers(list) k 2
    print('visual traj')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if len(xs) == len(coo_q) + 1:
        xs = xs[1:]
        flag = 1
    else:
        flag = 0
    Tout = len(xs)
    num_layers = len(xs[0])

    # padding the background
    # ====
    # paddingw = (padding[1]//2, padding[1]-padding[1]//2)
    # xs = [[F.pad(x, paddingw, mode='replicate') for x in xt] for xt in xs]
    # xs = [[torch.cat((x[0:1, :].repeat(padding[0]//2,1), x), dim=0) for x in xt] for xt in xs]
    # xs = [[torch.cat((x, x[-1, :].repeat(padding[0]-padding[0]//2, 1)), dim=0) for x in xt] for xt in xs]
    # ====
    # pad = [padding[1]//2, padding[1]-padding[1]//2, padding[0]//2, padding[0]-padding[0]//2]
    # xs = [[F.pad(x[None, ...], [e//2**i for e in pad], mode='replicate')[0] for i,x in enumerate(xt)] for xt in xs]
    # ====
    for i in range(num_layers):
        v_h, v_w = xs[0][i].shape
        print(i, v_h, v_w)
        for j in range(Tout):
            coo_h, coo_w = int(coo_q[j][i][:,0].max().ceil()), int(coo_q[j][i][:,1].max().ceil())
            print(i, j, coo_h, coo_w)
            diff_h, diff_w = max(coo_h - v_h, 0), max(coo_w - v_w, 0)
            pad = [diff_w//2, diff_w-diff_w//2, diff_h//2, diff_h-diff_h//2]
            xs[j][i] = F.pad(xs[j][i][None,...], pad, mode='replicate')[0]

    ks = [coo_q[0][i].shape[0] for i in range(num_layers)] # total sample points in each layer
    n_pts_vis = [min(n_pts, ks[i]) for i in range(num_layers)]
    idx = [torch.randperm(ks[i])[:n_pts_vis[i]] for i in range(num_layers)] 

    for j in range(num_layers):
        for i in range(Tout):
            # if i == 0:
            #     print(f'layer {j}: bg shape {xs[i][j].shape}, coo shape {coo_q[i][j].shape},', \
            #             f'x range ({coo_q[i][j][idx[j],1].min()},{coo_q[i][j][idx[j],1].max()})', \
            #             f'y range ({coo_q[i][j][idx[j],0].min()},{coo_q[i][j][idx[j],0].max()})')
            plt.clf()
            plt.imshow(xs[i][j])
            plt.scatter(coo_q[i][j][idx[j],1], coo_q[i][j][idx[j],0], c='r')
            plt.savefig(os.path.join(save_path, f'layer_{j}_step_{i+1 if flag else i}.png'))
            plt.close()
    for i in range(num_layers):
        plt.clf()
        plt.figure(figsize=(10,10))
        plt.imshow(xs[0][i])
        xy = torch.stack([coo_q[t][i] for t in range(Tout)]) # t k 2
        for j in range(n_pts_vis[i]):
            plt.plot(xy[:,idx[i][j],1], xy[:,idx[i][j],0], ls='-', marker='.')
        plt.savefig(os.path.join(save_path, f'traj_layer_{i}.png'))
        plt.close()

def visual_traj_zoy(xs, coo_q, n_pts=50, save_path='.'):
    # xs: T(list) num_layers(list) z h w, coo_q: T(list) num_layers(list) k 3
    size_z, size_x, size_y = xs[0][0].shape
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if len(xs) == len(coo_q) + 1:
        xs = xs[1:]
        flag = 1
    else:
        flag = 0
    Tout = len(xs)
    num_layers = len(xs[0])
    ks = [coo_q[0][i].shape[0] for i in range(num_layers)] # total sample points in each layer
    n_pts_vis = [min(n_pts, ks[i]) for i in range(num_layers)]
    idx = [torch.randperm(ks[i])[:n_pts_vis[i]] for i in range(num_layers)]

    for j in range(num_layers):
        for i in range(Tout):
            half_x = xs[i][j].shape[1] // 2
            plt.clf()
            plt.imshow(xs[i][j][:,half_x,:])
            plt.scatter(coo_q[i][j][idx[j],2], coo_q[i][j][idx[j],0], c='r')
            plt.savefig(os.path.join(save_path, f'layer_{j}_step_{i+1 if flag else i}.png'))
            plt.close()
    for i in range(num_layers):
        plt.clf()
        plt.figure(figsize=(10,10))
        plt.imshow(xs[0][i][:,half_x,:])
        xy = torch.stack([coo_q[t][i] for t in range(Tout)]) # t k 2
        for j in range(n_pts_vis[i]):
            plt.plot(xy[:,idx[i][j],2], xy[:,idx[i][j],0], ls='-', marker='.')
        plt.savefig(os.path.join(save_path, f'traj_layer_{i}.png'))
        plt.close()

def visual_offset(velocity_norm, coo_offset_xyts, delta=20, save_path='.'):
    # velocity_norm: T num_layers h w, coo_offset_xyts: T num_layers h w 2
    print('visual offset')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    Tout = len(velocity_norm)
    num_layers = len(velocity_norm[0])
    # padding the background
    for i in range(num_layers):
        v_h, v_w = velocity_norm[0][i].shape
        ofst_h, ofst_w = coo_offset_xyts[0][i].shape[:-1]
        diff_h, diff_w = ofst_h - v_h, ofst_w - v_w
        pad = [diff_w//2, diff_w-diff_w//2, diff_h//2, diff_h-diff_h//2]
        for j in range(Tout-1):
            velocity_norm[j][i] = F.pad(velocity_norm[j][i][None,...], pad, mode='replicate')[0]
    size_layers = [velocity_norm[0][i].shape for i in range(num_layers)]
    X_layers = [np.arange(0, size_layer[0], 1) for size_layer in size_layers]
    Y_layers = [np.arange(0, size_layer[1], 1) for size_layer in size_layers]
    grid_layers = [np.stack(np.meshgrid(X, Y), axis=-1) for X, Y in zip(X_layers, Y_layers)]
    for j in range(num_layers):
        for i in range(Tout-1):
            # print('coo_offset_xy shape:', coo_offset_xyts[i][j].shape)
            coo_offset_xy = coo_offset_xyts[i][j].reshape(size_layers[j][0], size_layers[j][1], 2)
            # if i == 0:
            #     print(f'layer {j}: bg shape {velocity_norm[i][j].shape}, ofst shape {coo_offset_xy.shape},', \
            #             f'x range ({coo_offset_xy[:,1].min()},{coo_offset_xy[:,1].max()})', \
            #             f'y range ({coo_offset_xy[:,0].min()},{coo_offset_xy[:,0].max()})')
            plt.clf()
            plt.imshow(velocity_norm[i][j])
            # print('quiver param:', len(grid_layers[j]), grid_layers[j].shape)
            delta_layer = delta // 2**j
            plt.quiver(grid_layers[j][::delta_layer, ::delta_layer, 1], grid_layers[j][::delta_layer, ::delta_layer, 0], 
                       coo_offset_xy[::delta_layer, ::delta_layer, 1], coo_offset_xy[::delta_layer, ::delta_layer, 0])
            plt.savefig(os.path.join(save_path, f'offset_layer_{j}_step_{i}.png'))
            plt.close()

def visual_offset_lag(velocity_norm, coo_q, coo_offset_xyts, delta=20, save_path='.'):
    # velocity_norm: T num_layers h w, coo_q: t(list) n_l(list) k 2, coo_offset_xyts: t(list) n_l(list) k 2
    print(velocity_norm[0][0].shape)
    print(coo_q[0][0].shape,)
    print(coo_offset_xyts[0][0].shape)
    print('visual offset')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    Tout = len(velocity_norm)
    num_layers = len(velocity_norm[0])
    # padding the background
    for i in range(num_layers):
        v_h, v_w = velocity_norm[0][i].shape
        for j in range(Tout-1):
            ofst_h, ofst_w = [int(e) for e in coo_q[j][i].amax(dim=0).ceil()]
            diff_h, diff_w = max(ofst_h - v_h, 0), max(ofst_w - v_w, 0)
            pad = [diff_w//2, diff_w-diff_w//2, diff_h//2, diff_h-diff_h//2]
            velocity_norm[j][i] = F.pad(velocity_norm[j][i][None,...], pad, mode='replicate')[0]

    for j in range(num_layers):
        for i in range(Tout-1):
            # print('coo_offset_xy shape:', coo_offset_xyts[i][j].shape)
            # if i == 0:
            #     coo_offset_xy = coo_offset_xyts[i][j]
            #     print(f'layer {j}: bg shape {velocity_norm[i][j].shape}, ofst shape {coo_offset_xy.shape},', \
            #             f'x range ({coo_offset_xy[:,1].min()},{coo_offset_xy[:,1].max()})', \
            #             f'y range ({coo_offset_xy[:,0].min()},{coo_offset_xy[:,0].max()})')
            plt.clf()
            plt.imshow(velocity_norm[i][j])
            # print('quiver param:', len(grid_layers[j]), grid_layers[j].shape)
            delta_layer = max(delta // 2**j, 1)
            plt.quiver(coo_q[i][j][::delta_layer, 1], coo_q[i][j][::delta_layer, 0],
                       coo_offset_xyts[i][j][::delta_layer, 1], coo_offset_xyts[i][j][::delta_layer, 0])# , units='xy', scale=1)
            plt.savefig(os.path.join(save_path, f'offset_layer_{j}_step_{i}.png'))
            plt.close()

def visual_offset_lag_mask(velocity_norm, coo_q, coo_offset_xyts, mask, delta=20, save_path='.'):
    # velocity_norm: T num_layers h w, coo_q: t(list) n_l(list) k 2, coo_offset_xyts: t(list) n_l(list) k 2
    # print(velocity_norm[0][0].shape)
    # print(coo_q[0][0].shape,)
    # print(coo_offset_xyts[0][0].shape)
    # print('visual offset')
    # print(save_path)
    # print(mask.shape)
    mask = np.rot90(mask, k=3)
    # np.save(os.path.join(save_path, 'mask.npy'), mask)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    Tout = len(velocity_norm)
    num_layers = len(velocity_norm[0])
    # padding the background
    for i in range(num_layers):
        v_h, v_w = velocity_norm[0][i].shape
        for j in range(Tout-1):
            ofst_h, ofst_w = [int(e) for e in coo_q[j][i].amax(dim=0).ceil()]
            diff_h, diff_w = max(ofst_h - v_h, 0), max(ofst_w - v_w, 0)
            pad = [diff_w//2, diff_w-diff_w//2, diff_h//2, diff_h-diff_h//2]
            velocity_norm[j][i] = F.pad(velocity_norm[j][i][None,...], pad, mode='replicate')[0]

    for j in range(num_layers):
        for i in range(Tout-1):
            # print('coo_offset_xy shape:', coo_offset_xyts[i][j].shape)
            # if i == 0:
            #     coo_offset_xy = coo_offset_xyts[i][j]
            #     print(f'layer {j}: bg shape {velocity_norm[i][j].shape}, ofst shape {coo_offset_xy.shape},', \
            #             f'x range ({coo_offset_xy[:,1].min()},{coo_offset_xy[:,1].max()})', \
            #             f'y range ({coo_offset_xy[:,0].min()},{coo_offset_xy[:,0].max()})')
            plt.clf()
            plt.imshow(velocity_norm[i][j])
            # print('quiver param:', len(grid_layers[j]), grid_layers[j].shape)
            delta_layer = max(delta // 2**j, 1)
            coo_q_y = coo_q[i][j][::delta_layer, 1]
            coo_q_x = coo_q[i][j][::delta_layer, 0]
            np.save(os.path.join(save_path, 'offset_x.npy'), coo_q_x)
            np.save(os.path.join(save_path, 'offset_y.npy'), coo_q_y)
            coo_offset_y = coo_offset_xyts[i][j][::delta_layer, 1] / mask.shape[-1] * 12
            coo_offset_x = coo_offset_xyts[i][j][::delta_layer, 0] / mask.shape[-2] * 12
            coo_offset_y = coo_offset_y * (2 * (coo_q_y > 32)-1)
            coo_offset_x = coo_offset_x * (2 * (coo_q_y > 32)-1)
            # print(coo_offset_y, 'offset_y')
            # print(coo_offset_x, 'offset_x')
            # print(coo_q_y, 'q_y')
            # print(coo_q_x, 'q_x')
            # assert 1 == 0
            after_q_y = coo_q_y + coo_offset_y
            after_q_x = coo_q_x + coo_offset_x
            after_q_y = after_q_y.long()
            after_q_x = after_q_x.long()

            mask_all = (coo_q_x >= 0) * (coo_q_x < mask.shape[-2]) * (coo_q_y >= 0) * (coo_q_y < mask.shape[-1]) \
                       * (after_q_x >= 0) * (after_q_x < mask.shape[-2]) * (after_q_y >= 0) * (after_q_y < mask.shape[-1])
            # print((mask_all*1.0).sum())
            coo_q_y = coo_q_y[mask_all]
            coo_q_x = coo_q_x[mask_all]
            coo_offset_y = coo_offset_y[mask_all]
            coo_offset_x = coo_offset_x[mask_all]
            after_q_y = after_q_y[mask_all]
            after_q_x = after_q_x[mask_all]
            # print(coo_q_y, 'q_y')
            # print(coo_q_x, 'q_x')
            mask_all = (mask[coo_q_x.long(), coo_q_y.long()] == 1) * (mask[after_q_x, after_q_y] == 1)
            # print((mask_all*1.0).sum(), 'mask_all2')
            # print(coo_q_y[mask_all], 'q_y')
            # print(coo_q_x[mask_all], 'q_x')
            # assert 1 == 0
            # plt.quiver(coo_q[i][j][::delta_layer, 1], coo_q[i][j][::delta_layer, 0],
            #            coo_offset_xyts[i][j][::delta_layer, 1], coo_offset_xyts[i][j][::delta_layer, 0])
            plt.quiver(coo_q_y[mask_all], coo_q_x[mask_all], coo_offset_y[mask_all], coo_offset_x[mask_all], units='xy', scale=1.8)
            
            # plt.quiver(coo_q_y, coo_q_x, coo_offset_y, coo_offset_x, units='xy', scale=1)
            plt.savefig(os.path.join(save_path, f'offset_layer_{j}_step_{i}.png'))
            plt.close()
            # assert 1 == 0
