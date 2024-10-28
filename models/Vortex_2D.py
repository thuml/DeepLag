from .libs.vortex.io_utils import *
from .libs.vortex.simulation_utils import *
from .libs.vortex.learning_utils import L2_Loss, vort_to_vel
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
torch.manual_seed(123)
import sys
import os

from functorch import jacrev, vmap
import torch
class SineResidualBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super(SineResidualBlock, self).__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # add shortcut
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
            )

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        out = torch.sin(self.omega_0 * self.linear(input))
        out += self.shortcut(input)
        out = nn.functional.relu(out)
        return out

class Dynamics_Net(nn.Module):
    def __init__(self):
        super(Dynamics_Net, self).__init__()
        in_dim = 1
        out_dim = 1
        width = 40
        self.layers = nn.Sequential(SineResidualBlock(in_dim, width, omega_0=1., is_first=True),
                                SineResidualBlock(width, width, omega_0=1.),
                                SineResidualBlock(width, width, omega_0=1.),
                                SineResidualBlock(width, width, omega_0=1.),
                                nn.Linear(width, out_dim),
                                )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

class Position_Net(nn.Module):
    def __init__(self, num_vorts):
        super(Position_Net, self).__init__()
        in_dim = 1
        out_dim = num_vorts * 2
        self.layers = nn.Sequential(SineResidualBlock(in_dim, 64, omega_0=1., is_first=True),
                                SineResidualBlock(64, 128, omega_0=1.),
                                SineResidualBlock(128, 256, omega_0=1.),
                                nn.Linear(256, out_dim)
                                )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.ckptdir = args.ckptdir if hasattr(args, 'ckptdir') else 'checkpoints'
        self.num_vorts = args.num_vorts if hasattr(args, 'num_vorts') else 16
        self.decay_gamma = args.decay_gamma if hasattr(args, 'decay_gamma') else 0.99
        self.decimate_point = args.decimate_point if hasattr(args,'decimate_point') else 20000 # LR decimates at this point
        self.decay_step = max(1, int(self.decimate_point/math.log(0.1, self.decay_gamma))) # decay once every (# >= 1) learning steps
        self.pre_ckptdir = args.pre_ckptdir if hasattr(args, 'pre_ckptdir') else './models/libs/vortex/pretrained.tar'
        self.width = int(((args.w - 1) / args.w_down) + 1)
        self.height = int(((args.h - 1) / args.h_down) + 1)
        self.C_out = args.out_dim * args.out_var
        self.device = torch.device('cuda')
        #self.net_dict, self.start, self.grad_vars, self.optimizer, self.lr_scheduler = create_bundle(self.ckptdir, self.num_vorts, self.decay_step, self.decay_gamma, pretrain_dir = self.pre_ckptdir)
        self.img_x = gen_grid(self.width, self.height, device) # grid coordinates'
        self.batch_size = args.batch_size
        self.vort_scale = args.vort_scale if hasattr(args, 'vort_scale') else 0.5
        self.num_sims = 1
        self.net_dict_len = Dynamics_Net()
        self.net_dict_pos = Position_Net(self.num_vorts)
        pre_ckpt = torch.load(self.pre_ckptdir)
        self.net_dict_pos.load_state_dict(pre_ckpt['model_pos_state_dict'])
        self.register_parameter('w_pred_param', nn.Parameter(torch.zeros(self.num_vorts, 1, dtype=torch.float32)))
        self.register_parameter('size_pred_param', nn.Parameter(torch.zeros(self.num_vorts, 1,dtype=torch.float32)))
    def eval_vel(self, vorts_size, vorts_w, vorts_pos, query_pos):
        return vort_to_vel(self.net_dict_len, vorts_size, vorts_w, vorts_pos, query_pos, length_scale = self.vort_scale)

    def dist_2_len_(self, dist):
        return self.net_dict_len(dist)

    def size_pred(self):
        pred = self.size_pred_param
        size =  0.03 + torch.sigmoid(pred)
        return size

    def w_pred(self):    
        pred = self.w_pred_param
        w = torch.sin(pred)
        return w

    def comp_velocity(self, timestamps):
        jac = vmap(jacrev((self.net_dict_pos)))(timestamps)
        post = jac[:, :, 0:1].view((timestamps.shape[0],-1,2,1))
        xt = post[:, :, 0, :]
        yt = post[:, :, 1, :]
        uv = torch.cat((xt, yt), dim = 2)
        return uv

    def forward(self, x, index):
        #x: b h w c
        index = index.unsqueeze(1).float()
        with torch.no_grad():
            pos_pred_gradless = self.net_dict_pos(index).view((-1,self.num_vorts,2))
            D_vel = self.eval_vel(self.size_pred(), self.w_pred(), pos_pred_gradless, pos_pred_gradless)

            # if boundary is not None:
            #     D_vel = boundary_treatment(pos_pred_gradless, D_vel, boundary, mode = 1)

        # velocity loss
        T_vel = self.comp_velocity(index) # velocity prescribed by trajectory module
        vel_loss = 0.001 * L2_Loss(T_vel, D_vel)
        pos_pred = self.net_dict_pos(index).view((self.batch_size,self.num_vorts,2))
        sim_imgs, sim_vorts_poss, sim_img_vels, sim_vorts_vels = simulate(x.clone(), self.img_x, pos_pred, self.w_pred(), \
                            self.size_pred(), self.num_sims, vel_func = self.eval_vel, boundary = None)
        return sim_imgs[-1][:, :, :, :self.C_out], vel_loss