import os
from pathlib import Path
import logging
import pickle
from timeit import default_timer
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from utils.data_factory import SmokeDataset, SmokeDatasetMemory
from utils.utilities3 import *
from utils.params import get_args, get_test_args
from utils.adam import Adam
from model_dict import get_model

from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


################################################################
# configs
################################################################
test_args = get_test_args()
ckpt_dir = test_args.ckpt_dir
dataset_nickname = test_args.dataset_nickname
model_name = test_args.model_name
time_str = test_args.time_str
milestone = test_args.milestone
T_out = test_args.T_out

args = get_args(cfg_file=Path(ckpt_dir)/dataset_nickname/model_name/time_str/'configs.txt')
test_save_path = os.path.join(args.run_save_path, f'test_{milestone}_{T_out}')
if not os.path.isdir(test_save_path):
    os.makedirs(test_save_path)

LOG_FORMAT = "%(message)s"
logger = logging.getLogger('Loss logger')
logger.setLevel(logging.INFO)
f_handler = logging.FileHandler(os.path.join(test_save_path, args.log_save_name))
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(f_handler)

padding = [int(p) for p in args.padding.split(',')]
ntrain = args.ntrain
ntest = args.ntest
N = args.ntotal
args.in_channels = args.in_dim * args.in_var
args.out_channels = args.out_dim * args.out_var
r1 = args.z_down
r2 = args.h_down
r3 = args.w_down
s1 = int(((args.z - 1) / r1) + 1)
s2 = int(((args.h - 1) / r2) + 1)
s3 = int(((args.w - 1) / r3) + 1)
T_in = args.T_in
# T_out = args.T_out
patch_size = tuple(int(x) for x in args.patch_size.split(','))

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma

model_save_path = args.model_save_path
model_save_name = args.model_save_name


################################################################
# models
################################################################
model = get_model(args, ckpt_dir=Path(ckpt_dir)/dataset_nickname/model_name/time_str)
state_dict = torch.load(Path(ckpt_dir)/dataset_nickname/model_name/time_str/ (model_save_name[:-3]+f'_{milestone}.pt'))
model.load_state_dict(state_dict)


################################################################
# load data and data normalization
################################################################
train_dataset = SmokeDatasetMemory(args, split='train')
test_dataset = SmokeDatasetMemory(args, split='test')
train_loader = train_dataset.loader()
test_loader = test_dataset.loader()

boundary = torch.ones(s1, s2, s3)
boundary[1:-1, 1:-1, 1:-1] = 0
domain = 1 - boundary.clone().detach()
if 'DeepLag' in args.model:
    model.set_bdydom(boundary, domain)
    if args.resample_strategy == 'uniform' or args.resample_strategy == 'learned':
        model.num_samples = min(model.num_samples, s1*s2*s3//model.pixel_per_patch)
    elif args.resample_strategy == 'boundary':
        model.num_samples = min(model.num_samples, model.coo_boundary_ms[0].shape[0])
    elif args.resample_strategy == 'domain':
        model.num_samples = min(model.num_samples, model.coo_domain_ms[0].shape[0])


################################################################
# evaluation
################################################################
myloss = LpLoss(size_average=False, channel_wise=False)
mseloss = nn.MSELoss()

step = 1
min_test_l2_full = 114514

t1 = default_timer()
test_l2_step = 0
test_l2_full = 0
test_vor_step = 0
test_vor_full = 0
with torch.no_grad():
    for batch_idx, (xx, yy) in enumerate(tqdm(test_loader)):
        loss = 0
        vor_loss = 0
        xx = xx.to(device) # B Z H W T*C
        yy = yy.to(device) # B Z H W T*C
        if 'DeepLag' in args.model:
            h_x_q, h_coo_q, h_coo_offset_q = [], [], []
            for i in range(model.num_layers):
                if args.resample_strategy == 'uniform':
                    num_samples = model.num_samples // (8**i)
                    coo_q = torch.cat([
                        torch.randint(0,model.img_z_layers[i]-1,(batch_size,num_samples,1)), 
                        torch.randint(0,model.img_h_layers[i]-1,(batch_size,num_samples,1)), 
                        torch.randint(0,model.img_w_layers[i]-1,(batch_size,num_samples,1))
                    ], dim=-1).to(torch.float32) # b k 3
                elif args.resample_strategy == 'boundary':
                    num_samples = min(model.num_samples//(8**i), model.coo_boundary_ms[i].shape[0])
                    idx_coo_sample = torch.multinomial(1./torch.ones(model.coo_boundary_ms[i].shape[0]), num_samples, replacement=False) # k
                    coo_q = model.coo_boundary_ms[i][idx_coo_sample][None, ...].repeat(batch_size,1,1).to(torch.float32) # b k 3
                elif args.resample_strategy == 'domain':
                    num_samples = min(model.num_samples//(8**i), model.coo_domain_ms[i].shape[0])
                    idx_coo_sample = torch.multinomial(1./torch.ones(model.coo_domain_ms[i].shape[0]), num_samples, replacement=False) # k
                    coo_q = model.coo_domain_ms[i][idx_coo_sample][None, ...].repeat(batch_size,1,1).to(torch.float32) # b k 3
                elif args.resample_strategy == 'learned':
                    num_samples = model.num_samples // (8**i)
                    coo_q = None
                num_chan = args.d_model*(2**i) if i < model.num_layers-1 else args.d_model*(2**(i-1))
                h_x_q.append(torch.zeros(batch_size, num_samples, num_chan).to(device))
                h_coo_q.append(coo_q.to(device) if args.resample_strategy != 'learned' else None)
                h_coo_offset_q.append(torch.zeros(batch_size, num_samples, 3).to(device))

        for t in range(0, T_out, step):
            y = yy[..., t*args.out_var : (t + step)*args.out_var] # B Z H W C_out=V_out
            if 'DeepLag' in args.model:
                im, h_x_q, h_coo_q, h_coo_offset_q, coo_offset_zxys = model(xx, h_x_q, h_coo_q, h_coo_offset_q) # B Z H W C_out=V_out
            else:
                im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
            vor_loss += mseloss(vorticity_3d(im[..., -3], im[..., -2], im[..., -1]), vorticity_3d(-y[..., -3], y[..., -2], y[..., -1]))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step*args.in_var:], im), dim=-1)

        test_l2_step += loss.item()
        test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
        test_vor_step += vor_loss.item()
        test_vor_full += mseloss(vorticity_3d(pred[..., 1::args.in_var], pred[..., 2::args.in_var], pred[..., 3::args.in_var]), vorticity_3d(yy[..., 1::args.in_var], yy[..., 2::args.in_var], yy[..., 3::args.in_var])).item()

t2 = default_timer()
if test_l2_full / ntest < min_test_l2_full:
    print(t2 - t1,
        'test_rel_l2:', 
        test_l2_step / ntest / (T_out / step), 
        test_l2_full / ntest, 
        'test_vor:', 
        test_vor_step / ntest / (T_out / step),
        test_vor_full / ntest)
    logger.info(f'{t2 - t1} ' + \
                f'test_rel_l2: {test_l2_step / ntest / (T_out / step)} {test_l2_full / ntest}  ' + \
                f'test_vor: {test_vor_step / ntest / (T_out / step)} {test_vor_full / ntest}')
    pd = pred[-1, :, :, :, -4:].detach().cpu().numpy()
    gt = yy[-1, :, :, :, -4:].detach().cpu().numpy()
    vars = ['field', 'ux', 'uy', 'uz']
    for i in range(4):
        visual_zoy(pd[...,i], os.path.join(test_save_path, f'{milestone}_{vars[i]}_pred.png'))
        visual_zoy(gt[...,i], os.path.join(test_save_path, f'{milestone}_{vars[i]}_gt.png'))
        visual_zoy(np.abs(gt-pd)[...,i], os.path.join(test_save_path, f'{milestone}_{vars[i]}_err.png'))
else:
    raise Exception('Abnormal loss!')
