import os
from pathlib import Path
import logging
import pickle
from timeit import default_timer
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from utils.data_factory import SeaDataset, SeaDatasetMemory
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
r1 = args.h_down
r2 = args.w_down
s1 = int(((args.h - 1) / r1) + 1)
s2 = int(((args.w - 1) / r2) + 1)
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
train_dataset = SeaDatasetMemory(args, region=args.region, split='train', return_idx=True)
test_dataset = SeaDatasetMemory(args, region=args.region, split='test', return_idx=True)
train_loader = train_dataset.loader()
test_loader = test_dataset.loader()

land, sea = get_land_sea_mask(args.data_path, args.fill_value)

data_mean = np.load(Path(args.data_path)/'..'/f'sea_{args.region}_mean.npy')


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
test_acc_step = torch.zeros(T_out//step).to(device)
test_acc_full = torch.zeros(T_out//step).to(device)
with torch.no_grad():
    for batch_idx, (index, xx, yy) in enumerate(tqdm(test_loader)):
        index = index.to(device) * 0.01
        loss = 0
        vor_loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        xx /= 256.0
        yy /= 256.0

        for i, t in enumerate(range(0, T_out, step)):
            y = yy[..., t*args.out_var : (t + step)*args.out_var] # B H W C_out=V_out
            im, _ = model(xx, index)
            im *= 256.0
            loss += myloss(im, y)
            vor_loss += mseloss(vorticity(-im[..., -2], im[..., -3]), vorticity(-y[..., -2], y[..., -3]))
            test_acc_step[i] += correct_acc_loss(im, y, data_mean)

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step*args.in_var:], im), dim=-1)

        test_l2_step += loss.item()
        test_l2_full += myloss(pred, yy).item()
        test_vor_step += vor_loss.item()
        test_vor_full += mseloss(vorticity(-pred[..., 3::args.in_var], pred[..., 2::args.in_var]), vorticity(-yy[..., 3::args.in_var], yy[..., 2::args.in_var])).item()
        for i, t in enumerate(range(0, T_out, step)):
            test_acc_full[i] += correct_acc_loss(pred[..., i*args.in_var:(i+1)*args.in_var], yy[..., i*args.in_var:(i+1)*args.in_var], data_mean)

t2 = default_timer()
if test_l2_full / ntest < min_test_l2_full:
    print(t2 - t1,
        'test_rel_l2:', 
        test_l2_step / ntest / (T_out / step), 
        test_l2_full / ntest, 
        'test_vor:', 
        test_vor_step / ntest / (T_out / step),
        test_vor_full / ntest, 
        'test_acc:', 
        test_acc_step / ntest,
        test_acc_full / ntest)
    logger.info(f'{t2 - t1} ' + \
                f'test_rel_l2: {test_l2_step / ntest / (T_out / step)} {test_l2_full / ntest}  ' + \
                f'test_vor: {test_vor_step / ntest / (T_out / step)} {test_vor_full / ntest}  ' + \
                f'test_acc: {test_acc_step / ntest} {test_acc_full / ntest}')
    pd = pred[-1, :, :, -5:].detach().cpu().numpy()
    gt = yy[-1, :, :, -5:].detach().cpu().numpy()
    vars = ['thetao', 'so', 'uo', 'vo', 'zos']
    for i in range(5):
        visual(pd[...,i], os.path.join(test_save_path, f'{milestone}_{vars[i]}_pred.png'))
        visual(gt[...,i], os.path.join(test_save_path, f'{milestone}_{vars[i]}_gt.png'))
        visual(np.abs(gt-pd)[...,i], os.path.join(test_save_path, f'{milestone}_{vars[i]}_err.png'))
else:
    raise Exception('Abnormal loss!')
