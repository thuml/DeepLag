import os
from timeit import default_timer
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from utils.data_factory import get_bc_dataset, BoundedNSDataset
from utils.utilities3 import *
from utils.params import get_args
from utils.adam import Adam
from model_dict import get_model

from tqdm import tqdm

time_str = (datetime.now()).strftime("%Y%m%d_%H%M%S")

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


################################################################
# configs
################################################################
args = get_args(time=time_str)

TRAIN_PATH = os.path.join(args.data_path, 're0.25_4c_gray_10000.npy')
TEST_PATH = os.path.join(args.data_path, 're0.25_4c_gray_10000.npy')
BOUNDARY_PATH = os.path.join(args.data_path, 'boundary_4c_rot.npy')

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
T_out = args.T_out
patch_size = tuple(int(x) for x in args.patch_size.split(','))

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma
delta_t = args.delta_t

model_save_path = args.model_save_path
model_save_name = args.model_save_name


################################################################
# models
################################################################
model = get_model(args)


################################################################
# load data and data normalization
################################################################
train_dataset = BoundedNSDataset(args, dataset_file=TRAIN_PATH, split='train',delta_t=delta_t)
test_dataset = BoundedNSDataset(args, dataset_file=TEST_PATH, split='test',delta_t=delta_t)
train_loader = train_dataset.loader()
test_loader = test_dataset.loader()

boundary, domain = process_boundary_condition(BOUNDARY_PATH, ds_rate=(r1,r2))

if 'DeepLag' in args.model:
    model.set_bdydom(boundary, domain)
    if args.resample_strategy == 'uniform' or args.resample_strategy == 'learned':
        model.num_samples = min(model.num_samples, s1*s2)
    elif args.resample_strategy == 'boundary':
        model.num_samples = min(model.num_samples, model.coo_boundary_ms[0].shape[0])
    elif args.resample_strategy == 'domain':
        model.num_samples = min(model.num_samples, model.coo_domain_ms[0].shape[0])


################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)

step = 1
min_test_l2_full = 114514
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for batch_idx, (xx, yy) in enumerate(tqdm(train_loader)):
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        if 'DeepLag' in args.model:
            h_x_q, h_coo_q, h_coo_offset_q = [], [], []
            for i in range(model.num_layers):
                if args.resample_strategy == 'uniform':
                    num_samples = model.num_samples // (4**i)
                    coo_q = torch.cat([
                        torch.randint(0,model.img_h_layers[i]-1,(batch_size,num_samples,1)), 
                        torch.randint(0,model.img_w_layers[i]-1,(batch_size,num_samples,1))
                    ], dim=-1).to(torch.float32)
                elif args.resample_strategy == 'boundary':
                    num_samples = min(model.num_samples//(4**i), model.coo_boundary_ms[i].shape[0])
                    idx_coo_sample = torch.multinomial(1./torch.ones(model.coo_boundary_ms[i].shape[0]), num_samples, replacement=False) # k
                    coo_q = model.coo_boundary_ms[i][idx_coo_sample][None, ...].repeat(batch_size,1,1).to(torch.float32) # b k 2
                elif args.resample_strategy == 'domain':
                    num_samples = min(model.num_samples//(4**i), model.coo_domain_ms[i].shape[0])
                    idx_coo_sample = torch.multinomial(1./torch.ones(model.coo_domain_ms[i].shape[0]), num_samples, replacement=False) # k
                    coo_q = model.coo_domain_ms[i][idx_coo_sample][None, ...].repeat(batch_size,1,1).to(torch.float32) # b k 2
                elif args.resample_strategy == 'learned':
                    num_samples = model.num_samples // (4**i)
                    coo_q = None # new_prob
                num_chan = args.d_model*(2**i) if i < model.num_layers-1 else args.d_model*(2**(i-1))
                h_x_q.append(torch.zeros(batch_size, num_samples, num_chan).to(device))
                h_coo_q.append(coo_q.to(device) if args.resample_strategy != 'learned' else None) # new_prob
                h_coo_offset_q.append(torch.zeros(batch_size, num_samples, 2).to(device))

        for t in range(0, T_out, step):
            y = yy[..., t*args.out_var : (t + step)*args.out_var]
            if 'DeepLag' in args.model:
                im, h_x_q, h_coo_q, h_coo_offset_q, coo_offset_xys = model(xx, h_x_q, h_coo_q, h_coo_offset_q) # B H W C_out=T_out_step
            else:
                im = model(xx)
            
            # print(xx.shape, y.shape)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step*args.in_var:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    t2 = default_timer()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for batch_idx, (xx, yy) in enumerate(test_loader):
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            if 'DeepLag' in args.model:
                h_x_q, h_coo_q, h_coo_offset_q = [], [], []
                for i in range(model.num_layers):
                    if args.resample_strategy == 'uniform':
                        num_samples = model.num_samples // (4**i)
                        coo_q = torch.cat([
                            torch.randint(0,model.img_h_layers[i]-1,(batch_size,num_samples,1)), 
                            torch.randint(0,model.img_w_layers[i]-1,(batch_size,num_samples,1))
                        ], dim=-1).to(torch.float32) # b k 2
                    elif args.resample_strategy == 'boundary':
                        num_samples = min(model.num_samples//(4**i), model.coo_boundary_ms[i].shape[0])
                        idx_coo_sample = torch.multinomial(1./torch.ones(model.coo_boundary_ms[i].shape[0]), num_samples, replacement=False) # k
                        coo_q = model.coo_boundary_ms[i][idx_coo_sample][None, ...].repeat(batch_size,1,1).to(torch.float32) # b k 2
                    elif args.resample_strategy == 'domain':
                        num_samples = min(model.num_samples//(4**i), model.coo_domain_ms[i].shape[0])
                        idx_coo_sample = torch.multinomial(1./torch.ones(model.coo_domain_ms[i].shape[0]), num_samples, replacement=False) # k
                        coo_q = model.coo_domain_ms[i][idx_coo_sample][None, ...].repeat(batch_size,1,1).to(torch.float32) # b k 2
                    elif args.resample_strategy == 'learned':
                        num_samples = model.num_samples // (4**i)
                        coo_q = None # new_prob
                    num_chan = args.d_model*(2**i) if i < model.num_layers-1 else args.d_model*(2**(i-1))
                    h_x_q.append(torch.zeros(batch_size, num_samples, num_chan).to(device))
                    h_coo_q.append(coo_q.to(device) if args.resample_strategy != 'learned' else None) # new_prob
                    h_coo_offset_q.append(torch.zeros(batch_size, num_samples, 2).to(device)) 

            for t in range(0, T_out, step):
                y = yy[..., t*args.out_var : (t + step)*args.out_var]
                if 'DeepLag' in args.model:
                    im, h_x_q, h_coo_q, h_coo_offset_q, coo_offset_xys = model(xx, h_x_q, h_coo_q, h_coo_offset_q) # B H W C_out=T_out_step, h_coo_q: n_l(list) b k 2
                else:
                    im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step*args.in_var:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    # t2 = default_timer()
    scheduler.step()
    if test_l2_full / ntest < min_test_l2_full:
        min_test_l2_full = test_l2_full / ntest
        print(ep, t2 - t1, train_l2_step / ntrain / (T_out / step), train_l2_full / ntrain,
            test_l2_step / ntest / (T_out / step),
            test_l2_full / ntest, 'new_best!')
        print('save best model')
        torch.save(model.state_dict(), os.path.join(args.run_save_path, model_save_name[:-3]+f'_best.pt'))
        pd = pred[-1, :, :, -1].detach().cpu().numpy()
        gt = yy[-1, :, :, -1].detach().cpu().numpy()
        visual(pd, os.path.join(args.run_save_path, f'best_pred.png'))
        visual(gt, os.path.join(args.run_save_path, f'best_gt.png'))
        visual(np.abs(gt-pd), os.path.join(args.run_save_path, f'best_err.png'))
    else:
        print(ep, t2 - t1, train_l2_step / ntrain / (T_out / step), train_l2_full / ntrain,
            test_l2_step / ntest / (T_out / step),
            test_l2_full / ntest)
    if ep % 10 == 0:
        # if not os.path.exists(model_save_path):
        #     os.makedirs(model_save_path)
        print('save latest model')
        torch.save(model.state_dict(), os.path.join(args.run_save_path, model_save_name[:-3]+f'_latest.pt'))
    if ep % 100 == 0:
        pd = pred[-1, :, :, -1].detach().cpu().numpy()
        gt = yy[-1, :, :, -1].detach().cpu().numpy()
        visual(pd, os.path.join(args.run_save_path, f'ep_{ep}_pred.png'))
        visual(gt, os.path.join(args.run_save_path, f'ep_{ep}_gt.png'))
        visual(np.abs(gt-pd), os.path.join(args.run_save_path, f'ep_{ep}_err.png'))
