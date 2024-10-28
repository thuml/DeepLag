from datetime import datetime, timedelta
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange, repeat


def get_bc_dataset(args, dataset_file):
    processed_data_path = os.path.splitext(dataset_file)[0] + '_processed.npy'
    if os.path.exists(processed_data_path):
        dataset = np.load(processed_data_path)
    else:
        data = np.load(dataset_file)[:, ::args.h_down, ::args.w_down] # 10000, args.h/args.h_down, args.w/args.w_down
        dataset = []
        for i in range(args.ntotal): 
            dataset.append(data[i:i + args.T_in + args.T_out, :, :].transpose(1, 2, 0))
        dataset = np.stack(dataset, axis=0)
        np.save(processed_data_path, dataset)
    dataset = torch.from_numpy(dataset.astype(np.float32))
    return dataset


class BoundedNSDataset(Dataset):
    def __init__(self, args, dataset_file, split, delta_t = 1, return_idx = False):
        self.data_path = args.data_path # /data/Bounded_NS
        self.dataset_file = dataset_file
        self.h, self.w = args.h, args.w
        self.h_down, self.w_down = args.h_down, args.w_down
        self.batch_size = args.batch_size
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.ntotal = args.ntotal
        self.delta_t = delta_t
        self.return_idx = return_idx

        self.split = split
        if split == 'train':
            self.length = args.ntrain
        elif split == 'test':
            self.length = args.ntest
        else:
            self.length = args.ntotal
        
        self.dataset = self._load_dataset()
    
    def _load_dataset(self):
        dataset = np.load(self.dataset_file)[:self.ntotal, ::self.h_down, ::self.w_down] # ntotal, h/h_down, w/w_down
        dataset = torch.from_numpy(dataset.astype(np.float32))
        print(f'Loaded dataset from {self.dataset_file}')
        return dataset
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if self.split == 'train':
            idx = index
        elif self.split == 'test':
            idx = self.ntrain + index
        else:
            raise Exception('split must be train or test')
        input_x = rearrange(self.dataset[idx:idx + self.T_in*self.delta_t:self.delta_t, :, :], 't h w -> h w t')
        input_y = rearrange(self.dataset[idx + self.T_in*self.delta_t:idx + self.T_in*self.delta_t + self.T_out*self.delta_t:self.delta_t, :, :], 't h w -> h w t')
        if not self.return_idx:
            return input_x, input_y # B H W T
        else:
            return index, input_x, input_y
    
    def loader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=True if self.split=='train' else False)


class SeaDataset(Dataset):
    '''
    ['thetao', 'so', 'uo', 'vo', 'zos'] (b, 5, 180, 300)
    thetao: sea_water_potential_temperature, range [4.537, 26.204]
    so: sea_water_salinity, range [32.241, 35.263]
    uo: eastward_sea_water_velocity, range [-0.893, 1.546]
    vo: northward_sea_water_velocity, range [-1.646, 1.088]
    zos: sea_surface_height_above_geoid, range [-0.342, 1.511]
    '''
    def __init__(self, args, region='kuroshio', split='train', var=None):
        self.data_path = args.data_path # /data/sea_data_small/data_sea
        self.region = region
        self.data_files = sorted([data_file for data_file in os.listdir(self.data_path) if region in data_file])
        self.h, self.w = args.h, args.w
        self.h_down, self.w_down = args.h_down, args.w_down
        self.fill_value = args.fill_value
        self.batch_size = args.batch_size
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.ntrain = args.ntrain
        var_dict = {'thetao':0, 'so':1, 'uo':2, 'vo':3, 'zos':4}
        if var is not None and var not in var_dict.keys():
            raise Exception('var must be None or one of [\'thetao\', \'so\', \'uo\', \'vo\', \'zos\']')
        elif var is not None:
            self.var = var
            self.var_idx = var_dict[var]
        else:
            self.var = None
            self.var_idx = None

        self.split = split
        if split == 'train':
            self.length = args.ntrain
        elif split == 'test':
            self.length = args.ntest
        else:
            self.length = args.ntotal
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.split == 'train':
            index_tmp = index
        elif self.split == 'test':
            index_tmp = self.ntrain + index
        else:
            raise Exception('split must be train or test when getting file as single data file')
        
        input_x, input_y = [], []
        for i in range(self.T_in + self.T_out):
            data_file = Path(self.data_path) / self.data_files[index_tmp + i]
            data = np.load(data_file).astype(np.float32)[:, :self.h:self.h_down, :self.w:self.w_down] # v=5 h=180/d w=300/d 
            data[data <= self.fill_value] = 0.
            data[1][data[1] == 0] = 30. # so \in (32.241, 35.263)
            if i < self.T_in:
                input_x.append(torch.from_numpy(data))
            else:
                input_y.append(torch.from_numpy(data))
        if self.var_idx is None:
            input_x = torch.stack(input_x, dim=-1) # v h w t=10
            input_y = torch.stack(input_y, dim=-1)
            input_x = rearrange(input_x, 'v h w t -> h w (t v)')
            input_y = rearrange(input_y, 'v h w t -> h w (t v)')
        else:
            input_x = torch.stack(input_x, dim=-1)[self.var_idx] # h w t=10
            input_y = torch.stack(input_y, dim=-1)[self.var_idx]
        return input_x, input_y # B H W T*V
    
    def loader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=True if self.split=='train' else False)
    
    def get_tensor_dataset(self):
        if self.split == 'train' or self.split == 'test':
            raise Exception('Tensor dataset as a whole, not split for train or test')
        if self.var_idx is None:
            dataset_npy_path = Path(self.data_path) /'..'/f'sea_{self.region}.npy'
        else:
            dataset_npy_path = Path(self.data_path) /'..'/f'sea_{self.region}_{self.var}.npy'
        
        if os.path.exists(dataset_npy_path):
            dataset = np.load(dataset_npy_path)
        else:
            data_list = []
            for i in range(len(self.data_files)):
                data_file = Path(self.data_path) / self.data_files[i]
                data_frame = np.load(data_file)[:, :self.h:self.h_down, :self.w:self.w_down]\
                    .transpose(1,2,0).astype(np.float32) # h=180/d w=300/d v=5
                data_frame[data_frame <= self.fill_value] = 0.
                data_frame[...,1][data_frame[...,1] == 0] = 30. # so \in (32.241, 35.263)
                data_list.append(data_frame if self.var is None else data_frame[...,self.var_idx:self.var_idx+1])
            data = np.stack(data_list, axis=0) # n h w v
            dataset = []
            for i in range(self.length):
                data_instance = rearrange(data[i:i + self.T_in + self.T_out, :, :, :], 't h w v -> h w (t v)')
                dataset.append(data_instance)
            dataset = np.stack(dataset, axis=0) # N, h/h_down, w/w_down, (T_in+T_out)*V
            # np.save(dataset_npy_path, dataset)
            np.save(dataset_npy_path.name, dataset)
            exit()
        dataset = torch.from_numpy(dataset.astype(np.float32))
        return dataset


class SeaDatasetMemory(Dataset):
    '''
    ['thetao', 'so', 'uo', 'vo', 'zos'] (b, 5, 180, 300)
    thetao: sea_water_potential_temperature, range [4.537, 26.204]
    so: sea_water_salinity, range [32.241, 35.263]
    uo: eastward_sea_water_velocity, range [-0.893, 1.546]
    vo: northward_sea_water_velocity, range [-1.646, 1.088]
    zos: sea_surface_height_above_geoid, range [-0.342, 1.511]
    '''
    def __init__(self, args, region='kuroshio', split='train', var=None, return_idx = False):
        self.data_path = args.data_path # /data/sea_data_small/data_sea
        self.region = region
        self.data_files = sorted([data_file for data_file in os.listdir(self.data_path) if region in data_file])
        self.h, self.w = args.h, args.w
        self.h_down, self.w_down = args.h_down, args.w_down
        self.fill_value = args.fill_value
        self.batch_size = args.batch_size
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.ntrain = args.ntrain
        self.return_idx = return_idx
        var_dict = {'thetao':0, 'so':1, 'uo':2, 'vo':3, 'zos':4}
        if var is not None and var not in var_dict.keys():
            raise Exception('var must be None or one of [\'thetao\', \'so\', \'uo\', \'vo\', \'zos\']')
        elif var is not None:
            self.var = var
            self.var_idx = var_dict[var]
        else:
            self.var = None
            self.var_idx = None

        self.split = split
        if split == 'train':
            self.length = args.ntrain
        elif split == 'test':
            self.length = args.ntest
        else:
            self.length = args.ntotal
        
        self.dataset = self._load_dataset()
    
    def _load_dataset(self):
        if self.var_idx is None:
            dataset_npy_path = Path(self.data_path) /'..'/f'sea_{self.region}.npy'
        else:
            dataset_npy_path = Path(self.data_path) /'..'/f'sea_{self.region}_{self.var}.npy'

        if os.path.exists(dataset_npy_path):
            dataset = np.load(dataset_npy_path)
            # mean = np.load(Path(self.data_path)/'..'/f'sea_{self.region}_mean2.npy')
            # std = np.load(Path(self.data_path)/'..'/f'sea_{self.region}_std.npy')
            # std = np.where(std==0, 1, std)
            # dataset = (dataset - mean) / std
        else:
            data_list = []
            for i in range(len(self.data_files)):
                data_file = Path(self.data_path) / self.data_files[i]
                data_frame = np.load(data_file)[:, :self.h:self.h_down, :self.w:self.w_down]\
                    .transpose(1,2,0).astype(np.float32) # h=180/d w=300/d v=5
                data_frame[data_frame <= self.fill_value] = 0.
                data_frame[...,1][data_frame[...,1] == 0] = 30. # so \in (32.241, 35.263)
                data_list.append(data_frame if self.var is None else data_frame[...,self.var_idx:self.var_idx+1])
            dataset = np.stack(data_list, axis=0) # n h w v
            # np.save(dataset_npy_path, dataset)
            np.save(dataset_npy_path.name, dataset)
            os.system(f'sudo cp {dataset_npy_path.name} {str(dataset_npy_path.absolute())}')
        dataset = torch.from_numpy(dataset.astype(np.float32))
        return dataset
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if self.split == 'train':
            idx = index
        elif self.split == 'test':
            idx = self.ntrain + index
        else:
            raise Exception('split must be train or test')
        
        input_x = rearrange(self.dataset[idx:idx + self.T_in, :, :, :], 't h w v -> h w (t v)')
        input_y = rearrange(self.dataset[idx + self.T_in:idx + self.T_in + self.T_out, :, :, :], 't h w v -> h w (t v)')
        if not self.return_idx:
            return input_x, input_y # B H W T*V
        else:
            return index, input_x, input_y
    
    def loader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=True if self.split=='train' else False)


class SmokeDataset(Dataset):
    def __init__(self, args, split='train', var=None):
        self.data_path = args.data_path
        self.h, self.w, self.z = args.h, args.w, args.z
        self.data_files = sorted(os.listdir(Path(self.data_path) / f'smoke_data_{self.h}_{self.w}_{self.z}'))
        self.h_down, self.w_down, self.z_down = args.h_down, args.w_down, args.z_down
        self.batch_size = args.batch_size
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.ntrain = args.ntrain
        var_dict = {'f':0, 'u':1, 'v':2, 'w':3}
        if var is not None and var not in var_dict.keys():
            raise Exception('var must be None or one of [\'f\', \'u\', \'v\', \'w\']')
        elif var is not None:
            self.var = var
            self.var_idx = var_dict[var]
        else:
            self.var = None
            self.var_idx = None

        self.split = split
        if split == 'train':
            self.length = args.ntrain
        elif split == 'test':
            self.length = args.ntest
        else:
            self.length = args.ntotal

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.split == 'train':
            index_tmp = index
        elif self.split == 'test':
            index_tmp = self.ntrain + index
        else:
            raise Exception('split must be train or test when getting file as single data file')
        
        data_file = Path(self.data_path) / f'smoke_data_{self.h}_{self.w}_{self.z}' / f'smoke_{index_tmp}.npz'
        data = np.load(data_file)
        data = torch.cat([torch.from_numpy(data['fluid_field']), torch.from_numpy(data['velocity'])], dim=-1)  # 20 32 32 32 1+3=4
        input_x = rearrange(data[:self.T_in, :self.h:self.h_down, :self.w:self.w_down, :self.z:self.z_down], 't h w z v -> z h w (t v)')
        input_y = rearrange(data[self.T_in : self.T_in+self.T_out, :self.h:self.h_down, :self.w:self.w_down, :self.z:self.z_down], 't h w z v -> z h w (t v)')
        return input_x, input_y # B Z H W T*V
    
    def loader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=True if self.split=='train' else False)
    
    def get_tensor_dataset(self):
        if self.split == 'train' or self.split == 'test':
            raise Exception('Tensor dataset as a whole, not split for train or test')
        if self.var_idx is None:
            dataset_npy_path = Path(self.data_path) / f'smoke_{self.h}_{self.w}_{self.z}.npy'
        else:
            dataset_npy_path = Path(self.data_path) / f'smoke_{self.h}_{self.w}_{self.z}_{self.var}.npy'
        
        if os.path.exists(dataset_npy_path):
            dataset = np.load(dataset_npy_path)
        else:
            data_list = []
            for i in range(len(self.data_files)):
                data_file = Path(self.data_path) / f'smoke_data_{self.h}_{self.w}_{self.z}' / f'smoke_{i}.npz'
                data_instance = np.load(data_file)
                data_instance = torch.cat([torch.from_numpy(data_instance['fluid_field']), torch.from_numpy(data_instance['velocity'])], dim=-1)  # 20 32 32 32 1+3=4
                data_instance = data_instance[:, :self.h:self.h_down, :self.w:self.w_down, :self.z:self.z_down]
                data_list.append(data_instance if self.var is None else data_instance[...,self.var_idx:self.var_idx+1])
            data = np.stack(data_list, axis=0) # n t z h w v
            dataset = []
            for i in range(self.length):
                data_instance = rearrange(data[i, :, :, :, :, :], 't z h w v -> z h w (t v)')
                dataset.append(data_instance)
            dataset = np.stack(dataset, axis=0) # N, z/z_down, h/h_down, w/w_down, (T_in+T_out)*V
            # np.save(dataset_npy_path, dataset)
            np.save(dataset_npy_path.name, dataset)
            exit()
        dataset = torch.from_numpy(dataset.astype(np.float32))
        return dataset


class SmokeDatasetMemory(Dataset):
    def __init__(self, args, split='train', var=None):
        self.data_path = args.data_path
        self.h, self.w, self.z = args.h, args.w, args.z
        self.data_files = sorted(os.listdir(Path(self.data_path) / f'smoke_data_{self.h}_{self.w}_{self.z}'))
        self.h_down, self.w_down, self.z_down = args.h_down, args.w_down, args.z_down
        self.batch_size = args.batch_size
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.ntrain = args.ntrain
        var_dict = {'f':0, 'u':1, 'v':2, 'w':3}
        if var is not None and var not in var_dict.keys():
            raise Exception('var must be None or one of [\'f\', \'u\', \'v\', \'w\']')
        elif var is not None:
            self.var = var
            self.var_idx = var_dict[var]
        else:
            self.var = None
            self.var_idx = None

        self.split = split
        if split == 'train':
            self.length = args.ntrain
        elif split == 'test':
            self.length = args.ntest
        else:
            self.length = args.ntotal
        
        self.dataset = self._load_dataset()
    
    def _load_dataset(self):
        if self.var_idx is None:
            dataset_npy_path = Path(self.data_path) / f'smoke_{self.h}_{self.w}_{self.z}.npy'
        else:
            dataset_npy_path = Path(self.data_path) / f'smoke_{self.h}_{self.w}_{self.z}_{self.var}.npy'
        
        if os.path.exists(dataset_npy_path):
            dataset = np.load(dataset_npy_path)
        else:
            data_list = []
            for i in range(len(self.data_files)):
                data_file = Path(self.data_path) / f'smoke_data_{self.h}_{self.w}_{self.z}' / f'smoke_{i}.npz'
                data_instance = np.load(data_file)
                data_instance = torch.cat([torch.from_numpy(data_instance['fluid_field']), torch.from_numpy(data_instance['velocity'])], dim=-1)  # 20 32 32 32 1+3=4
                data_instance = data_instance[:, :self.h:self.h_down, :self.w:self.w_down, :self.z:self.z_down]
                data_list.append(data_instance if self.var is None else data_instance[...,self.var_idx:self.var_idx+1])
            dataset = np.stack(data_list, axis=0) # n t z h w v
            # np.save(dataset_npy_path, dataset)
            np.save(dataset_npy_path.name, dataset)
            os.system(f'sudo cp {dataset_npy_path.name} {str(dataset_npy_path.absolute())}')
        dataset = torch.from_numpy(dataset.astype(np.float32))
        return dataset

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if self.split == 'train':
            idx = index
        elif self.split == 'test':
            idx = self.ntrain + index
        else:
            raise Exception('split must be train or test')
        
        input_x = rearrange(self.dataset[idx, :self.T_in], 't z h w v -> z h w (t v)')
        input_y = rearrange(self.dataset[idx, -self.T_out:], 't z h w v -> z h w (t v)')
        return input_x, input_y # B Z H W T*V
    
    def loader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=True if self.split=='train' else False)
