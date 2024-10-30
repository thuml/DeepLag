import numpy as np
import os


def split_npy_file(input_file, output_dir, chunk_size=1000):
    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read large .npy
    filename = os.path.basename(input_file)
    large_array = np.load(input_file)
    n, h, w = large_array.shape
    
    # split by chunk_size and save
    for i in range(0, n, chunk_size):
        split_array = large_array[i:i+chunk_size]
        output_file = os.path.join(output_dir, f'{filename[:-4]}_split_{i // chunk_size}.npy')
        np.save(output_file, split_array)
        print(f'Saved {output_file}')


def merge_npy_files(input_dir, output_file):
    # get all .npy in dir, sort by file name
    split_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy') and 'split' in f])
    
    # read files and concat arrays
    arrays = [np.load(f) for f in split_files]
    merged_array = np.concatenate(arrays, axis=0)
    
    # save marged file
    np.save(output_file, merged_array)
    print(f'Saved merged file as {output_file}')


# usage
large_npy_file = '/home/miaoshangchen/NAS/Bounded_NS/re0.25_4c_gray_10000.npy'
split_save_dir = os.path.dirname(large_npy_file)
# split_npy_file(large_npy_file, split_save_dir, chunk_size=2500)
merge_npy_files(split_save_dir, large_npy_file)
