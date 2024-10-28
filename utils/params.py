import argparse
import os
import sys


class MyArgumentParser(argparse.ArgumentParser):
    def _read_args_from_files(self, arg_strings):
        # expand arguments referencing files
        new_arg_strings = []
        for arg_string in arg_strings:

            # for regular arguments, just add them back into the list
            if not arg_string or arg_string[0] not in self.fromfile_prefix_chars:
                new_arg_strings.append(arg_string)

            # replace arguments referencing files with the file content
            else:
                try:
                    with open(arg_string[1:]) as args_file:
                        arg_strings = []
                        for arg_line in args_file.read().splitlines():
                            argName, argValue = self.convert_arg_line_to_args(arg_line)
                            arg_strings.append(argName)
                            if argValue is not None:
                                arg_strings.append(argValue)
                        arg_strings = self._read_args_from_files(arg_strings)
                        new_arg_strings.extend(arg_strings)
                except OSError:
                    err = sys.exc_info()[1]
                    self.error(str(err))

        # return the modified argument list
        return new_arg_strings
    
    def convert_arg_line_to_args(self, arg_line):
        name, value = arg_line.strip().split(' : ')
        name = name.replace('_', '-')
        # print(f'name={name}, value={value}.')
        if value == 'True' or value == 'False':
            return ['--' + name, None]
        else:
            return ['--' + name, value]


def get_args(time='', cfg_file=None):
    parser = MyArgumentParser('', add_help=False, fromfile_prefix_chars='@')

    # dataset
    parser.add_argument('--dataset-nickname', default='bc', type=str)
    parser.add_argument('--data-path', default='./dataset', type=str, help='dataset folder')
    parser.add_argument('--region', default='kuroshio', type=str, help='region for sea dataset')
    parser.add_argument('--ntotal', default=1200, type=int, help='number of overall data')
    parser.add_argument('--ntrain', default=1000, type=int, help='number of train set')
    parser.add_argument('--ntest', default=200, type=int, help='number of test set')
    parser.add_argument('--in-dim', default=1, type=int, help='input data dimension(frames)')
    parser.add_argument('--out-dim', default=1, type=int, help='output data dimension(frames)')
    parser.add_argument('--in-var', default=1, type=int, help='number of variables')
    parser.add_argument('--out-var', default=1, type=int, help='number of variables')
    parser.add_argument('--xmin', default=0, type=float, help='lower bound of x-axis')
    parser.add_argument('--xmax', default=1, type=float, help='upper bound of x-axis')
    parser.add_argument('--ymin', default=0, type=float, help='lower bound of y-axis')
    parser.add_argument('--ymax', default=1, type=float, help='upper bound of y-axis')
    parser.add_argument('--zmin', default=0, type=float, help='lower bound of z-axis')
    parser.add_argument('--zmax', default=1, type=float, help='upper bound of z-axis')
    parser.add_argument('--has-t', action='store_true', default=False,
                        help='If the dataset has the temporal dimension')
    parser.add_argument('--tmin', default=0, type=float, help='lower bound of t-axis')
    parser.add_argument('--tmax', default=1, type=float, help='upper bound of t-axis')
    # parser.add_argument('--order', default=2, type=int, help='highest order of diff')
    # parser.add_argument('--exterp', action='store_true', default=False,
    #                     help='If use exterpolate in diff calc')
    parser.add_argument('--h', default=1, type=int, help='input data height')
    parser.add_argument('--w', default=1, type=int, help='input data width')
    parser.add_argument('--z', default=1, type=int, help='input data layers')
    parser.add_argument('--T-in', default=10, type=int,
                        help='input data time points (only for temporal related experiments)')
    parser.add_argument('--T-out', default=10, type=int,
                        help='predict data time points (only for temporal related experiments)')
    parser.add_argument('--h-down', default=1, type=int, help='height downsample rate of input')
    parser.add_argument('--w-down', default=1, type=int, help='width downsample rate of input')
    parser.add_argument('--z-down', default=1, type=int, help='layer downsample rate of input')
    parser.add_argument('--fill-value', default=-32760., type=float, help='mask value for land in sea dataset')
    parser.add_argument('--delta-t', default=1, type=int, help='time interval between frames')

    # optimization
    parser.add_argument('--batch-size', default=20, type=int, help='batch size of training')
    parser.add_argument('--learning-rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=501, type=int, help='training epochs')
    parser.add_argument('--step-size', default=100, type=int, help='interval of model save')
    parser.add_argument('--gamma', default=0.5, type=float, help='parameter of learning rate scheduler')

    # Model parameters
    parser.add_argument('--model', default='DeepLag_2D', type=str, help='model name')
    parser.add_argument('--model-nickname', default='deeplag', type=str)
    parser.add_argument('--depth', default=3, type=int, help='depth of the transformer')
    parser.add_argument('--d-model', default=32, type=int, help='channels of hidden variates')
    parser.add_argument('--heads', default=6, type=int, help='num of heads of the transformer')
    parser.add_argument('--dim-head', default=64, type=int, help='dim per head of the transformer')
    parser.add_argument('--num-samples', default=100, type=int, help='number of sample points')
    parser.add_argument('--num-layers', default=4, type=int, help='number of basic layers')
    parser.add_argument('--num-basis', default=12, type=int, help='number of basis operators')
    parser.add_argument('--num-token', default=4, type=int, help='number of latent tokens')
    parser.add_argument('--patch-size', default='3,3', type=str, help='patch size of different dimensions')
    parser.add_argument('--padding', default='3,3', type=str, help='padding size of different dimensions')
    parser.add_argument('--dropout', default=0., type=float, help='dropout rate')
    parser.add_argument('--emb-dropout', default=0., type=float, help='dropout rate of patch embedding')
    parser.add_argument('--mlp-dropout', default=0., type=float, help='dropout rate of mlp')
    parser.add_argument('--kernel-size', default=3, type=int, help='')
    parser.add_argument('--offset-ratio-range', default='16,8', type=str, help='min & max offset ratio wrt whole field of selected points')
    parser.add_argument('--resample-strategy', default='uniform', choices=['uniform', 'boundary', 'domain', 'learned'], type=str, help='how to resample points out of bound')

    # save & load
    parser.add_argument('--model-save-path', default='./checkpoints/', type=str, help='model save path')
    parser.add_argument('--model-save-name', default='deeplag.pt', type=str, help='model name')
    parser.add_argument('--log-save-name', default='log.txt', type=str, help='log name')
    parser.add_argument('--run-save-path', default='./checkpoints/', type=str, help='save path for all outputs of this run')
    parser.add_argument('--save-char-res', action='store_true', default=False,
                        help='If save result of characteristic surface')
    
    parser.add_argument('--sampling-rate', default=0.5, type=float, help='sampling rate of imputation task')

    if cfg_file is None:
        args = parser.parse_args()
        argsDict = args.__dict__
        args.run_save_path = f'{args.model_save_path}/{args.model}/{time}'
        if not os.path.exists(args.run_save_path):
            os.makedirs(args.run_save_path)
        os.system(f'cp ./scripts/{args.dataset_nickname}_{args.model_nickname}.sh {args.run_save_path}/')
        os.system(f'cp ./exp_{args.dataset_nickname}_h.py {args.run_save_path}/')
        os.system(f'cp ./models/{args.model}.py {args.run_save_path}/')
        with open(os.path.join(args.run_save_path, 'configs.txt'), 'w') as f:
            for arg, value in argsDict.items():
                f.writelines(arg + ' : ' + str(value) + '\n')
    else:
        args = parser.parse_args([f'@{str(cfg_file)}'])
    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', default='./checkpoints/', type=str, help='save path of all checkpoints')
    parser.add_argument('--dataset-nickname', default='bc', type=str, help='Dataset short name in codebase files')
    parser.add_argument('--data-path', default='./dataset', type=str, help='dataset folder')
    parser.add_argument('--model-name', default='DeepLag_2D', type=str, help='model name in model files')
    parser.add_argument('--time-str', default='19190810_114514', type=str, help='version of model running at that time')
    parser.add_argument('--milestone', default='best', type=str, help='which model to use (e.g. best or some epoch)')
    parser.add_argument('--T-out', default=10, type=int,
                        help='predict data time points (only for temporal related experiments)')
    args = parser.parse_args()
    return args