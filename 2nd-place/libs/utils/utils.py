import os
import torch
import random
import numpy as np


# ------------------------------------------------------------------------------
# training initialization


def check_train_args(args):

    if not os.path.isdir(args.data_root):
        raise IOError(f'data_root {args.data_root} is not exist')

    if not os.path.isfile(args.config_file):
        raise IOError(f'config_file {args.config_file} is not exist')

    if args.process_method not in ['log2', 'plain']:
        raise ValueError(f'process_method {args.process_method} is not one of log2 or plain')

    return


def init_environment(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return


# ------------------------------------------------------------------------------
# predicting initialization


def check_predict_args(args):

    if not os.path.isdir(args.data_root):
        raise IOError(f'data_root {args.data_root} is not exist')
    
    if not os.path.isdir(args.exp_root):
        raise IOError(f'exp_root {args.exp_root} is not exist')

    if not os.path.isfile(args.config_file):
        raise IOError(f'config_file {args.config_file} is not exist')

    if args.process_method not in ['log2', 'plain']:
        raise ValueError(f'process_method {args.process_method} is not one of log2 or plain')

    return
