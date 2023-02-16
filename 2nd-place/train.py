import os
import yaml
import pickle
import argparse
import warnings

from libs.utils import *
from libs.train import *
from omegaconf import OmegaConf


warnings.filterwarnings('ignore')


def main(args):

    # --------------------------------------------------------------------------
    # loads configs

    with open(args.config_file, 'r') as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    configs = OmegaConf.create(configs)

    # --------------------------------------------------------------------------
    # loads data splits and stats

    splits_path = os.path.join(args.data_root, f'splits.pkl')
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)

    stats_file = f'stats_{args.process_method}.pkl'
    stats_path = os.path.join(args.data_root, stats_file)
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # --------------------------------------------------------------------------
    # generates folds

    fold_id_list = list(map(int, args.folds.split(',')))
    fold_id_list = [f for f in fold_id_list if f < configs.cv]
    if len(fold_id_list) == 0:
        raise ValueError(f'folds {args.folds} are not available')

    # --------------------------------------------------------------------------
    # initialize environments

    init_environment(configs.seed)

    # --------------------------------------------------------------------------
    # training in cross validation manner

    for fold_id in fold_id_list:
        exp_dir = os.path.join(args.exp_root, configs.exp, f'fold{fold_id}')
        train_list = splits[fold_id]['train']
        val_list   = splits[fold_id]['val']

        # prints information
        print('-' * 100)
        print('BioMassters Training ...\n')
        print(f'- Data Root : {args.data_root}')
        print(f'- Exp Dir   : {exp_dir}')
        print(f'- Configs   : {args.config_file}')
        print(f'- Fold      : {fold_id}')
        print(f'- Num Train : {len(train_list)}')
        print(f'- Num Val   : {len(val_list)}\n')

        loader_kwargs = dict(
            configs        = configs.loader,
            norm_stats     = stats,
            process_method = args.process_method
        )
        train_loader = get_dataloader('train', train_list, **loader_kwargs)
        val_loader   = get_dataloader('val',   val_list,   **loader_kwargs)

        # initialize trainer
        trainer = BMTrainer(
            configs = configs,
            exp_dir = exp_dir,
            resume  = args.resume
        )

        # training model
        trainer.forward(train_loader, val_loader)

        print('-' * 100, '\n')

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BioMassters Training')
    parser.add_argument('--data_root',      type=str, help='dir path of training data')
    parser.add_argument('--exp_root',       type=str, help='root dir of experiments')
    parser.add_argument('--config_file',    type=str, help='yaml path of configs')
    parser.add_argument('--process_method', type=str, default='plain', help='method for processing, log2 or plain')
    parser.add_argument('--resume',         action='store_true', help='if resume from checkpoint')
    parser.add_argument('--folds',          type=str, default='0,1,2,3,4', help='list of folds, separated by ,')
    args = parser.parse_args()

    check_train_args(args)
    main(args)
