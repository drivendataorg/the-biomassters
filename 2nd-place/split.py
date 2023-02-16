import os
import pickle
import random
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BioMassters Splitting')
    parser.add_argument('--data_root',   type=str, help='dir path of dataset')
    parser.add_argument('--split_seed',  type=int, default=42, help='random seed')
    parser.add_argument('--split_folds', type=int, default=5,  help='number of folds')
    args = parser.parse_args()

    data_root = args.data_root
    split_seed = args.split_seed
    split_folds = args.split_folds

    data_dir = os.path.join(data_root, 'train')
    subjects = os.listdir(data_dir)
    subjects.sort()

    random.seed(split_seed)
    random.shuffle(subjects)

    splits = {}
    per_fold = round(len(subjects) / split_folds)
    for k in range(split_folds):
        start = k * per_fold
        end = (k + 1) * per_fold if k < (split_folds - 1) else len(subjects)

        val   = [i for i in subjects[start:end]]
        train = [i for i in subjects if i not in val]
        val   = [os.path.join(data_dir, i) for i in val]
        train = [os.path.join(data_dir, i) for i in train]
        splits[k] = {'train': train, 'val': val}

    splits_path = os.path.join(data_root, f'splits.pkl')
    with open(splits_path, 'wb') as f:
        pickle.dump(splits, f)
