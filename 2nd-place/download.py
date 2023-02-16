import os
import argparse
import subprocess
import pandas as pd

from tqdm import tqdm


DOWNLOAD_ROOT_DIR          = './data/source'
FEATURES_METADATA_PATH     = './data/information/features_metadata.csv'
TRAIN_LABELS_METADATA_PATH = './data/information/train_agbm_metadata.csv'


def download_from_s3_path(s3_path, local_path):
    try:
        command = f'aws s3 cp {s3_path} {local_path} --no-sign-request'
        subprocess.call(
            command.split(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )

    except Exception as e:
        print(f'Faild to download from {s3_path}: {e}')


def get_s3_node(node='as'):
    assert node in ['us', 'eu', 'as'], f'{node} should be one of us, eu or as'
    return f's3path_{node}'


def check_file_exist(filepath, filesize):
    if os.path.isfile(filepath):
        if os.path.getsize(filepath) == filesize:
            return True
    return False


def download_features(download_root=None, features_metadata_path=None, node='as', target_split='all'):
    print(f'Downloading {target_split} features ...')

    if download_root is None:
        download_root = DOWNLOAD_ROOT_DIR
    if features_metadata_path is None:
        features_metadata_path = FEATURES_METADATA_PATH

    features_metadata = pd.read_csv(features_metadata_path)
    if target_split != 'all':
        features_metadata = features_metadata.loc[features_metadata['split'] == target_split]
    num_files = len(features_metadata)

    s3_node = get_s3_node(node)
    for i, row in tqdm(features_metadata.iterrows(), total=num_files, ncols=88):
        split     = row['split']
        s3_path   = row[s3_node]
        chip_id   = row['chip_id']
        filesize  = row['size']
        filename  = row['filename']
        satellite = row['satellite']

        local_dir  = os.path.join(download_root, split, chip_id, satellite)
        local_path = os.path.join(local_dir, filename)

        if not check_file_exist(local_path, filesize):
            download_from_s3_path(s3_path, local_path)


def download_training_labels(download_root=None, training_labels_metadata_path=None, node='as'):
    print('Downloading training labels ...')

    if download_root is None:
        download_root = DOWNLOAD_ROOT_DIR
    if training_labels_metadata_path is None:
        training_labels_metadata_path = TRAIN_LABELS_METADATA_PATH

    training_labels_metadata = pd.read_csv(training_labels_metadata_path)
    num_files = len(training_labels_metadata)

    s3_node = get_s3_node(node)
    for i, row in tqdm(training_labels_metadata.iterrows(), total=num_files, ncols=88):
        s3_path  = row[s3_node]
        chip_id  = row['chip_id']
        filesize = row['size']
        filename = row['filename']

        local_dir  = os.path.join(download_root, 'train', chip_id)
        local_path = os.path.join(local_dir, filename)

        if not check_file_exist(local_path, filesize):
            download_from_s3_path(s3_path, local_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BioMassters Downloading')
    parser.add_argument('--download_root',            type=str, help='dir path for saving dataset')
    parser.add_argument('--features_metadata',        type=str, help='file path of metadata of features')
    parser.add_argument('--training_labels_metadata', type=str, help='file path of metadata of training labels')
    parser.add_argument('--s3_node',                  type=str, help='s3 node, us, eu or as')
    parser.add_argument('--split',                    type=str, choices=['train', 'test', 'all'], help='dataset split, train, test or all')
    args = parser.parse_args()

    download_features(
        download_root=args.download_root,
        features_metadata_path=args.features_metadata,
        node=args.s3_node,
        target_split=args.split
    )

    if args.split != 'test':
        download_training_labels(
            download_root=args.download_root,
            training_labels_metadata_path=args.training_labels_metadata,
            node=args.s3_node
        )
