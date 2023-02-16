__all__ = ['get_dataloader']


import os
import numpy as np
import volumentations as V

from os.path import join as opj
from torch.utils.data import Dataset, DataLoader

from ..utils import *
from ..process import *


class BMDataset(Dataset):

    def __init__(
        self, mode, data_list, norm_stats, augment=False,
        process_method='plain', s1_index_list='all',
        s2_index_list='all', months_list='all'
    ):
        super(BMDataset, self).__init__()
        assert norm_stats is not None
        assert process_method in ['log2', 'plain']

        self.mode           = mode
        self.augment        = augment
        self.transform      = None
        self.data_list      = data_list
        self.norm_stats     = norm_stats
        self.process_method = process_method

        if self.augment:
            self.transform = V.Compose([
                V.Flip(1, p=0.1),
                V.Flip(2, p=0.1),
                V.RandomRotate90((1, 2), p=0.1)
            ], p=1.0)

        if process_method == 'log2':
            self.remove_outliers_func = remove_outliers_by_log2
        elif process_method == 'plain':
            self.remove_outliers_func = remove_outliers_by_plain

        self.months_list = months_list
        if months_list == 'all':
            self.months_list = list(range(12))

        self.s1_index_list = s1_index_list
        if s1_index_list == 'all':
            self.s1_index_list = list(range(4))
        
        self.s2_index_list = s2_index_list
        if s2_index_list == 'all':
            self.s2_index_list = list(range(11))

    def __len__(self):
        return len(self.data_list)

    def _load_data(self, subject_path):

        subject = os.path.basename(subject_path)

        # loads label data
        label_path = opj(subject_path, f'{subject}_agbm.tif')
        assert os.path.isfile(label_path), f'label {label_path} is not exist'
        label = read_raster(label_path, True, GT_SHAPE)
        if self.mode == 'train':
            label = self.remove_outliers_func(label, 'label')
            label = normalize(label, self.norm_stats['label'], 'minmax')
        label = np.expand_dims(label, axis=-1)

        # loads S1 and S2 features
        feature_list = []
        for month in self.months_list:
            s1_path = opj(subject_path, 'S1', f'{subject}_S1_{month:02d}.tif')
            s2_path = opj(subject_path, 'S2', f'{subject}_S2_{month:02d}.tif')
            s1 = read_raster(s1_path, True, S1_SHAPE)
            s2 = read_raster(s2_path, True, S2_SHAPE)

            s1_list = []
            for index in self.s1_index_list:
                s1i = self.remove_outliers_func(s1[index], 'S1', index)
                s1i = normalize(s1i, self.norm_stats['S1'][index], 'zscore')
                s1_list.append(s1i)

            s2_list = []
            for index in self.s2_index_list:
                s2i = self.remove_outliers_func(s2[index], 'S2', index)
                s2i = normalize(s2i, self.norm_stats['S2'][index], 'zscore')
                s2_list.append(s2i)

            feature = np.stack(s1_list + s2_list, axis=-1)
            feature = np.expand_dims(feature, axis=0)
            feature_list.append(feature)
        feature = np.concatenate(feature_list, axis=0)

        return label, feature

    def __getitem__(self, index):

        subject_path = self.data_list[index]
        label, feature = self._load_data(subject_path)
        # label:   (1, 256, 256, 1)
        # feature: (M, 256, 256, F)

        if self.augment:
            data = {'image': feature, 'mask': label}
            aug_data = self.transform(**data)
            feature, label = aug_data['image'], aug_data['mask']
            if label.shape[0] > 1:
                label = label[:1]

        feature = feature.transpose(3, 0, 1, 2).astype(np.float32)
        # feature: (F, M, 256, 256)
        label = label[0].transpose(2, 0, 1).astype(np.float32)
        # label: (1, 256, 256)

        return feature, label


def get_dataloader(
    mode, data_list, configs, norm_stats=None,
    process_method='plain'
):
    assert mode in ['train', 'val']

    if mode == 'train':
        batch_size = configs.train_batch
        drop_last  = True
        shuffle    = True
        augment    = configs.apply_augment
    else:  # mode == 'val'
        batch_size = configs.val_batch
        drop_last  = False
        shuffle    = False
        augment    = False

    dataset = BMDataset(
        mode           = mode,
        data_list      = data_list,
        norm_stats     = norm_stats,
        augment        = augment,
        process_method = process_method,
        s1_index_list  = configs.s1_index_list,
        s2_index_list  = configs.s2_index_list,
        months_list    = configs.months_list,
    )

    dataloader = DataLoader(
        dataset,
        batch_size  = batch_size,
        num_workers = configs.num_workers,
        pin_memory  = configs.pin_memory,
        drop_last   = drop_last,
        shuffle     = shuffle,
    )

    return dataloader
