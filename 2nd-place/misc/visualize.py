import os
import warnings
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from os.path import join as opj


warnings.filterwarnings('ignore')


GT_SHAPE = (1,  256, 256)
S1_SHAPE = (4,  256, 256)
S2_SHAPE = (11, 256, 256)

INLIERS_PLAIN = {
    'S1': {
        0: {'min': -25.0, 'max': 30.0},
        1: {'min': -63.0, 'max': 29.0},
        2: {'min': -25.0, 'max': 32.0},
        3: {'min': -70.0, 'max': 23.0},
    },
    'S2': {
        10: {'min': 0.0, 'max': 100.0},
    }
}


def remove_outliers_by_plain(data, data_name, data_index=None):

    if data_name == 'label':
        data = np.where(data < 0.0, 0.0, data)

    elif data_name == 'S1':
        inlier_dict = INLIERS_PLAIN['S1'][data_index]
        min_ = inlier_dict['min']
        max_ = inlier_dict['max']
        data = np.where(data < min_, min_, data)
        data = np.where(data > max_, max_, data)

    elif data_name == 'S2':
        if data_index == 10:
            inlier_dict = INLIERS_PLAIN['S2'][10]
            min_ = inlier_dict['min']
            max_ = inlier_dict['max']
            data = np.where(data < min_, min_, data)
            data = np.where(data > max_, min_, data)

    return data


def read_raster(data_path, return_zeros=False, data_shape=None):

    if os.path.isfile(data_path):
        raster = rasterio.open(data_path)
        data = raster.read()
    else:
        if return_zeros:
            assert data_shape is not None
            data = np.zeros(data_shape).astype(np.float32)
        else:
            data = None

    return data


if __name__ == '__main__':

    data_dir = './data/source/train'
    subjects = os.listdir(data_dir)
    subjects.sort()

    for subject in tqdm(subjects, ncols=88):
        subject_dir = opj(data_dir, subject)

        # load label data
        label_path = opj(subject_dir, f'{subject}_agbm.tif')
        label = read_raster(label_path, return_zeros=True, data_shape=GT_SHAPE)
        label = remove_outliers_by_plain(label, 'label')

        # load features
        for i in range(12):
            s1_path = opj(subject_dir, 'S1', f'{subject}_S1_{i:02d}.tif')
            s2_path = opj(subject_dir, 'S2', f'{subject}_S2_{i:02d}.tif')
            s1 = read_raster(s1_path, return_zeros=True, data_shape=S1_SHAPE)
            s2 = read_raster(s2_path, return_zeros=True, data_shape=S2_SHAPE)

            plt.figure(f'{subject} - {i:02d}', figsize=(15, 15))
            plt.subplot(4, 4, 1)
            plt.title('GT')
            plt.imshow(label[0])
            plt.axis('off')
            for s1_index in range(s1.shape[0]):
                s1_index_data = s1[s1_index]
                s1_index_data = remove_outliers_by_plain(s1_index_data, 'S1', s1_index)
                plt.subplot(4, 4, s1_index + 2)
                plt.title(f'S1-{s1_index + 1}')
                plt.imshow(s1_index_data)
                plt.axis('off')
            for s2_index in range(s2.shape[0]):
                s2_index_data = s2[s2_index]
                s2_index_data = remove_outliers_by_plain(s2_index_data, 'S2', s2_index)
                plt.subplot(4, 4, s2_index + 2 + s1.shape[0])
                plt.title(f'S2-{s2_index + 1}')
                plt.imshow(s2_index_data)
                plt.axis('off')
            plt.tight_layout()
            plt.show()
