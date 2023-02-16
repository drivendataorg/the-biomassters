import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from .clean import INLIERS_LOG2


GT_SHAPE = (1,  256, 256)
S1_SHAPE = (4,  256, 256)
S2_SHAPE = (11, 256, 256)


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


def calculate_statistics(
    data, data_name, exclude_mins=False,
    p=None, hist=False, plot_dir=None
):

    if exclude_mins:
        min_ = np.min(data) 
        data = data[np.where(data > min_)]

    if p is not None:
        assert 0 <= p <= 100
        data_min = np.percentile(data, 100 - p)
        data_max = np.percentile(data, p)
        data = np.where(data < data_min, data_min, data)
        data = np.where(data > data_max, data_max, data)
    else:
        data_min = np.min(data)
        data_max = np.max(data)

    data_avg = np.mean(data)
    data_std = np.std(data)

    print(f'Statistics of {data_name} with percentile {p}:')
    print(f'- min: {data_min:.3f}')
    print(f'- max: {data_max:.3f}')
    print(f'- avg: {data_avg:.3f}')
    print(f'- std: {data_std:.3f}')

    if hist:
        assert plot_dir is not None
        os.makedirs(plot_dir, exist_ok=True)
        plot_file = f'stats_{data_name}_p{p}.png'
        plot_path = os.path.join(plot_dir, plot_file)

        plt.figure()
        plt.title(f'{data_name} - P:{p}')
        plt.hist(data.reshape(-1), bins=100, log=True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return {
        'min': data_min,
        'max': data_max,
        'avg': data_avg,
        'std': data_std,
    }


def normalize(data, norm_stats, norm_method):
    assert norm_method in ['minmax', 'zscore']

    min_ = norm_stats['min']
    max_ = norm_stats['max']
    data = np.where(data < min_, min_, data)
    data = np.where(data > max_, max_, data)

    if norm_method == 'minmax':
        range_ = max_ - min_
        data = (data - min_) / range_
    elif norm_method == 'zscore':
        avg = norm_stats['avg']
        std = norm_stats['std']
        data = (data - avg) / std

    return data


def recover_label(data, norm_stats, recover_method, norm_method='minmax'):
    assert norm_method in ['minmax', 'zscore']
    assert recover_method in ['log2', 'plain']

    if norm_method == 'minmax':
        min_ = norm_stats['min']
        max_ = norm_stats['max']
        range_ = max_ - min_
        data = data * range_ + min_
    else:
        avg = norm_stats['avg']
        std = norm_stats['std']
        data = data * std + avg

    if recover_method == 'log2':
        data = 2 ** data
        min_thresh = 2 ** INLIERS_LOG2['label']['2pow']
        data = np.where(data < min_thresh, 0, data)

    return data
