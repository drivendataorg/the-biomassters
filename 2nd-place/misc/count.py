import os
import warnings
import rasterio
import matplotlib
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


def heatmap(
    data, row_labels, col_labels, ax=None,
    cbar_kw=None, cbarlabel='', **kwargs
):

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va='bottom', fontsize=12)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize=12)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize=12)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im, data=None, valfmt='{x:.2f}', textcolors=('black', 'white'),
    threshold=None, **textkw
):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment='center', verticalalignment='center')
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each 'pixel'.
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


if __name__ == '__main__':

    data_dir = './data/source/train'
    subjects = os.listdir(data_dir)
    subjects.sort()

    count_data = np.zeros((15, 12))

    for subject in tqdm(subjects, ncols=88):
        subject_dir = opj(data_dir, subject)

        # load features
        for i in range(12):
            s1_path = opj(subject_dir, 'S1', f'{subject}_S1_{i:02d}.tif')
            s2_path = opj(subject_dir, 'S2', f'{subject}_S2_{i:02d}.tif')
            s1 = read_raster(s1_path, return_zeros=True, data_shape=S1_SHAPE)
            s2 = read_raster(s2_path, return_zeros=True, data_shape=S2_SHAPE)

            for s1_index in range(4):
                s1_index_data = s1[s1_index]
                s1_index_data = remove_outliers_by_plain(s1_index_data, 'S1', s1_index)
                if s1_index_data.min() < s1_index_data.max():
                    count_data[s1_index, i] += 1

            for s2_index in range(11):
                s2_index_data = s2[s2_index]
                s2_index_data = remove_outliers_by_plain(s2_index_data, 'S2', s2_index)
                if s2_index_data.min() < s2_index_data.max():
                    count_data[s2_index + 4, i] += 1

    count_data = (count_data / len(subjects)) * 100.0
    row_labels = [f'S1-{i:02d}' for i in range(4)] + [f'S2-{i:02d}' for i in range(11)]
    col_labels = [f'M{i:02d}' for i in range(12)]

    fig, ax = plt.subplots(figsize=(8, 8))
    im, cbar = heatmap(count_data, row_labels, col_labels, ax=ax,
                    cmap='YlGn', cbarlabel='Valid Data Proportion [%]')
    texts = annotate_heatmap(im, valfmt='{x:.1f}', threshold=50.0)
    fig.tight_layout()
    # plt.show()
    fig_path = './assets/counts.png'
    plt.savefig(fig_path)
