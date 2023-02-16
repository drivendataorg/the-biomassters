import numpy as np


# ------------------------------------------------------------------------------
#  remove outliers from log2 transformed labels and features


INLIERS_LOG2 = {
    'label': {'2pow': -6, 'min': -3.0, 'max': 14.0},
    'S1': {
        0: {'2pow': None, 'min': -25.0, 'max': 30.0},
        1: {'2pow': None, 'min': -63.0, 'max': 29.0},
        2: {'2pow': None, 'min': -25.0, 'max': 32.0},
        3: {'2pow': None, 'min': -70.0, 'max': 23.0},
    },
    'S2': {
        0:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        1:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        2:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        3:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        4:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        5:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        6:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        7:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        8:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        9:  {'2pow': -1,   'min': 0.0, 'max': 14.0 },
        10: {'2pow': None, 'min': 0.0, 'max': 100.0},
    }
}


def remove_outliers_by_log2(data, data_name, data_index=None):

    if data_name == 'label':
        inlier_dict = INLIERS_LOG2['label']
    else:
        inlier_dict = INLIERS_LOG2[data_name][data_index]

    if inlier_dict['2pow'] is not None:
        min_thresh = 2 ** inlier_dict['2pow']
        data = np.where(data < min_thresh, min_thresh, data)
        data = np.log2(data)

    min_ = inlier_dict['min']
    max_ = inlier_dict['max']

    if (data_name == 'S2') and (data_index == 10):
        data = np.where(data < min_, min_, data)
        data = np.where(data > max_, min_, data)
    else:
        data = np.where(data < min_, min_, data)
        data = np.where(data > max_, max_, data)

    return data


# ------------------------------------------------------------------------------
#  remove outliers from plain labels and features


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
