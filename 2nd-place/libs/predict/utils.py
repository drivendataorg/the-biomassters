import numpy as np


def tta(data, no):

    if no == 0:
        data_ = data.copy()
    elif no == 1:
        data_ = np.flip(data, axis=3)
    elif no == 2:
        data_ = np.flip(data, axis=4)
    elif no == 3:
        data_ = np.transpose(data, axes=(0, 1, 2, 4, 3))
    else:
        raise NotImplemented('unknown no for tta')

    return np.ascontiguousarray(data_)


def untta(data, no):

    if no == 0:
        data_ = data.copy()
    elif no == 1:
        data_ = np.flip(data, axis=0)
    elif no == 2:
        data_ = np.flip(data, axis=1)
    elif no == 3:
        data_ = np.transpose(data, axes=(1, 0))
    else:
        raise NotImplemented('unknown no for untta')

    return np.ascontiguousarray(data_)
