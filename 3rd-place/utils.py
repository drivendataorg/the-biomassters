import os
import random
import numpy as np
from skimage import morphology, measure
from glob import glob
import rioxarray as rioxr
from datetime import datetime
import xarray as xr
import pandas as pd
from xrspatial import multispectral
from tqdm import tqdm
import numpy.ma as ma
import rasterio as rio
from natsort import os_sorted
from scipy import stats
from sklearn.model_selection import train_test_split
from torchvision import transforms as T
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch.nn.functional as F
import torch


## 1_preprocessing.ipynb

def reclassify_month(month):
    if month == '00':
        return '2022-09'
    elif month == '01':
        return '2022-10'
    elif month == '02':
        return '2022-11'
    elif month == '03':
        return '2022-12'
    elif month == '04':
        return '2022-01'
    elif month == '05':
        return '2022-02'
    elif month == '06':
        return '2022-03'
    elif month == '07':
        return '2022-04'
    elif month == '08':
        return '2022-05'
    elif month == '09':
        return '2022-06'
    elif month == '10':
        return '2022-07'
    elif month == '11':
        return '2022-08'
    else:
        raise Exception("Unknown month encountered")

def dilate_image(input_array, thresholod):
    output_mask = np.where(input_array <= thresholod, 1, 0).astype(bool)
    output_mask = ~output_mask

    # # Visualize the mask
    # plt.imshow(output_mask, cmap=plt.cm.gray)
    # plt.show()

    # MORPHOLOGY:
    output_mask = morphology.binary_erosion(output_mask, selem=morphology.square(3))
    output_mask = morphology.binary_dilation(output_mask, selem=morphology.square(6))
    output_mask = output_mask.astype(bool)
    output_mask = morphology.remove_small_holes(output_mask, 3, connectivity=1)

    return output_mask

def preprocess_s1(uID, indir, outdir, composite_type):
    # TODO: Calculate VV/VH
    S1_TIFs = sorted(glob(f"{indir}/{uID}*S1*.tif"))

    dst_list = []
    for S1_TIF in S1_TIFs:
        ds = rioxr.open_rasterio(S1_TIF, parse_coordinates=True).astype(np.float)

        # Replace -9999 with NaNs
        ds = ds.where(ds != -9999)
        ds = ds.where(ds != 0)

        ## Add time dimension:
        month_str = S1_TIF.split('S1_')[1].split('.')[0]
        month_str = reclassify_month(month_str)
        dt = datetime.strptime(month_str, "%Y-%m")
        dt = pd.to_datetime(dt)
        dst = ds.assign_coords(time=dt)
        dst = dst.expand_dims(dim="time")
        ## Accumulate months:
        dst_list.append(dst)

    # stack dataarrays in list
    dst_stack = xr.combine_by_coords(dst_list, combine_attrs='override')

    if composite_type == '2S':
        ## dst_composite_summer:
        dst_composite_summer = dst_stack.isel(time=dst_stack.time.dt.month.isin([5, 6, 7, 8, 9, 10])).median('time')
        ## dst_composite_winter:
        dst_composite_winter = dst_stack.isel(time=dst_stack.time.dt.month.isin([11, 12, 1, 2, 3, 4])).median('time')
        dst_composite = xr.concat([dst_composite_summer, dst_composite_winter], dim="band")

        # Fill NaNs from corresponding bands
        dst_numpy = dst_composite[0, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC summmer) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[2, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC summer) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[4, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC winter) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[6, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC winter) contains {dst_not_nans:.2f}% of NaNs")

        ## Copy ASC to DESC where NaNs
        dst_composite_numpy = dst_composite.values
        mask_idx = (np.isnan(dst_composite_numpy[2, :, :]))
        np.putmask(dst_composite_numpy[2, :, :], mask_idx, dst_composite_numpy[0, :, :])
        np.putmask(dst_composite_numpy[3, :, :], mask_idx, dst_composite_numpy[1, :, :])
        mask_idx = (np.isnan(dst_composite_numpy[6, :, :]))
        np.putmask(dst_composite_numpy[6, :, :], mask_idx, dst_composite_numpy[4, :, :])
        np.putmask(dst_composite_numpy[7, :, :], mask_idx, dst_composite_numpy[5, :, :])
        dst_composite = xr.DataArray(dst_composite_numpy, coords={'band': dst_composite.band, 'y': dst_composite.y,
                                                                  'x': dst_composite.x}, dims=['band', 'y', 'x'])
    elif composite_type == '2SI':
        ## dst_composite_summer:
        dst_composite_summer = dst_stack.isel(time=dst_stack.time.dt.month.isin([5, 6, 7, 8, 9, 10])).median('time')
        vv_vh_asc = (dst_composite_summer.isel(band=0)/dst_composite_summer.isel(band=1)).expand_dims(band=[1])
        vv_vh_dsc = (dst_composite_summer.isel(band=2) / dst_composite_summer.isel(band=3)).expand_dims(band=[1])
        dst_composite_summer = xr.concat([dst_composite_summer, vv_vh_asc, vv_vh_dsc], dim="band")

        ## dst_composite_winter:
        dst_composite_winter = dst_stack.isel(time=dst_stack.time.dt.month.isin([11, 12, 1, 2, 3, 4])).median('time')
        vv_vh_asc = (dst_composite_winter.isel(band=0) / dst_composite_winter.isel(band=1)).expand_dims(band=[1])
        vv_vh_dsc = (dst_composite_winter.isel(band=2) / dst_composite_winter.isel(band=3)).expand_dims(band=[1])
        dst_composite_winter = xr.concat([dst_composite_winter, vv_vh_asc, vv_vh_dsc], dim="band")

        dst_composite = xr.concat([dst_composite_summer, dst_composite_winter], dim="band")

        # Fill NaNs from corresponding bands
        dst_numpy = dst_composite[0, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC summer) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[2, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC summer) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[6, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC winter) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[8, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC winter) contains {dst_not_nans:.2f}% of NaNs")

        # 0,1,2,3,4,5: VV_ASC_SUM, VH_ASC_SUM, VV_DSC_SUM, VH_DSC_SUM, VV_ASC_SUM/VH_ASC_SUM, VV_DSC_SUM/VH_DSC_SUM
        # 6,7,8,9,10,11: VV_ASC_WIN, VH_ASC_WIN, VV_DSC_WIN, VH_DSC_WIN, VV_ASC_WIN/VH_ASC_WIN, VV_DSC_WIN/VH_DSC_WIN
        ## Copy ASC to DESC where NaNs
        dst_composite_numpy = dst_composite.values
        mask_idx = (np.isnan(dst_composite_numpy[2, :, :]))
        np.putmask(dst_composite_numpy[2, :, :], mask_idx, dst_composite_numpy[0, :, :])
        np.putmask(dst_composite_numpy[3, :, :], mask_idx, dst_composite_numpy[1, :, :])
        np.putmask(dst_composite_numpy[5, :, :], mask_idx, dst_composite_numpy[4, :, :])
        mask_idx = (np.isnan(dst_composite_numpy[8, :, :]))
        np.putmask(dst_composite_numpy[8, :, :], mask_idx, dst_composite_numpy[6, :, :])
        np.putmask(dst_composite_numpy[9, :, :], mask_idx, dst_composite_numpy[7, :, :])
        np.putmask(dst_composite_numpy[11, :, :], mask_idx, dst_composite_numpy[10, :, :])
        dst_composite = xr.DataArray(dst_composite_numpy, coords={'band': dst_composite.band, 'y': dst_composite.y,
                                                                  'x': dst_composite.x}, dims=['band', 'y', 'x'])
    elif composite_type == '3S':
        # 1: Sep, Oct, Nov, Dec; 2: Jan, Feb, Mar, Apr; 3: May, Jun, Jul, Aug
        dst_composite_1 = dst_stack.isel(time=dst_stack.time.dt.month.isin([9, 10, 11, 12])).median('time')
        dst_composite_2 = dst_stack.isel(time=dst_stack.time.dt.month.isin([1, 2, 3, 4])).median('time')
        dst_composite_3 = dst_stack.isel(time=dst_stack.time.dt.month.isin([5, 6, 7, 8])).median('time')
        dst_composite = xr.concat([dst_composite_1, dst_composite_2, dst_composite_3], dim="band")

        # Fill NaNs from corresponding bands
        dst_numpy = dst_composite[0, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC: Sep, Oct, Nov, Dec) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[2, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: Sep, Oct, Nov, Dec) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[4, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC: Jan, Feb, Mar, Apr) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[6, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: Jan, Feb, Mar, Apr) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[8, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC: May, Jun, Jul, Aug) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[10, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: May, Jun, Jul, Aug) contains {dst_not_nans:.2f}% of NaNs")

        ## Copy ASC to DESC where NaNs
        dst_composite_numpy = dst_composite.values
        mask_idx = (np.isnan(dst_composite_numpy[2, :, :]))
        np.putmask(dst_composite_numpy[2, :, :], mask_idx, dst_composite_numpy[0, :, :])
        np.putmask(dst_composite_numpy[3, :, :], mask_idx, dst_composite_numpy[1, :, :])
        mask_idx = (np.isnan(dst_composite_numpy[6, :, :]))
        np.putmask(dst_composite_numpy[6, :, :], mask_idx, dst_composite_numpy[4, :, :])
        np.putmask(dst_composite_numpy[7, :, :], mask_idx, dst_composite_numpy[5, :, :])
        mask_idx = (np.isnan(dst_composite_numpy[10, :, :]))
        np.putmask(dst_composite_numpy[10, :, :], mask_idx, dst_composite_numpy[8, :, :])
        np.putmask(dst_composite_numpy[11, :, :], mask_idx, dst_composite_numpy[9, :, :])
        dst_composite = xr.DataArray(dst_composite_numpy, coords={'band': dst_composite.band, 'y': dst_composite.y,
                                                                  'x': dst_composite.x}, dims=['band', 'y', 'x'])
    elif composite_type == '4S':
        # 1: Sep, Oct, Nov; 2: Dec, Jan, Feb, 3: Mar, Apr, May; 4: Jun, Jul, Aug
        dst_composite_1 = dst_stack.isel(time=dst_stack.time.dt.month.isin([9, 10, 11])).median('time')
        dst_composite_2 = dst_stack.isel(time=dst_stack.time.dt.month.isin([12, 1, 2])).median('time')
        dst_composite_3 = dst_stack.isel(time=dst_stack.time.dt.month.isin([3, 4, 5])).median('time')
        dst_composite_4 = dst_stack.isel(time=dst_stack.time.dt.month.isin([6, 7, 8])).median('time')
        dst_composite = xr.concat([dst_composite_1, dst_composite_2, dst_composite_3, dst_composite_4], dim="band")

        # Fill NaNs from corresponding bands
        dst_numpy = dst_composite[0, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC: Sep, Oct, Nov) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[2, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: Sep, Oct, Nov) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[4, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC: Dec, Jan, Feb) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[6, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: Dec, Jan, Feb) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[8, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC: Mar, Apr, May) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[10, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: Mar, Apr, May) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[12, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC: Jun, Jul, Aug) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[14, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: Jun, Jul, Aug) contains {dst_not_nans:.2f}% of NaNs")

        ## Copy ASC to DESC where NaNs
        dst_composite_numpy = dst_composite.values
        mask_idx = (np.isnan(dst_composite_numpy[2, :, :]))
        np.putmask(dst_composite_numpy[2, :, :], mask_idx, dst_composite_numpy[0, :, :])
        np.putmask(dst_composite_numpy[3, :, :], mask_idx, dst_composite_numpy[1, :, :])
        mask_idx = (np.isnan(dst_composite_numpy[6, :, :]))
        np.putmask(dst_composite_numpy[6, :, :], mask_idx, dst_composite_numpy[4, :, :])
        np.putmask(dst_composite_numpy[7, :, :], mask_idx, dst_composite_numpy[5, :, :])
        mask_idx = (np.isnan(dst_composite_numpy[10, :, :]))
        np.putmask(dst_composite_numpy[10, :, :], mask_idx, dst_composite_numpy[8, :, :])
        np.putmask(dst_composite_numpy[11, :, :], mask_idx, dst_composite_numpy[9, :, :])
        mask_idx = (np.isnan(dst_composite_numpy[14, :, :]))
        np.putmask(dst_composite_numpy[14, :, :], mask_idx, dst_composite_numpy[12, :, :])
        np.putmask(dst_composite_numpy[15, :, :], mask_idx, dst_composite_numpy[13, :, :])

        dst_composite = xr.DataArray(dst_composite_numpy, coords={'band': dst_composite.band, 'y': dst_composite.y,
                                                                  'x': dst_composite.x}, dims=['band', 'y', 'x'])
    elif composite_type == '4SI':
        # 1: Sep, Oct, Nov, Dec; 2: Jan, Feb, Mar, Apr; 3: May, Jun, Jul, Aug
        dst_composite_1 = dst_stack.isel(time=dst_stack.time.dt.month.isin([9, 10, 11])).median('time')
        vv_vh_asc_1 = (dst_composite_1.isel(band=0) - dst_composite_1.isel(band=1)).expand_dims(band=[1])
        vv_vh_dsc_1 = (dst_composite_1.isel(band=2) - dst_composite_1.isel(band=3)).expand_dims(band=[1])
        dst_composite_1 = xr.concat([dst_composite_1, vv_vh_asc_1, vv_vh_dsc_1], dim="band")

        dst_composite_2 = dst_stack.isel(time=dst_stack.time.dt.month.isin([12, 1, 2])).median('time')
        vv_vh_asc_2 = (dst_composite_2.isel(band=0) - dst_composite_2.isel(band=1)).expand_dims(band=[1])
        vv_vh_dsc_2 = (dst_composite_2.isel(band=2) - dst_composite_2.isel(band=3)).expand_dims(band=[1])
        dst_composite_2 = xr.concat([dst_composite_2, vv_vh_asc_2, vv_vh_dsc_2], dim="band")

        dst_composite_3 = dst_stack.isel(time=dst_stack.time.dt.month.isin([3, 4, 5])).median('time')
        vv_vh_asc_3 = (dst_composite_3.isel(band=0) - dst_composite_3.isel(band=1)).expand_dims(band=[1])
        vv_vh_dsc_3 = (dst_composite_3.isel(band=2) - dst_composite_3.isel(band=3)).expand_dims(band=[1])
        dst_composite_3 = xr.concat([dst_composite_3, vv_vh_asc_3, vv_vh_dsc_3], dim="band")

        dst_composite_4 = dst_stack.isel(time=dst_stack.time.dt.month.isin([6, 7, 8])).median('time')
        vv_vh_asc_4 = (dst_composite_4.isel(band=0) - dst_composite_4.isel(band=1)).expand_dims(band=[1])
        vv_vh_dsc_4 = (dst_composite_4.isel(band=2) - dst_composite_4.isel(band=3)).expand_dims(band=[1])
        dst_composite_4 = xr.concat([dst_composite_4, vv_vh_asc_4, vv_vh_dsc_4], dim="band")

        dst_composite = xr.concat([dst_composite_1, dst_composite_2, dst_composite_3, dst_composite_4], dim="band")

        # Fill NaNs from corresponding bands
        dst_numpy = dst_composite[0, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC: Sep, Oct, Nov) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[2, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: Sep, Oct, Nov) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[6, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC: Dec, Jan, Feb) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[8, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: Dec, Jan, Feb) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[12, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC: Mar, Apr, May) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[14, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: Mar, Apr, May) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[18, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC: Jun, Jul, Aug) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[20, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: Jun, Jul, Aug) contains {dst_not_nans:.2f}% of NaNs")

        ## Copy ASC to DESC where NaNs
        dst_composite_numpy = dst_composite.values
        mask_idx = (np.isnan(dst_composite_numpy[2, :, :]))
        np.putmask(dst_composite_numpy[2, :, :], mask_idx, dst_composite_numpy[0, :, :])
        np.putmask(dst_composite_numpy[3, :, :], mask_idx, dst_composite_numpy[1, :, :])
        mask_idx = (np.isnan(dst_composite_numpy[8, :, :]))
        np.putmask(dst_composite_numpy[8, :, :], mask_idx, dst_composite_numpy[6, :, :])
        np.putmask(dst_composite_numpy[9, :, :], mask_idx, dst_composite_numpy[7, :, :])
        mask_idx = (np.isnan(dst_composite_numpy[14, :, :]))
        np.putmask(dst_composite_numpy[14, :, :], mask_idx, dst_composite_numpy[12, :, :])
        np.putmask(dst_composite_numpy[15, :, :], mask_idx, dst_composite_numpy[13, :, :])
        mask_idx = (np.isnan(dst_composite_numpy[20, :, :]))
        np.putmask(dst_composite_numpy[20, :, :], mask_idx, dst_composite_numpy[18, :, :])
        np.putmask(dst_composite_numpy[21, :, :], mask_idx, dst_composite_numpy[19, :, :])

        dst_composite = xr.DataArray(dst_composite_numpy, coords={'band': dst_composite.band, 'y': dst_composite.y,
                                                                  'x': dst_composite.x}, dims=['band', 'y', 'x'])
    elif composite_type == '6S':
        # 1: Sep, Oct, 2: Nov, Dec; 3: Jan, Feb, 4: Mar, Apr; 5: May, Jun, 6: Jul, Aug
        dst_composite_1 = dst_stack.isel(time=dst_stack.time.dt.month.isin([9, 10])).median('time')
        dst_composite_2 = dst_stack.isel(time=dst_stack.time.dt.month.isin([11, 12])).median('time')
        dst_composite_3 = dst_stack.isel(time=dst_stack.time.dt.month.isin([1, 2])).median('time')
        dst_composite_4 = dst_stack.isel(time=dst_stack.time.dt.month.isin([3, 4])).median('time')
        dst_composite_5 = dst_stack.isel(time=dst_stack.time.dt.month.isin([5, 6])).median('time')
        dst_composite_6 = dst_stack.isel(time=dst_stack.time.dt.month.isin([7, 8])).median('time')
        dst_composite = xr.concat([dst_composite_1, dst_composite_2, dst_composite_3, dst_composite_4,
                                   dst_composite_5, dst_composite_6], dim="band")


        # Fill NaNs from corresponding bands
        dst_numpy = dst_composite[0, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC: Sep, Oct) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[2, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: Sep, Oct) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[4, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC: Nov, Dec) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[6, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: Nov, Dec) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[8, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC: Jan, Feb) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[10, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: Jan, Feb) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[12, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            # ASC don't have NaNs
            print(f"{outdir}/{uID}_S1.tif (ASC: Mar, Apr) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[14, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: Mar, Apr) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[16, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: May, Jun) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[18, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: May, Jun) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[20, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: Jul, Aug) contains {dst_not_nans:.2f}% of NaNs")

        dst_numpy = dst_composite[22, :, :].values
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
        if dst_not_nans > 0:
            print(f"{outdir}/{uID}_S1.tif (DSC: Jul, Aug) contains {dst_not_nans:.2f}% of NaNs")

        ## Copy ASC to DESC where NaNs
        dst_composite_numpy = dst_composite.values
        mask_idx = (np.isnan(dst_composite_numpy[2, :, :]))
        np.putmask(dst_composite_numpy[2, :, :], mask_idx, dst_composite_numpy[0, :, :])
        np.putmask(dst_composite_numpy[3, :, :], mask_idx, dst_composite_numpy[1, :, :])
        mask_idx = (np.isnan(dst_composite_numpy[6, :, :]))
        np.putmask(dst_composite_numpy[6, :, :], mask_idx, dst_composite_numpy[4, :, :])
        np.putmask(dst_composite_numpy[7, :, :], mask_idx, dst_composite_numpy[5, :, :])
        mask_idx = (np.isnan(dst_composite_numpy[10, :, :]))
        np.putmask(dst_composite_numpy[10, :, :], mask_idx, dst_composite_numpy[8, :, :])
        np.putmask(dst_composite_numpy[11, :, :], mask_idx, dst_composite_numpy[9, :, :])
        mask_idx = (np.isnan(dst_composite_numpy[14, :, :]))
        np.putmask(dst_composite_numpy[14, :, :], mask_idx, dst_composite_numpy[12, :, :])
        np.putmask(dst_composite_numpy[15, :, :], mask_idx, dst_composite_numpy[13, :, :])
        mask_idx = (np.isnan(dst_composite_numpy[18, :, :]))
        np.putmask(dst_composite_numpy[18, :, :], mask_idx, dst_composite_numpy[16, :, :])
        np.putmask(dst_composite_numpy[19, :, :], mask_idx, dst_composite_numpy[17, :, :])
        mask_idx = (np.isnan(dst_composite_numpy[22, :, :]))
        np.putmask(dst_composite_numpy[22, :, :], mask_idx, dst_composite_numpy[20, :, :])
        np.putmask(dst_composite_numpy[23, :, :], mask_idx, dst_composite_numpy[21, :, :])

        dst_composite = xr.DataArray(dst_composite_numpy, coords={'band': dst_composite.band, 'y': dst_composite.y,
                                                                  'x': dst_composite.x}, dims=['band', 'y', 'x'])
    else:
        raise Exception(f'Unknown composite_type')

    if not dst_composite.isnull().any():
        ## TODO: Change ransform=self.transform(recalc=recalc_transform) to transform=None in ...\biomassters\Lib\site-packages\rioxarray\raster_array.py
        dst_composite.rio.to_raster(f'{outdir}/{uID}_S1.tif', compress='DEFLATE', dtype="float32",
                                    nodata=None, driver='GTiff', width=256, height=256, count=10, crs=None,
                                    transorm=None)
    else:
        print(f"{outdir}/{uID}_S1.tif still contains NaNs, filling with 0s")
        dst_composite = dst_composite.fillna(0)
        dst_composite.rio.to_raster(f'{outdir}/{uID}_S1.tif', compress='DEFLATE', dtype="float32",
                                    nodata=None, driver='GTiff', width=256, height=256, count=10, crs=None,
                                    transorm=None)

def preprocess_s2(uID, indir, outdir, composite_type):
    # TODO: Calculate NDSI: NIR1/NIR2, RE3/NIR1, SWIR1/SWIR2 (NDTI), G/NIR2, G/NIR1
    S2_TIFs = sorted(glob(f"{indir}/{uID}*S2*.tif"))

    dst_list = []
    for S2_TIF in S2_TIFs:
        ds = rioxr.open_rasterio(S2_TIF, parse_coordinates=True).astype(np.float)

        # Find blobs in the first band
        ds_labeled = measure.label(ds[9, :, :].values, background=None, connectivity=1) # use SWIR for this
        uValues, uCounts = np.unique(ds_labeled, return_counts=True)
        uCounts_sorted = uCounts.argsort()
        uCounts = uCounts[uCounts_sorted]
        uValues = uValues[uCounts_sorted]
        uValues_nan = uValues[uCounts > 256*256*0.03] # > 3%
        # Get indices of pixels to remove
        idx_nan = np.argwhere(np.isin(ds_labeled, uValues_nan))

        ## Convert to numpy for indexing and then recreate ds, because xarray multidimensional indexing is stupid
        ds_numpy = ds.values
        ds_numpy[:, idx_nan[:, 0], idx_nan[:, 1]] = np.nan
        ds = xr.DataArray(ds_numpy, coords={'band': ds.band, 'y': ds.y, 'x': ds.x}, dims=['band', 'y', 'x'])

        ## Mask data
        # ds = ds[:10, :, :].where(~(ds[10, :, :] > 50))
        # Remove where dilated/eroded CLS > 25
        ds = ds[:10, :, :].where(~dilate_image(ds[10, :, :].values, 50))

        ## Add time dimension:
        month_str = S2_TIF.split('S2_')[1].split('.')[0]
        month_str = reclassify_month(month_str)
        dt = datetime.strptime(month_str, "%Y-%m")
        dt = pd.to_datetime(dt)
        dst = ds.assign_coords(time=dt)
        dst = dst.expand_dims(dim="time")
        ## Accumulate months:
        dst_list.append(dst)

    # stack dataarrays in list
    dst_stack = xr.combine_by_coords(dst_list, combine_attrs='override')

    if composite_type == '2S':
        ## dst_composite_summer:
        dst_composite_summer = dst_stack.isel(time=dst_stack.time.dt.month.isin([5, 6, 7, 8, 9, 10])).median('time')
        ## dst_composite_winter:
        dst_composite_winter = dst_stack.isel(time=dst_stack.time.dt.month.isin([11, 12, 1, 2, 3, 4])).median('time')
        dst_composite = xr.concat([dst_composite_summer, dst_composite_winter], dim="band")
        # dst_numpy = dst_composite[[0, 10], :, :].values # TODO: Why 10 bands?
        dst_numpy = dst_composite.values
    elif composite_type == '2SI':
        ## dst_composite_summer:

        dst_composite_summer = dst_stack.isel(time=dst_stack.time.dt.month.isin([5, 6, 7, 8, 9, 10])).median('time')

        ##  0/B2/B, 1/B3/G, 2/B4/R, 3/B5/RE1, 4/B6/RE2, 5/B7/RE3, 6/B8/NIR1, 7/B8A/NIR2, 8/B11/SWIR1, 9/B12/SWIR2
        arvi = multispectral.arvi(nir_agg=dst_composite_summer.isel(band=6),
                                  red_agg=dst_composite_summer.isel(band=2),
                                  blue_agg=dst_composite_summer.isel(band=0))
        # print(f'ARVI min/max: {arvi.min().values}/{arvi.max().values}')
        evi = multispectral.evi(nir_agg=dst_composite_summer.isel(band=7), # B8A is selected here over B08 due to its better fit to the MODIS band range (using which EVI was developed and tested)
                                red_agg=dst_composite_summer.isel(band=2),
                                blue_agg=dst_composite_summer.isel(band=0),
                                c1=6.0, c2=7.5, soil_factor=1.0, gain=2.5)
        # print(f'EVI min/max: {evi.min().values}/{evi.max().values}')
        gci = multispectral.gci(nir_agg=dst_composite_summer.isel(band=6),
                                green_agg=dst_composite_summer.isel(band=1))
        # print(f'GCI min/max: {gci.min().values}/{gci.max().values}')
        nbr = multispectral.nbr(nir_agg=dst_composite_summer.isel(band=6),
                                swir2_agg=dst_composite_summer.isel(band=9))
        # print(f'NBR min/max: {nbr.min().values}/{nbr.max().values}')
        nbr2 = multispectral.nbr2(swir1_agg=dst_composite_summer.isel(band=8),
                                  swir2_agg=dst_composite_summer.isel(band=9))
        # print(f'NBR2 min/max: {nbr2.min().values}/{nbr2.max().values}')
        ndmi = multispectral.ndmi(nir_agg=dst_composite_summer.isel(band=6),
                                  swir1_agg=dst_composite_summer.isel(band=8))
        # print(f'NDMI min/max: {ndmi.min().values}/{ndmi.max().values}')
        ndvi = multispectral.ndvi(nir_agg=dst_composite_summer.isel(band=6),
                                  red_agg=dst_composite_summer.isel(band=2))
        # print(f'NDVI min/max: {ndvi.min().values}/{ndvi.max().values}')
        savi = multispectral.savi(nir_agg=dst_composite_summer.isel(band=6),
                                  red_agg=dst_composite_summer.isel(band=2))
        # print(f'SAVI min/max: {savi.min().values}/{savi.max().values}')
        sipi = multispectral.sipi(nir_agg=dst_composite_summer.isel(band=6),
                                  red_agg=dst_composite_summer.isel(band=2),
                                  blue_agg=dst_composite_summer.isel(band=0))
        # print(f'SIPI min/max: {sipi.min().values}/{sipi.max().values}')
        dst_composite_summer = xr.concat([dst_composite_summer, arvi, evi, gci, nbr, nbr2, ndmi, ndvi, savi, sipi],
                                         dim="band")

        ## dst_composite_winter:
        dst_composite_winter = dst_stack.isel(time=dst_stack.time.dt.month.isin([11, 12, 1, 2, 3, 4])).median('time')

        ##  0/B2/B, 1/B3/G, 2/B4/R, 3/B5/RE1, 4/B6/RE2, 5/B7/RE3, 6/B8/NIR1, 7/B8A/NIR2, 8/B11/SWIR1, 9/B12/SWIR2
        arvi = multispectral.arvi(nir_agg=dst_composite_winter.isel(band=6),
                                  red_agg=dst_composite_winter.isel(band=2),
                                  blue_agg=dst_composite_winter.isel(band=0))
        # print(f'ARVI min/max: {arvi.min().values}/{arvi.max().values}')
        evi = multispectral.evi(nir_agg=dst_composite_winter.isel(band=7),
                                # B8A is selected here over B08 due to its better fit to the MODIS band range (using which EVI was developed and tested)
                                red_agg=dst_composite_winter.isel(band=2),
                                blue_agg=dst_composite_winter.isel(band=0),
                                c1=6.0, c2=7.5, soil_factor=1.0, gain=2.5)
        # print(f'EVI min/max: {evi.min().values}/{evi.max().values}')
        gci = multispectral.gci(nir_agg=dst_composite_winter.isel(band=6),
                                green_agg=dst_composite_winter.isel(band=1))
        # print(f'GCI min/max: {gci.min().values}/{gci.max().values}')
        nbr = multispectral.nbr(nir_agg=dst_composite_winter.isel(band=6),
                                swir2_agg=dst_composite_winter.isel(band=9))
        # print(f'NBR min/max: {nbr.min().values}/{nbr.max().values}')
        nbr2 = multispectral.nbr2(swir1_agg=dst_composite_winter.isel(band=8),
                                  swir2_agg=dst_composite_winter.isel(band=9))
        # print(f'NBR2 min/max: {nbr2.min().values}/{nbr2.max().values}')
        ndmi = multispectral.ndmi(nir_agg=dst_composite_winter.isel(band=6),
                                  swir1_agg=dst_composite_winter.isel(band=8))
        # print(f'NDMI min/max: {ndmi.min().values}/{ndmi.max().values}')
        ndvi = multispectral.ndvi(nir_agg=dst_composite_winter.isel(band=6),
                                  red_agg=dst_composite_winter.isel(band=2))
        # print(f'NDVI min/max: {ndvi.min().values}/{ndvi.max().values}')
        savi = multispectral.savi(nir_agg=dst_composite_winter.isel(band=6),
                                  red_agg=dst_composite_winter.isel(band=2))
        # print(f'SAVI min/max: {savi.min().values}/{savi.max().values}')
        sipi = multispectral.sipi(nir_agg=dst_composite_winter.isel(band=6),
                                  red_agg=dst_composite_winter.isel(band=2),
                                  blue_agg=dst_composite_winter.isel(band=0))
        # print(f'SIPI min/max: {sipi.min().values}/{sipi.max().values}')
        dst_composite_winter = xr.concat([dst_composite_winter, arvi, evi, gci, nbr, nbr2, ndmi, ndvi, savi, sipi],
                                         dim="band")
        # Combine:
        dst_composite = xr.concat([dst_composite_summer, dst_composite_winter], dim="band")
        # dst_numpy = dst_composite[[0, 10], :, :].values # TODO: Why 10 bands?
        dst_numpy = dst_composite.values
    elif composite_type == '3S':
        # 1: Sep, Oct, Nov, Dec; 2: Jan, Feb, Mar, Apr; 3: May, Jun, Jul, Aug
        dst_composite_1 = dst_stack.isel(time=dst_stack.time.dt.month.isin([9, 10, 11, 12])).median('time')
        dst_composite_2 = dst_stack.isel(time=dst_stack.time.dt.month.isin([1, 2, 3, 4])).median('time')
        dst_composite_3 = dst_stack.isel(time=dst_stack.time.dt.month.isin([5, 6, 7, 8])).median('time')
        dst_composite = xr.concat([dst_composite_1, dst_composite_2, dst_composite_3], dim="band")
        # dst_numpy = dst_composite[[0, 10], :, :].values # TODO: Why 10 bands?
        dst_numpy = dst_composite.values
    elif composite_type == '4S':
        # 1: Sep, Oct, Nov, Dec; 2: Jan, Feb, Mar, Apr; 3: May, Jun, Jul, Aug
        dst_composite_1 = dst_stack.isel(time=dst_stack.time.dt.month.isin([9, 10, 11])).median('time')
        dst_composite_2 = dst_stack.isel(time=dst_stack.time.dt.month.isin([12, 1, 2])).median('time')
        dst_composite_3 = dst_stack.isel(time=dst_stack.time.dt.month.isin([3, 4, 5])).median('time')
        dst_composite_4 = dst_stack.isel(time=dst_stack.time.dt.month.isin([6, 7, 8])).median('time')
        dst_composite = xr.concat([dst_composite_1, dst_composite_2, dst_composite_3, dst_composite_4], dim="band")
        # dst_numpy = dst_composite[[0, 10], :, :].values # TODO: Why 10 bands?
        dst_numpy = dst_composite.values
    elif composite_type == '4SI':
        # 1: Sep, Oct, Nov, Dec; 2: Jan, Feb, Mar, Apr; 3: May, Jun, Jul, Aug
        dst_composite_1 = dst_stack.isel(time=dst_stack.time.dt.month.isin([9, 10, 11])).median('time')
        gndvi_1 = multispectral.ndvi(nir_agg=dst_composite_1.isel(band=6), red_agg=dst_composite_1.isel(band=1))
        # print(f'NDVI min/max: {gndvi_1.min().values}/{gndvi_1.max().values}')
        nbr2_1 = multispectral.nbr2(swir1_agg=dst_composite_1.isel(band=8), swir2_agg=dst_composite_1.isel(band=9))
        # print(f'NBR2 min/max: {nbr2_1.min().values}/{nbr2_1.max().values}')
        dst_composite_2 = dst_stack.isel(time=dst_stack.time.dt.month.isin([12, 1, 2])).median('time')
        gndvi_2 = multispectral.ndvi(nir_agg=dst_composite_2.isel(band=6), red_agg=dst_composite_2.isel(band=1))
        # print(f'NDVI min/max: {gndvi_2.min().values}/{gndvi_2.max().values}')
        nbr2_2 = multispectral.nbr2(swir1_agg=dst_composite_2.isel(band=8), swir2_agg=dst_composite_2.isel(band=9))
        # print(f'NBR2 min/max: {nbr2_2.min().values}/{nbr2_2.max().values}')
        dst_composite_3 = dst_stack.isel(time=dst_stack.time.dt.month.isin([3, 4, 5])).median('time')
        gndvi_3 = multispectral.ndvi(nir_agg=dst_composite_3.isel(band=6), red_agg=dst_composite_3.isel(band=1))
        # print(f'NDVI min/max: {gndvi_3.min().values}/{gndvi_3.max().values}')
        nbr2_3 = multispectral.nbr2(swir1_agg=dst_composite_3.isel(band=8), swir2_agg=dst_composite_3.isel(band=9))
        # print(f'NBR2 min/max: {nbr2_3.min().values}/{nbr2_3.max().values}')
        dst_composite_4 = dst_stack.isel(time=dst_stack.time.dt.month.isin([6, 7, 8])).median('time')
        gndvi_4 = multispectral.ndvi(nir_agg=dst_composite_4.isel(band=6), red_agg=dst_composite_4.isel(band=1))
        # print(f'NDVI min/max: {gndvi_4.min().values}/{gndvi_4.max().values}')
        nbr2_4 = multispectral.nbr2(swir1_agg=dst_composite_4.isel(band=8), swir2_agg=dst_composite_4.isel(band=9))
        # print(f'NBR2 min/max: {nbr2_4.min().values}/{nbr2_4.max().values}')
        dst_composite = xr.concat([dst_composite_1, gndvi_1, nbr2_1,
                                   dst_composite_2, gndvi_2, nbr2_2,
                                   dst_composite_3, gndvi_3, nbr2_3,
                                   dst_composite_4, gndvi_4, nbr2_4], dim="band")

        # dst_numpy = dst_composite[[0, 10], :, :].values # TODO: Why 10 bands?
        dst_numpy = dst_composite.values
    elif composite_type == '6S':
        # 1: Sep, Oct, 2: Nov, Dec; 3: Jan, Feb, 4: Mar, Apr; 5: May, Jun, 6: Jul, Aug
        dst_composite_1 = dst_stack.isel(time=dst_stack.time.dt.month.isin([9, 10])).median('time')
        dst_composite_2 = dst_stack.isel(time=dst_stack.time.dt.month.isin([11, 12])).median('time')
        dst_composite_3 = dst_stack.isel(time=dst_stack.time.dt.month.isin([1, 2])).median('time')
        dst_composite_4 = dst_stack.isel(time=dst_stack.time.dt.month.isin([3, 4])).median('time')
        dst_composite_5 = dst_stack.isel(time=dst_stack.time.dt.month.isin([5, 6])).median('time')
        dst_composite_6 = dst_stack.isel(time=dst_stack.time.dt.month.isin([7, 8])).median('time')
        dst_composite = xr.concat([dst_composite_1, dst_composite_2, dst_composite_3, dst_composite_4,
                                   dst_composite_5, dst_composite_6], dim="band")
        # dst_numpy = dst_composite[[0, 10], :, :].values # TODO: Why 10 bands?
        dst_numpy = dst_composite.values
    else:
        raise Exception(f'Unknown composite_type')

    ## Fill NaNs with 0s (if less than 10%)
    dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100
    # TODO: Interploation sometimes leads to
    if dst_not_nans > 0 and dst_not_nans <= 0.1:
        print(f"Interpolating {outdir}/{uID}_S2.tif as it contains {dst_not_nans:.3f}% NaNs")
        dst_composite = dst_composite.interpolate_na(dim='x', method="linear", fill_value="extrapolate")
        dst_composite = dst_composite.fillna(0) # TODO: Interploation sometimes leads to NaN: 45fb7068_S2 in 4SI
    elif dst_not_nans > 0.1 and dst_not_nans <= 100:
        print(f"Filling {outdir}/{uID}_S2.tif with 0s as it contains {dst_not_nans:.3f}% NaNs")
        # TODO: doesn't fill all NaNs ??
        dst_composite = dst_composite.fillna(0)
    ## TODO: Change ransform=self.transform(recalc=recalc_transform) to transform=None in ...\biomassters\Lib\site-packages\rioxarray\raster_array.py
    dst_composite.rio.to_raster(f'{outdir}/{uID}_S2.tif', compress='DEFLATE', dtype="float32", nodata=None,
                                driver='GTiff', width=256, height=256, count=10, crs=None, transorm=None)

## 2_train.ipynb

def calcuate_mean_std(image_dir, train_set, percent, channels, nodata, data, log_scale=False):
    files = [f"{os.path.join(image_dir, x)}_{data}.tif" for x in train_set]

    random.shuffle(files)
    files = files[0: int(len(files) * percent/100)]

    if not files:
        print("INFO: No Image Found!")
        return

    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(channels)
    channel_sum_squared = np.zeros(channels)

    for item in tqdm(files, total=len(files)):
        arr = rio.open(item).read() # (1, 256, 256)
        assert np.all(np.isfinite(arr))
        assert ~np.any(np.isnan(arr))
        if log_scale:
            arr = np.log1p(arr)
            # arr = np.sqrt(arr)
        if nodata is not None:
            arr = ma.masked_where(arr == nodata, arr)
        pixel_num += ma.count(arr, axis=(1, 2))
        channel_sum += ma.sum(arr, axis=(1, 2))
        channel_sum_squared += ma.sum(np.square(arr), axis=(1, 2))

    mean = channel_sum / pixel_num
    std = np.sqrt(channel_sum_squared / pixel_num - np.square(mean))

    # print("MEAN:{} \nSTD: {} ".format(list(mean), list(std)))
    return list(mean), list(std)

def stratify_data(s2_path_train, agb_path, s2_path_test, test_size, random_state):
    filenames = []
    agb_std = []
    agb_avg = []
    s2_files = sorted(glob(os.path.join(s2_path_train, "*.tif")))
    for s2_file in tqdm(s2_files):
        filenames.append(os.path.basename(s2_file).split("_S2")[0])
        with rio.open(os.path.join(agb_path, f"{os.path.basename(s2_file).split('_S2')[0]}_agbm.tif"), "r") as agb_src:
            agb_arr = agb_src.read().flatten()
            agb_std.append(np.std(agb_arr))
            agb_avg.append(np.mean(agb_arr))
    df = pd.DataFrame({"id": filenames})
    df['agb_std'] = agb_std
    df['agb_avg'] = agb_avg
    # Sort by index
    df = df.sort_values(by="id", key=lambda x: np.argsort(os_sorted(df["id"])))
    ranked = stats.rankdata(df['agb_std'])
    agb_std_percentile = ranked / len(df['agb_std']) * 100
    ranked = stats.rankdata(df['agb_avg'])
    agb_avg_percentile = ranked / len(df['agb_avg']) * 100
    bins_percentile = [0, 25, 50, 75, 100]
    df['agb_std_bin'] = np.digitize(agb_std_percentile, bins_percentile, right=True)
    df['agb_avg_bin'] = np.digitize(agb_avg_percentile, bins_percentile, right=True)
    X_train, X_val, _, _ = train_test_split(df, df, test_size=test_size,
                                            stratify=df[["agb_avg_bin", "agb_std_bin"]],
                                            random_state=random_state)

    X_train['dataset'] = 0
    X_val['dataset'] = 1
    s2_files_test = sorted(glob(os.path.join(s2_path_test, "*.tif")))
    s2_files_test = [os.path.basename(f).split('_S2')[0] for f in s2_files_test]
    X_test = pd.DataFrame({"id": s2_files_test})
    X_test['agb_std'] = 0
    X_test['agb_avg'] = 0
    X_test['dataset'] = 2

    return pd.concat([X_train, X_val, X_test], axis=0)

class BioMasstersDatasetS2S1(Dataset):
    def __init__(self, s2_path, s1_path, agb_path, X, mean, std, mean_agb, std_agb, transform=None):
        self.s2_path = s2_path
        self.agb_path = agb_path
        self.s1_path = s1_path
        self.X = X
        self.mean = mean
        self.std = std
        self.mean_agb = mean_agb
        self.std_agb = std_agb
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        s2 = rio.open(os.path.join(self.s2_path, f"{self.X[idx]}_S2.tif")).read()
        if self.s1_path is not None:
            s1 = rio.open(os.path.join(self.s1_path, f"{self.X[idx]}_S1.tif")).read()
        if self.agb_path is not None:
            agb = rio.open(os.path.join(self.agb_path, f"{self.X[idx]}_agbm.tif")).read()
            agb = np.moveaxis(agb, 0, 2)

        # Concatenate
        # TODO: Change input from (1, 256, 256) to (256, 256, 1) for Compose
        if self.s1_path is not None:
            s2s1 = np.vstack((s2, s1))
            s2s1 = np.moveaxis(s2s1, 0, 2)
        else:
            s2s1 = np.moveaxis(s2, 0, 2)

        if self.transform is not None:
            # In PyTorch and Rasterio, images are represented as [channels, height, width]
            aug = self.transform(image=s2s1, mask=agb)
            s2s1 = aug["image"]
            if self.agb_path is not None:
                agb = aug["mask"]

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        s2s1 = t(s2s1) # TODO
        if self.agb_path is not None:
            t_agb = T.Compose([T.ToTensor(), T.Normalize(self.mean_agb, self.std_agb)])
            agb = t_agb(agb)
            return s2s1, agb
        else:
            return s2s1

class UnNormalize(T.Normalize):
    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m/s for m, s in zip(mean, std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)

class SentinelModel(pl.LightningModule):
    def __init__(self, model, mean_agb, std_agb, lr=0.001, wd=0.0001):
        super().__init__()
        self.model = model
        self.mean_agb = mean_agb
        self.std_agb = std_agb
        self.lr = lr
        self.wd = wd

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = F.huber_loss(y_hat, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True)

        ## Normalization:
        t = T.Compose([UnNormalize(self.mean_agb, self.std_agb)])

        ## Clamp AGB after UnNormalize for RMSE calculation:
        _y_hat = t(y_hat)
        _y_hat[_y_hat < 0] = 0
        _y = t(y)
        _y[_y < 0] = 0

        self.log("train/rmse", torch.sqrt(F.mse_loss(_y_hat, _y)), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat = torch.nan_to_num(y_hat) # TODO: to avoid NaNs in validation when test=96

        loss = F.huber_loss(y_hat, y)
        self.log("valid/loss", loss, on_step=False, on_epoch=True)

        ## AGB Nomralization:
        t = T.Compose([UnNormalize(self.mean_agb, self.std_agb)])

        ## Clamp AGB after UnNormalize for RMSE calculation:
        _y_hat = t(y_hat)
        _y_hat[_y_hat < 0] = 0
        _y = t(y)
        _y[_y < 0] = 0

        self.log("valid/rmse", torch.sqrt(F.mse_loss(_y_hat, _y)), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               min_lr=0.00005)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': "valid/loss"}

    def forward(self, x):
        return self.model(x)

def freeze_encoder(model):
    for child in model.encoder.children():
        for param in child.parameters():
            param.requires_grad = False
    return


## 3_inference.ipynb

def inference_agb_2m(model, model2, test_set, test_set2, device,
                     mean_agb=None, std_agb=None, clamp_threshold=None,
                     preds_agbm_dir='preds_agbm', save_ground_truth=False):
    rmse_list = []
    validation = False
    for i in tqdm(range(len(test_set))):
        image = test_set[i]
        if len(image) == 2:
            validation = True
            gt = image[1]
            image = image[0]
            gt = gt[None, None, :, :]
        model.eval()
        model.to(device)
        image = image.to(device)
        if model2 is not None:
            image2 = test_set2[i]
            if len(image2) == 2:
                image2 = test_set2[i][0]
            else:
                image2 = test_set2[i]
            model2.eval()
            model2.to(device)
            image2 = image2.to(device)
        with torch.no_grad():
            pred = model(image.unsqueeze(0))
            if model2 is not None:
                pred2 = model2(image2.unsqueeze(0))
            if mean_agb is not None:
                t = T.Compose([UnNormalize(mean_agb, std_agb)])
                pred = t(pred)
                if model2 is not None:
                    pred2 = t(pred2)
                if validation:
                    gt = t(gt)
            pred = pred.squeeze().cpu().detach().numpy()
            if model2 is not None:
                pred2 = pred2.squeeze().cpu().detach().numpy()
                pred = np.mean(np.dstack((pred, pred2)), axis=2)
            # pred = np.square(pred) # TODO
            if validation:
                gt = gt.squeeze().cpu().detach().numpy()
            # Clamp predictions
            if clamp_threshold is not None:
                pred[pred < clamp_threshold] = 0
            # Calculate RMSE:
            if validation:
                rmse_list.append(np.sqrt(np.mean((pred - gt) ** 2)))
            if preds_agbm_dir is not None:
                s2_src = rio.open(f"{test_set.s2_path}/{test_set.X[i]}_S2.tif")
                meta = s2_src.meta
                meta.update({'driver': 'GTiff', 'dtype': 'float32', 'nodata': None, 'width': 256, 'height': 256,
                             'count': 1, 'crs': None, 'transform': None, 'compress': 'deflate'})
                with rio.open(f"{preds_agbm_dir}/{test_set.X[i]}_agbm.tif", "w", **meta) as dst:
                    dst.write(pred[np.newaxis, ...])
                if save_ground_truth:
                    with rio.open(f"{preds_agbm_dir}/{test_set.X[i]}_agbm_gt.tif", "w", **meta) as dst:
                        dst.write(gt[np.newaxis, ...])
    if validation:
        rmse_avg = np.mean(np.array(rmse_list))
        print(f"Average RMSE: {rmse_avg}")
        return rmse_avg
    else:
        return None


## 4_ensemble.ipynb

def f_15m(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
    return a + b*x[0] + c*x[1] + d*x[2] + e*x[3] + f*x[4] + g*x[5] + h*x[6] + i*x[7] + j*x[8] + k*x[9] + l*x[10] + \
           m*x[11] + n*x[12] + o*x[13] + p*x[14]

def ensemble_agb_15m(uIDs, popt, m1_path, m2_path, m3_path, m4_path, m5_path, m6_path, m7_path, m8_path, m9_path,
                     m10_path, m11_path, m12_path, m13_path, m14_path, m15_path, output_path_dir):
    for ID in tqdm(uIDs, total=len(uIDs)):
        pred1 = rio.open(os.path.join(m1_path, f"{ID}_agbm.tif")).read()
        pred2 = rio.open(os.path.join(m2_path, f"{ID}_agbm.tif")).read()
        pred3 = rio.open(os.path.join(m3_path, f"{ID}_agbm.tif")).read()
        pred4 = rio.open(os.path.join(m4_path, f"{ID}_agbm.tif")).read()
        pred5 = rio.open(os.path.join(m5_path, f"{ID}_agbm.tif")).read()
        pred6 = rio.open(os.path.join(m6_path, f"{ID}_agbm.tif")).read()
        pred7 = rio.open(os.path.join(m7_path, f"{ID}_agbm.tif")).read()
        pred8 = rio.open(os.path.join(m8_path, f"{ID}_agbm.tif")).read()
        pred9 = rio.open(os.path.join(m9_path, f"{ID}_agbm.tif")).read()
        pred10 = rio.open(os.path.join(m10_path, f"{ID}_agbm.tif")).read()
        pred11 = rio.open(os.path.join(m11_path, f"{ID}_agbm.tif")).read()
        pred12 = rio.open(os.path.join(m12_path, f"{ID}_agbm.tif")).read()
        pred13 = rio.open(os.path.join(m13_path, f"{ID}_agbm.tif")).read()
        pred14 = rio.open(os.path.join(m14_path, f"{ID}_agbm.tif")).read()
        pred15 = rio.open(os.path.join(m15_path, f"{ID}_agbm.tif")).read()

        ## Weighted SLR:
        pred = popt[0] + popt[1] * pred1 + popt[2] * pred2 + popt[3] * pred3 +  popt[4] * pred4 + popt[5] * pred5 + \
               popt[6] * pred6 + popt[7] * pred7 + popt[8] * pred8 + popt[9] * pred9 + popt[10] * pred10 + \
               popt[11] * pred11 + popt[12] * pred12 + popt[13] * pred13 + popt[14] * pred14 + popt[15] * pred15
        pred[pred < 0] = 0

        s2_src = rio.open(f"{m1_path}/{ID}_agbm.tif")
        meta = s2_src.meta
        meta.update({'driver': 'GTiff', 'dtype': 'float32', 'nodata': None, 'width': 256, 'height': 256,
                     'count': 1, 'crs': None, 'transform': None, 'compress': 'deflate'})
        if not os.path.exists(output_path_dir):
            os.mkdir(output_path_dir)
        with rio.open(f"{output_path_dir}/{ID}_agbm.tif", "w", **meta) as dst:
            dst.write(pred)