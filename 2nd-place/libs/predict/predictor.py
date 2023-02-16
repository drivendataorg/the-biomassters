import os
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from os.path import join as opj

from ..process import *
from .utils import tta, untta
from ..models import define_model


class BMPredictor(object):

    def __init__(self, model_paths, configs, norm_stats, process_method='plain'):

        self.norm_stats = norm_stats
        self.process_method = process_method
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # loads models
        self.models = []
        for model_path in model_paths:
            if not os.path.isfile(model_path):
                pass

            model_dict = torch.load(model_path, map_location='cpu')
            model = define_model(configs.model)
            model.load_state_dict(model_dict)
            model = model.to(self.device)
            model.eval()
            self.models.append(model)

        if process_method == 'log2':
            self.remove_outliers_func = remove_outliers_by_log2
        elif process_method == 'plain':
            self.remove_outliers_func = remove_outliers_by_plain

        self.months_list = configs.loader.months_list
        if configs.loader.months_list == 'all':
            self.months_list = list(range(12))

        self.s1_index_list = configs.loader.s1_index_list
        if configs.loader.s1_index_list == 'all':
            self.s1_index_list = list(range(4))
        
        self.s2_index_list = configs.loader.s2_index_list
        if configs.loader.s2_index_list == 'all':
            self.s2_index_list = list(range(11))

    @torch.no_grad()
    def predict(self, data_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        subjects = os.listdir(data_dir)
        for subject in tqdm(subjects, ncols=88):
            subject_dir = opj(data_dir, subject)
            feature = self._load_data(subject_dir)
            feature = torch.from_numpy(feature)
            feature = feature.to(self.device)

            preds = []
            for model in self.models:
                pred = model(feature)
                pred = pred.cpu().numpy()[0, 0]
                pred = recover_label(
                    pred, self.norm_stats['label'],
                    self.process_method
                )
                preds.append(pred)

            pred = np.mean(preds, axis=0)
            pred = Image.fromarray(pred)
            output_path = opj(output_dir, f'{subject}_agbm.tif')
            pred.save(output_path, format='TIFF', save_all=True)

    @torch.no_grad()
    def predict_tta(self, data_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        subjects = os.listdir(data_dir)
        for subject in tqdm(subjects, ncols=88):
            subject_dir = opj(data_dir, subject)
            feature = self._load_data(subject_dir)

            preds = []
            for no in range(4):
                feature_tta = tta(feature, no=no)
                feature_tta = torch.from_numpy(feature_tta)
                feature_tta = feature_tta.to(self.device)

                for model in self.models:
                    pred = model(feature_tta)
                    pred = pred.cpu().numpy()[0, 0]
                    pred = untta(pred, no=no)
                    pred = recover_label(
                        pred, self.norm_stats['label'],
                        self.process_method
                    )
                    preds.append(pred)

            pred = np.mean(preds, axis=0)
            pred = Image.fromarray(pred)
            output_path = opj(output_dir, f'{subject}_agbm.tif')
            pred.save(output_path, format='TIFF', save_all=True)

    def _load_data(self, subject_dir):
        subject = os.path.basename(subject_dir)

        # loads S1 and S2 features
        feature_list = []
        for month in self.months_list:
            s1_path = opj(subject_dir, 'S1', f'{subject}_S1_{month:02d}.tif')
            s2_path = opj(subject_dir, 'S2', f'{subject}_S2_{month:02d}.tif')
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
        feature = feature.transpose(3, 0, 1, 2).astype(np.float32)
        feature = np.expand_dims(feature, axis=0)
        # feature: (1, F, M, 256, 256)

        return feature
