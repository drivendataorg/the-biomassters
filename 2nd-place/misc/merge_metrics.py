import os
import numpy as np
import pandas as pd

from os.path import join as opj


if __name__ == '__main__':

    cv = 5
    output_dir = './assets'
    os.makedirs(output_dir, exist_ok=True)
    output_path = opj(output_dir, 'val_metrics.csv')

    exp_root = './experiments'
    data_list = []

    try:
        for process in os.listdir(exp_root):
            process_dir = opj(exp_root, process)
            for model in os.listdir(process_dir):
                model_dir = opj(process_dir, model)
                for exp in os.listdir(model_dir):
                    exp_dir = opj(model_dir, exp)

                    best_val_rmse_list = []
                    for fold in range(cv):
                        fold_dir = opj(exp_dir, f'fold{fold}')
                        logs_path = opj(fold_dir, 'logs.csv')

                        if os.path.isfile(logs_path):
                            logs = pd.read_csv(logs_path)
                            best_val_rmse = logs['val_rmse'].min()
                            best_val_rmse_list.append(best_val_rmse)
                        else:
                            best_val_rmse_list.append(np.nan)

                    best_val_rmse = np.nanmean(best_val_rmse_list)
                    data = [process, model, exp] + best_val_rmse_list + [best_val_rmse]
                    data_list.append(data)
    except Exception as e:
        print(e)

    columns = ['process', 'model', 'exp'] + [f'val_rmse_fold{i}' for i in range(cv)] + ['val_rmse_avg']
    data_df = pd.DataFrame(data=data_list, columns=columns)
    data_df.sort_values(by='val_rmse_avg', inplace=True)
    data_df.to_csv(output_path, index=False)
