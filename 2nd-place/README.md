# The BioMassters

[Competition Page](https://www.drivendata.org/competitions/99/biomass-estimation/page/534/) and [Leaderboard](https://www.drivendata.org/competitions/99/biomass-estimation/leaderboard/)

Team: **Just4Fun**

Contact: quqixun@gmail.com

Source Code: https://github.com/quqixun/BioMassters

## 1. Method

- S1 and S2 features and AGBM labels were carefully preprocessed according to statistics of training data. See code in [process.py](./process.py) and [./libs/process](./libs/process) for details.
- Training data was splited into 5 folds for cross validation in [split.py](./split.py).
- Processed S1 and S2 features were concatenated to 3D tensor in shape [B, 15, 12, 256, 256] as input, targets were AGBM labels in shape [B, 1, 256, 256].
- Some operations, including horizontal flipping, vertical flipping and random rotation in 90 degrees, were used as data augmentation on 3D features [12, 256, 256] and 2D labels [256, 256].
- We applied [Swin UNETR](https://arxiv.org/abs/2201.01266) with the attention from [Swin Transformer V2](https://arxiv.org/abs/2111.09883) as the regression model. In [./libs/models](./libs/models), Swin UNETR was adapted from [the implementation by MONAI project](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py).
- In training steps, Swin UNETR was optimized by the sum of weighted MAE and [SSIM](https://github.com/francois-rozet/piqa). RMSE of validation data was used to select the best model.
- We trained Swin UNETR using 5 folds, and got 5 models.
- For each testing sample, the average of 5 predictions was the final result.

## 2. Environment

- Ubuntu 20.04 LTS
- CUDA 11.3 or later
- Any GPU with at least 40Gb VRAM for training
- Any GPU with at least 8Gb VRAM for predicting
- At least 16Gb RAM for training and predicting
- [Minconda](https://docs.conda.io/en/main/miniconda.html) or [Anaconda](https://www.anaconda.com/) for Python environment management
- [AWS CLI](https://aws.amazon.com/cli/) for downloading dataset

```bash
# create environment
conda create --name biomassters python=3.9
conda activate biomassters

# install dependencies
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

Clone code:

```bash
git clone git@github.com:quqixun/BioMassters.git
# working dir
cd BioMassters
```

## 3. Dataset Preparation

- Download metadata from [DATA DOWNLOAD page](https://www.drivendata.org/competitions/99/biomass-estimation/data/), and put all files in [./data/information](./data/information) as following structure:

```bash
./data/information
├── biomassters-download-instructions.txt  # Instructions to download satellite images and AGBM data
├── features_metadata_FzP19JI.csv          # Metadata for satellite images
└── train_agbm_metadata.csv                # Metadata for training set AGBM tifs
```

- Download image data by running ```./scripts/download.sh```:

```bash
s3_node=as  # options: as, us, eu
split=all   # download specific dataset, options: train, test, all
            # set split to test for predicting only
            # set split to train for training only
            # set split to all otherwise
download_root=./data/source
features_metadata=./data/information/features_metadata_FzP19JI.csv
training_labels_metadata=./data/information/train_agbm_metadata.csv

python download.py \
    --download_root            $download_root            \
    --features_metadata        $features_metadata        \
    --training_labels_metadata $training_labels_metadata \
    --s3_node                  $s3_node                  \
    --split                    $split
```

Data will be saved in [./data/source](./data/source) as following arrangement.
Or you can reorganize the exist dataset in the same structure.

```
./data/source
├── test
│   ├── aa5e092e
│   │   ├── S1
│   │   │   ├── aa5e092e_S1_00.tif
│   │   │   ├── ...
│   │   │   └── aa5e092e_S1_11.tif
│   │   └── S2
│   │       ├── aa5e092e_S2_00.tif
│   │       ├── ...
│   │       └── aa5e092e_S2_11.tif
|   ├── ...
│   └── fff812c0
└── train
    ├── aa018d7b
    |   ├── S1
    |   |   └── ...
    |   ├── S2
    |   |   └── ...
    |   └── aa018d7b_agbm.tif
    ├── ...
    └── fff05995
```

- Calculate statistics for normalization and split dataset into 5 folds by running ```./scripts/process.sh```:

```bash
source_root=./data/source
split_seed=42
split_folds=5

python process.py \
    --source_root    $source_root \
    --process_method plain

python split.py \
    --data_root   $source_root \
    --split_seed  $split_seed  \
    --split_folds $split_folds
```

Outputs in [./data/source](./data/source) should be same as the following structure:

```bash
./data/sourcel
├── plot              # plot of data distribution
├── splits.pkl        # 5 folds for cross validation
├── stats_log2.pkl    # statistics of log2 transformed dataset
├── stats_plain.pkl   # statistics of original dataset
├── test
└── train
```

This step takes about 80Gb RAM. You don't have to run the above script again since all outputs can be found in [./data/source](./data/source).

## 4. Training

Train model with arguments (see [./scripts/train.sh](./scripts/train.sh)):

- ```data_root```: root directory of training dataset
- ```exp_root```: root directory to save checkpoints, logs and models
- ```config_file```: file path of configurations
- ```process_method```: processing method to calculate statistics, ```log2``` or ```plain```, default is ```plain```
- ```folds```: list of folds, separated by ```,```

```bash
device=0
process=plain
folds=0,1,2,3,4
data_root=./data/source
config_file=./configs/swin_unetr/exp1.yaml

CUDA_VISIBLE_DEVICES=$device \
python train.py              \
    --data_root      $data_root             \
    --exp_root       ./experiments/$process \
    --config_file    $config_file           \
    --process_method $process               \
    --folds          $folds
```

Run ```./scripts/train.sh``` for training, then models and logs will be saved in **./experiments/plain/swin_unetr/exp1**.

Training on 5 folds will take about 1 week if only one GPU is available.
If you have 5 GPUs, you can run each fold training on each GPU, and it will take less than 2 days.
You can download the trained models from [BaiduDisc (code:jarp)](https://pan.baidu.com/s/13yRip4gSd67vNXrn-jI5CQ), [MEGA](https://mega.nz/file/XNpWBZSY#rkA2O5JsR6TZ_xfqS3TV4I0V_xs76ni9_2PlFmfhUh8) or [Google Drive](https://drive.google.com/file/d/1R-nonruzxUU6uJYraJ6cA8Z-nUVVAJa7/view?usp=share_link), and then unzip models as following arrangement:

```bash
./experiments/plain/swin_unetr/exp1
├── fold0
│   ├── logs.csv
│   └── model.pth
├── fold1
│   ├── logs.csv
│   └── model.pth
├── fold2
│   ├── logs.csv
│   └── model.pth
├── fold3
│   ├── logs.csv
│   └── model.pth
└── fold4
    ├── logs.csv
    └── model.pth
```

## 5. Predicting

Make predictions with almost the same arguments as training  (see [./scripts/predict.sh](./scripts/predict.sh)):

- ```data_root```: root directory of training dataset
- ```exp_root```: root directory of checkpoints, logs and models
- ```output_root```: root directory to save predictions
- ```config_file```: file path of configurations
- ```process_method```: processing method to calculate statistics, ```log2``` or ```plain```, default is ```plain```
- ```folds```: list of folds, separated by ```,```
- ```apply_tta```: if apply test-time augmentation, default is ```False```

```bash
device=0
process=plain
folds=0,1,2,3,4
apply_tta=false
data_root=./data/source
config_file=./configs/swin_unetr/exp1.yaml

CUDA_VISIBLE_DEVICES=$device \
python predict.py            \
    --data_root      $data_root             \
    --exp_root       ./experiments/$process \
    --output_root    ./predictions/$process \
    --config_file    $config_file           \
    --process_method $process               \
    --folds          $folds                 \
    --apply_tta      $apply_tta
```

Run ```./scripts/predict.sh``` for predicting, then predictions will be saved in **./predictions/plain/swin_unetr/exp1/folds_0-1-2-3-4**.

Predicting public testing samples on 5 folds and calculating the average will take about 30 minutes.
You can download the submission for public testing dataset from [BaiduDisc (code:w61j)](https://pan.baidu.com/s/1KpmT2WRFHeyjN_gJXPmEHQ) or [MEGA](https://mega.nz/file/6M410TRY#ozfptCWDvYatZHMAm18uOayYDwAZSAw3U-ZfxAONLN8).

## 6. Metrics

Metrics of submitted models and predictions on validation dataset and testing dataset.

|     Metrics      | Val<br/>Fold 0 | Val<br/>Fold 1 | Val<br/>Fold 2 | Val<br/>Fold 3 | Val<br/>Fold 4 | Val<br/>Average | Test<br/>Public | Test<br/>Private |
| :--------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :-------------: | :-------------: | :--------------: |
| L<sub>rec</sub>  |    0.03562     |    0.03516     |    0.03527     |    0.03522     |    0.03626     |        -        |        -        |                  |
| L<sub>ssim</sub> |    0.04758     |    0.04684     |    0.04713     |    0.04691     |    0.04834     |        -        |        -        |                  |
|       RMSE       |    27.9676     |    27.4368     |    27.5011     |    27.8954     |    28.0946     |     27.7781     |   **27.3891**   |   **27.6779**    |

## 7. Reference

- Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. [[Paper](https://arxiv.org/abs/2201.01266)]
- Swin Transformer V2: Scaling Up Capacity and Resolution. [[Paper](https://arxiv.org/abs/2111.09883) , [Code](https://github.com/microsoft/Swin-Transformer)]
- Implementation of Swin UNETR by MONAI project. [[Code]](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py)
- Differentiable structure similarity metric. [[Code]](https://github.com/francois-rozet/piqa)
- Library for 3D augmentations. [[Paper](https://arxiv.org/abs/2104.01687), [Code](https://github.com/ZFTurbo/volumentations)]

## 8. License

- MIT License
