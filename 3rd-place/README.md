# [The BioMassters DrivenData Challenge](drivendata.org/competitions/99/biomass-estimation/page/534/)

[3<sup>rd</sup> place (4<sup>th</sup> rank)](https://www.drivendata.org/competitions/99/biomass-estimation/leaderboard/) out of 976 participants with *RMSE* (private) of 28.0438

### **1. Method**
* *Sentinel-1* and *Sentinel-2* imagery were preprocessed into 6 cloud-free median composites to reduce data dimensionality while preserving the maximum amount of information
* 15 models were trained using a *UNet++* architecture in combination with various encoders (e.g. *se_resnext50_32x4d* and *efficientnet-b{7,8}*) and median cloud-free composites. From the experiments, *UNet++* showed better performance as compared to other decoders (e.g *UNet*, *MANet* etc.)
* The models were pretrained with multiple augmentations (*HorizontalFlip*, *VerticalFlip*, *RandomRotate90*, *Transpose*, *ShiftScaleRotate*), batch size of 32, *AdamW* optimizer with 0.001 initial learning rate, weight decay of 0.0001, and a *ReduceLROnPlateau* scheduler
* *UNet++* models were optimized using a *Huber* loss to reduce the effect of outliers in the data for 200 epochs
* To improve the performance of each *UNet++* model they were further fine-tuned (after freezing pre-trained encoder weights and removing augmentations) for another 100 epochs
* For each *UNet++* model the average of 2 best predictions was used for further ensembling
* The ensemble of all 15 models using weighted average was used for the final submission

NOTE: The final solution turned-out to be quite a bloated ensemble, as my initial goal was to test different encoders and median composites. Moving forward an ensemble of a few <code>se_resnext50_32x4d</code> and <code>efficientnet-b{7,8}</code> models trained on a full dataset (i.e. without median compositing) would probably provide a similar RMSE score if not better.

### **2. Prerequisites**
* Windows/Linux
* 32 Gb RAM
* 24GB VRAM GPU (e.g. RTX 3090)

### **3. Install dependencies**
```bash
conda env create -f biomassters.yaml
conda activate biomassters
```

### **4. Download data**
```bash
aws s3 cp s3://drivendata-competition-biomassters-public-as/train_features ./train_features --no-sign-request --recursive
aws s3 cp s3://drivendata-competition-biomassters-public-as/test_features ./test_features --no-sign-request --recursive
aws s3 cp s3://drivendata-competition-biomassters-public-as/train_agbm ./train_agbm --no-sign-request --recursive
```

### **5. Project structure**  
Run:
* <code>1_preprocessing.ipynb</code> for pre-processing *Sentinel-1* and *Sentinel-2* images into 6 cloud-free median composites
* <code>2_train.ipynb</code> for training and fine-tuning 15 *UNet++* models
* <code>3_inference.ipynb</code> for generating 15 average model predictions
* <code>4_ensemble.ipynb</code> for ensembling 15 model predictions using weighted average

### **6. Results**
Final model weights could be downloaded here (should be stored in <code>./models</code>):  
https://drive.google.com/drive/folders/1rHulX-OK0VxYBiOII1Dr8TGyif27OlGr?usp=sharing

| Model | Data | Composite | Encoder |Attention | RMSE | 
| --- | --- | --- | ---| --- | --- | 
| 1 | val  | 4S | se_resnext50_32x4d | - | 28.7029 | 
| 2 | val  | 4S | se_resnext101_32x4d | - | 28.9442 |  
| 3 | val  | 4S | se_resnext50_32x4d | scse | 29.0036 |
| 4 | val  | 3S | se_resnext50_32x4d | scse | 29.0204 |
| 5 | val  | 4S | efficientnet-b6 | - | 29.0719 | 
| 6 | val  | 4SI | efficientnet-b5 | - | 29.1720 |  
| 7 | val  | 4S | xception | - | 29.2953 |
| 8 | val  | 2SI | se_resnext50_32x4d | - | 29.3084 |
| 9 | val  | 2S | se_resnext50_32x4d | - | 29.3894 | 
| 10 | val  | 6S | timm-efficientnet-b7 | - | **28.5587** |  
| 11 | val  | 6S | timm-efficientnet-b8 | - | 28.5591 |
| 12 | val  | 6S | se_resnext50_32x4d | - | 28.7986 |
| 13 | val  | 4S | timm-efficientnet-b8 | - | 28.9115 | 
| 14 | val  | 4SI | efficientnet-b7 | - | 29.1265 |  
| 15 | val  | 4S | senet154 | - | 29.1071 |
| Ensemble | val | - | - | - | **27.8108** |
| Ensemble | public | - | - | - | **27.7930** |
| Ensemble | private | - | - | - | **28.0438** |