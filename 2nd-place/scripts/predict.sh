#!/bin/bash


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
