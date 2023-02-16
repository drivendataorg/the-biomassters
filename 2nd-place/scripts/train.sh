#!/bin/bash


device=0
folds=0,1,2,3,4
process=plain
config_file=./configs/swin_unetr/exp1.yaml

CUDA_VISIBLE_DEVICES=$device \
python train.py              \
    --data_root      ./data/source          \
    --exp_root       ./experiments/$process \
    --config_file    $config_file           \
    --process_method $process               \
    --folds          $folds
