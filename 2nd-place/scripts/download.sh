#!/bin/bash


s3_node=us # change to your nearest node
split=train
download_root=./data/source
features_metadata=./data/information/features_metadata.csv
training_labels_metadata=./data/information/train_agbm_metadata.csv

python download.py \
    --download_root            $download_root            \
    --features_metadata        $features_metadata        \
    --training_labels_metadata $training_labels_metadata \
    --s3_node                  $s3_node                  \
    --split                    $split
