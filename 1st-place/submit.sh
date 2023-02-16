#!/usr/bin/env sh


python \
    ./src/submit.py \
    --test-df ./data/features_metadata.csv \
    --test-images-dir ./data/test_features \
    --model-path ./models/tf_efficientnetv2_l_in21k_f0_b8x2_e100_nrmse_devscse_attnlin_augs_decplus7_plus800eb_200ft/modelo_best.pth \
    --tta 4 \
    --batch-size 16 \
    --out-dir ./preds \
