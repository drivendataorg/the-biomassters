#!/usr/bin/env sh


set -eu  # o pipefail

GPU=${GPU:-0,1}
PORT=${PORT:-29500}
N_GPUS=${N_GPUS:-1} # change to your number of GPUs

OPTIM=adamw
LR=0.001
WD=0.01

SCHEDULER=cosa
MODE=epoch

N_EPOCHS=800
T_MAX=800
loss=nrmse
attn=scse
data_dir=./data
chkps_dir=./models

backbone=tf_efficientnetv2_l_in21k
BS=8
FOLD=0

CHECKPOINT=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7
MASTER_PORT="${PORT}" CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node="${N_GPUS}" \
    ./src/train.py \
        --train-df $data_dir/features_metadata.csv \
        --train-images-dir $data_dir/train_features \
        --train-labels-dir $data_dir/train_agbm \
        --backbone "${backbone}" \
        --loss "${loss}" \
        --in-channels 15 \
        --optim "${OPTIM}" \
        --learning-rate "${LR}" \
        --weight-decay "${WD}" \
        --scheduler "${SCHEDULER}" \
        --T-max "${T_MAX}" \
        --num-epochs "${N_EPOCHS}" \
        --checkpoint-dir "${CHECKPOINT}" \
        --fold "${FOLD}" \
        --scheduler-mode "${MODE}" \
        --batch-size "${BS}" \
        --augs \
        --dec-attn-type $attn \
        --dec-channels 384 368 352 336 320 \
        --fp16 \


LR=0.0001
N_EPOCHS=100
T_MAX=100
CHECKPOINT_LOAD=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7
CHECKPOINT=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7_plus800eb

MASTER_PORT="${PORT}" CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node="${N_GPUS}" \
    ./src/train.py \
        --train-df $data_dir/features_metadata.csv \
        --train-images-dir $data_dir/train_features \
        --train-labels-dir $data_dir/train_agbm \
        --backbone "${backbone}" \
        --loss "${loss}" \
        --in-channels 15 \
        --optim "${OPTIM}" \
        --learning-rate "${LR}" \
        --weight-decay "${WD}" \
        --scheduler "${SCHEDULER}" \
        --T-max "${T_MAX}" \
        --num-epochs "${N_EPOCHS}" \
        --checkpoint-dir "${CHECKPOINT}" \
        --fold "${FOLD}" \
        --scheduler-mode "${MODE}" \
        --batch-size "${BS}" \
        --load $CHECKPOINT_LOAD/model_last.pth \
        --augs \
        --dec-attn-type $attn \
        --dec-channels 384 368 352 336 320 \
        --fp16 \


N_EPOCHS=100
T_MAX=100
CHECKPOINT_LOAD=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7_plus800eb
CHECKPOINT=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7_plus800eb_100ft
MASTER_PORT="${PORT}" CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node="${N_GPUS}" \
    ./src/train.py \
        --train-df $data_dir/features_metadata.csv \
        --train-images-dir $data_dir/train_features \
        --train-labels-dir $data_dir/train_agbm \
        --backbone "${backbone}" \
        --loss "${loss}" \
        --in-channels 15 \
        --optim "${OPTIM}" \
        --learning-rate "${LR}" \
        --weight-decay "${WD}" \
        --scheduler "${SCHEDULER}" \
        --T-max "${T_MAX}" \
        --num-epochs "${N_EPOCHS}" \
        --checkpoint-dir "${CHECKPOINT}" \
        --fold "${FOLD}" \
        --scheduler-mode "${MODE}" \
        --batch-size "${BS}" \
        --load $CHECKPOINT_LOAD/model_last.pth \
        --augs \
        --dec-attn-type $attn \
        --dec-channels 384 368 352 336 320 \
        --fp16 \
        --ft \


CHECKPOINT_LOAD=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7_plus800eb_100ft
CHECKPOINT=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7_plus800eb_200ft
MASTER_PORT="${PORT}" CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node="${N_GPUS}" \
    ./src/train.py \
        --train-df $data_dir//features_metadata.csv \
        --train-images-dir $data_dir/train_features \
        --train-labels-dir $data_dir/train_agbm \
        --backbone "${backbone}" \
        --loss "${loss}" \
        --in-channels 15 \
        --optim "${OPTIM}" \
        --learning-rate "${LR}" \
        --weight-decay "${WD}" \
        --scheduler "${SCHEDULER}" \
        --T-max "${T_MAX}" \
        --num-epochs "${N_EPOCHS}" \
        --checkpoint-dir "${CHECKPOINT}" \
        --fold "${FOLD}" \
        --scheduler-mode "${MODE}" \
        --batch-size "${BS}" \
        --load $CHECKPOINT_LOAD/model_last.pth \
        --augs \
        --dec-attn-type $attn \
        --dec-channels 384 368 352 336 320 \
        --fp16 \
        --ft \
