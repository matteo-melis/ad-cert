#!/bin/bash

gpu_idx=4
dataset=mnist
init=default
net=cnn_7layer_bn
L1=1e-5
robust_weight=0

# --- Standard training ---

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-std-training --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 50 60 --train-eps 0 --test-eps 0 --train-steps 20 --test-steps 20 --train-batch 256 --test-batch 256 --grad-clip 10 --n-epochs 70 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight --save-dir ./benchmark_models/