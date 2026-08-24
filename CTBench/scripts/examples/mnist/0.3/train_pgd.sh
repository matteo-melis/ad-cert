#!/bin/bash

gpu_idx=4
dataset=mnist
init=default
net=cnn_7layer_bn
L1=5e-6
robust_weight=1

train_eps=0.3
test_eps=0.3
train_steps=5
test_steps=5
restarts=3

# --- vanilla PGD ---

# CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-pgd-training --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 50 60 --train-eps $train_eps --test-eps $test_eps --train-steps $train_steps --test-steps $test_steps  --restarts $restarts --train-batch 256 --test-batch 256 --grad-clip 10 --n-epochs 70 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight --start-epoch-eps 0 --end-epoch-eps 20  --save-dir ./benchmark_models

# --- PGD with EDAC step size ---

EDAC_step_size=0.3

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-pgd-training --use-EDAC-step --EDAC-step-size $EDAC_step_size --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 50 60 --train-eps $train_eps --test-eps $test_eps --train-steps $train_steps --test-steps $test_steps --restarts $restarts --train-batch 256 --test-batch 256 --grad-clip 10 --n-epochs 70 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight  --start-epoch-eps 0 --end-epoch-eps 20  --save-dir ./benchmark_models/  