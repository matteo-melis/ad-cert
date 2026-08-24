#!/bin/bash

gpu_idx=2
dataset=mnist
init=default
net=cnn_7layer_bn
L1=1e-5
robust_weight=1

train_eps=0.2
test_eps=0.1
train_steps=10
test_steps=10
restarts=3

# --- vanilla PGD ---

# CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-pgd-training --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 50 60 --train-eps $train_eps --test-eps $test_eps --train-steps $train_steps --test-steps $test_steps  --restarts $restarts --train-batch 256 --test-batch 256 --grad-clip 10 --n-epochs 70 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight --start-epoch-eps 0 --end-epoch-eps 20  --save-dir ./benchmark_models_more_steps

# --- PGD with EDAC step size ---

EDAC_step_size=0.3

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-pgd-training --use-EDAC-step --EDAC-step-size $EDAC_step_size --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 50 60 --train-eps $train_eps --test-eps $test_eps --train-steps $train_steps --test-steps $test_steps --restarts $restarts --train-batch 256 --test-batch 256 --grad-clip 10 --n-epochs 70 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight  --start-epoch-eps 0 --end-epoch-eps 20  --save-dir ./benchmark_models/  