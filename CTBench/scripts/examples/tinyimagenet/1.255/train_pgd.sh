#!/bin/bash

gpu_idx=1
dataset=tinyimagenet
init=default
net=cnn_7layer_bn_tinyimagenet
L1=1e-5
robust_weight=1

train_eps=0.00392156863
test_eps=0.00392156863
train_steps=1
test_steps=1
restarts=3

# --- vanilla PGD ---

# CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-pgd-training --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 120 140 --train-eps $train_eps --test-eps $test_eps --train-steps $train_steps --test-steps $test_steps --restarts $restarts --train-batch 128 --test-batch 128 --grad-clip 10 --n-epochs 160 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight  --start-epoch-eps 1 --end-epoch-eps 81  --save-dir ./benchmark_models/  

# --- PGD with EDAC step size ---

EDAC_step_size=0.1

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-pgd-training --use-EDAC-step --EDAC-step-size $EDAC_step_size --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 120 140 --train-eps $train_eps --test-eps $test_eps --train-steps $train_steps --test-steps $test_steps --restarts $restarts --train-batch 128 --test-batch 128 --grad-clip 10 --n-epochs 160 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight  --start-epoch-eps 1 --end-epoch-eps 81  --save-dir ./benchmark_models/  