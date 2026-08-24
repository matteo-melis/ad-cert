#!/bin/bash

gpu_idx=0
dataset=mnist
init=default
net=cnn_3layer_bn
L1=1e-6
robust_weight=1

train_eps=0.2
test_eps=0.1
train_steps=5
test_steps=5
restarts=3

arow_reg_weight=2
arow_label_smoothing=0.2

# --- AROW ---

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-arow-training --arow-reg-weight $arow_reg_weight --arow-label-smoothing $arow_label_smoothing  --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 50 60 --train-eps $train_eps --test-eps $test_eps --train-steps $train_steps --test-steps $test_steps --train-batch 256 --test-batch 256 --grad-clip 10 --n-epochs 70 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight --restarts $restarts --start-epoch-eps 0 --end-epoch-eps 20  --save-dir ./benchmark_models/   

# --- AROW with EDAC step size ---

EDAC_step_size=0.1

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-arow-training --arow-reg-weight $arow_reg_weight --arow-label-smoothing $arow_label_smoothing --use-EDAC-step --EDAC-step-size $EDAC_step_size  --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 50 60 --train-eps $train_eps --test-eps $test_eps --train-steps $train_steps --test-steps $test_steps --train-batch 256 --test-batch 256 --grad-clip 10 --n-epochs 70 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight --restarts $restarts --start-epoch-eps 0 --end-epoch-eps 20  --save-dir ./benchmark_models/   
