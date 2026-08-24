#!/bin/bash

gpu_idx=1
dataset=tinyimagenet
net=cnn_7layer_bn_tinyimagenet
L1=1e-6
robust_weight=1

train_eps=0.00392156863
test_eps=0.00392156863


# --- EXP-IBP ---

init=fast
fast_reg=0.2
ibp_coef=4e-1
train_steps=1
test_steps=1
restarts=1

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-expibp-training --ibp-coef $ibp_coef --model-selection None  --fast-reg $fast_reg --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 60 70 --train-eps $train_eps --test-eps $test_eps --train-steps $train_steps --test-steps $test_steps  --restarts $restarts --train-batch 128 --test-batch 128 --grad-clip 10 --n-epochs 80 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight --start-epoch-eps 1 --end-epoch-eps 21  --save-dir ./benchmark_models
