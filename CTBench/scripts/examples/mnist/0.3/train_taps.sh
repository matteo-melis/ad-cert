#!/bin/bash

gpu_idx=5
dataset=mnist
net=cnn_7layer_bn
L1=0
robust_weight=1

train_eps=0.4
test_eps=0.3
train_steps=5
test_steps=5
restarts=3

init=fast
fast_reg=0.5

# ---- TAPS ----
block_sizes="13 8"
taps_grad_scale=4
soft_thre=0.5

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-taps-training --block-sizes $block_sizes --taps-grad-scale $taps_grad_scale --soft-thre $soft_thre --fast-reg $fast_reg --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 50 60 --train-eps $train_eps --test-eps $test_eps --train-steps $train_steps --test-steps $test_steps  --restarts $restarts --train-batch 256 --test-batch 256 --grad-clip 10 --n-epochs 70 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight --start-epoch-eps 0 --end-epoch-eps 20  --save-dir ./benchmark_models/ 
