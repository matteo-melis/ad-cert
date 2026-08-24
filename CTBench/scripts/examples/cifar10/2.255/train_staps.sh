#!/bin/bash

gpu_idx=0
dataset=cifar10
net=cnn_7layer_bn
L1=1e-6
robust_weight=1

train_eps=0.007843137255
test_eps=0.007843137255
train_steps=8
test_steps=8
restarts=3

init=fast
fast_reg=0.5

# ---- TAPS ----
block_sizes="17 4"
taps_grad_scale=2
soft_thre=0.5
eps_shrinkage=0.1
relu_shrinkage=0.8

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-taps-training --use-small-box --eps-shrinkage $eps_shrinkage --relu-shrinkage $relu_shrinkage --block-sizes $block_sizes --taps-grad-scale $taps_grad_scale --soft-thre $soft_thre --fast-reg $fast_reg --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 120 140 --train-eps $train_eps --test-eps $test_eps --train-steps $train_steps --test-steps $test_steps  --restarts $restarts --train-batch 128 --test-batch 128 --grad-clip 10 --n-epochs 160 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight --start-epoch-eps 1 --end-epoch-eps 81  --save-dir ./benchmark_models/ 
