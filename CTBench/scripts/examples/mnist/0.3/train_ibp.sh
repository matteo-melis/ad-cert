#!/bin/bash

gpu_idx=4
dataset=mnist
net=cnn_7layer_bn
L1=2e-6
robust_weight=0.7

train_eps=0.3
test_eps=0.3

# --- IBP ---

init=fast
fast_reg=0.5

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-ibp-training --fast-reg $fast_reg --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 50 60 --train-eps $train_eps --test-eps $test_eps --train-batch 256 --test-batch 256 --grad-clip 10 --n-epochs 70 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight --start-epoch-eps 0 --end-epoch-eps 20  --save-dir ./benchmark_models