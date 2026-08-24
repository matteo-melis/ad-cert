#!/bin/bash

gpu_idx=6
dataset=cifar10
net=cnn_7layer_bn
L1=1e-6
robust_weight=0.8

train_eps=0.007843137255
test_eps=0.007843137255

# --- IBP ---

init=fast
fast_reg=0.5

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-ibp-training --fast-reg $fast_reg --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 120 140 --train-eps $train_eps --test-eps $test_eps --train-batch 128 --test-batch 128 --grad-clip 10 --n-epochs 160 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight --start-epoch-eps 1 --end-epoch-eps 81  --save-dir ./benchmark_models