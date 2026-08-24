#!/bin/bash

gpu_idx=5
dataset=tinyimagenet
net=cnn_7layer_bn_tinyimagenet
L1=2e-5
robust_weight=1

train_eps=0.00392156863
test_eps=0.00392156863

# --- IBP ---

init=fast
fast_reg=0.2

CUDA_VISIBLE_DEVICES=$gpu_idx python mix_train.py --use-pop-bn-stats --use-DPBox-training --use-loss-fusion --keep-fusion-when-test --fast-reg $fast_reg --init $init --dataset $dataset --net $net --lr 0.0005 --lr-milestones 120 140 --train-eps $train_eps --test-eps $test_eps --train-batch 128 --test-batch 128 --grad-clip 10 --n-epochs 160 --L1-reg $L1 --start-value-robust-weight $robust_weight --end-value-robust-weight $robust_weight --start-epoch-eps 1 --end-epoch-eps 81  --save-dir ./benchmark_models --grad-accu-batch 64 --model-selection None --use-swa --swa-start 150
