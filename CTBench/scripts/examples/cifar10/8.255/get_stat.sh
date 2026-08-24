#!/bin/bash

# ---- example usage of get_stat.py ----

CUDA_VISIBLE_DEVICES=4 python3 get_stat.py \
    --dataset cifar10 \
    --net cnn_7layer_bn \
    --train-eps 0.03137254901 --test-eps 0.03137254901 \
    --load-model ../CTBenchRelease/cifar10/Standard/model.ckpt \
    --train-batch 256 --test-batch 256