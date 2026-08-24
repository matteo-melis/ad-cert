#!/bin/bash

# ---- example usage of get_stat.py ----

CUDA_VISIBLE_DEVICES=5 python3 get_stat.py \
    --dataset mnist \
    --net cnn_7layer_bn \
    --train-eps 0.1 --test-eps 0.1 \
    --load-model ../CTBenchRelease/mnist/Standard/model.ckpt \
    --train-batch 256 --test-batch 256