#!/bin/bash

# ---- example usage of get_stat.py ----

CUDA_VISIBLE_DEVICES=3 python3 get_stat.py \
    --dataset tinyimagenet \
    --net cnn_7layer_bn_tinyimagenet \
    --train-eps 0.00392156863 --test-eps 0.00392156863 \
    --load-model ../CTBenchRelease/tinyimagenet/1.255/STAPS/model.ckpt \
    --train-batch 128 --test-batch 128