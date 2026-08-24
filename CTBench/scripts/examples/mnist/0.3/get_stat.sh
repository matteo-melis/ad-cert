#!/bin/bash

# ---- example usage of get_stat.py ----

CUDA_VISIBLE_DEVICES=1 python3 get_stat.py \
    --dataset mnist \
    --net cnn_7layer_bn \
    --train-eps 0.3 --test-eps 0.3 \
    --load-model ../cert-base-master/benchmark_models/mnist/eps0.3/TAPS_trained/cnn_7layer_bn/init_fast/fast_reg_0.5/TAPS_block_8_scale_4.0/L1_1e-06/model.ckpt \
    --train-batch 128 --test-batch 128