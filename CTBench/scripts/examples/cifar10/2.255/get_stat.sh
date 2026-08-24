#!/bin/bash

# ---- example usage of get_stat.py ----

CUDA_VISIBLE_DEVICES=4 python3 get_stat.py \
    --dataset cifar10 --use-ibp-training --use-small-box --eps-shrinkage 0.1 --relu-shrinkage 0.8 --train-steps 5 --test-steps 5 --restarts 3 \
    --net cnn_7layer_bn \
    --train-eps 0.00784313725 --test-eps 0.00784313725 \
    --load-model ../cert-base-master/benchmark_models/cifar10/eps0.0078431/train_eps0.011764/SABR_trained/cnn_7layer_bn/init_fast/fast_reg_0.5/eps_shrink_0.1/relu_shrink_0.8/L1_1e-06/pop_bn_stats/model.ckpt \
    --train-batch 128 --test-batch 128