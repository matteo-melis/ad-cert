#!/bin/bash

dataset=cifar10
net=cnn_7layer_bn
L1=2e-6
robust_weight=1

train_eps=0.007843137255
test_eps=0.007843137255
train_steps=5
test_steps=5
restarts=3

init=fast
fast_reg=0.5

# ---- TAPS ----
block_sizes="4 17"
taps_grad_scale=5
soft_thre=0.5
relu_shrinkage=0.8

cmds=()
for m in  STAPS 
do
    for eps in 0.00784313725
    do
        c3="python3 get_stat.py \
        --dataset cifar10 \
        --net cnn_7layer_bn \
        --train-eps $eps --test-eps $eps \
        --load-model ../CTBenchRelease/cifar10/2.255/$m/model.ckpt \
        --train-batch 256 --test-batch 256"
        cmds+=("$c3")
    done
done

gpu_list=(1 2 3 4 5 6)

# use lscpu to ensure these are on the same socket to avoid communication overhead
cpu_list=(12-23 24-35  36-47 0-11 )

num_cmds=${#cmds[@]}
echo "Commands in total will run: $num_cmds"
echo "Number of parallel tasks for each GPU?"
read max_parallel_task
echo "Estimated time for each process in secs?"
read estimation
secs=$(($estimation * $num_cmds / $max_parallel_task))
echo "Your estimated execution time:"

printf '%02dh:%02dm:%02fs\n' $(echo -e "$secs/3600\n$secs%3600/60\n$secs%60"| bc)

echo "Running in a screen is highly recommended. Proceed? y/n: "
read decision

if [ $decision != "y" ]
then
    exit
else
    echo "Your job will start now. Good luck!"
    sleep 1
fi


gpu_prefix="CUDA_VISIBLE_DEVICES"
num_gpus=${#gpu_list[@]}
num_cpus=${#cpu_list[@]}

for ((i = 0; i < ${#cmds[@]}; i++))
do
    gpu_index="$(($i % ($num_gpus * $max_parallel_task) ))"
    cpu_index="$(($i % $num_cpus ))"
    c="$gpu_prefix=${gpu_list[$(($gpu_index / $max_parallel_task))]} taskset -c ${cpu_list[$cpu_index]} ${cmds[$i]}"
    if [ "$(( $(($i + 1)) % $(($num_gpus * $max_parallel_task)) ))" == "0" ] || [ "$(($i + 1))" == $num_cmds ]
    then
        true
    else
        c="$c &"
    fi
    eval " $c"
done