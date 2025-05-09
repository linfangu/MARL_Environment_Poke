#!/bin/bash

training_phase=$1 
output_dir=${2:-"./results/single1"}
resume=${3:-"False"} # False or name of folder 
load_policy=$4  # No default here, just check if it's set

GPUS="1"

# non-coop phase 
if [[ "${training_phase}" == "non_coop" ]]; then
  cmd="CUDA_VISIBLE_DEVICES=${GPUS} python train.py --gpu_id ${GPUS} --condition MultiAgentSing_fullobs \
    --output_dir ${output_dir} --l2_curr 0.1 --train_iter 4000 --resume ${resume}"

# coop phase 
elif [[ "${training_phase}" == "coop" ]]; then
  cmd="CUDA_VISIBLE_DEVICES=${GPUS} python train.py --gpu_id ${GPUS} --condition MultiAgentSync_fullobs \
    --output_dir ${output_dir} --l2_curr 0.1 --train_iter 4000 --resume ${resume}" 

# callout, non-coop phase 
elif [[ "${training_phase}" == "call_non_coop" ]]; then
  cmd="CUDA_VISIBLE_DEVICES=${GPUS} python train.py --gpu_id ${GPUS} --condition MultiAgentSing_call2 \
    --output_dir ${output_dir} --l2_curr 0.1 --train_iter 4000 --resume ${resume}" 

# callout, coop phase 
elif [[ "${training_phase}" == "call_coop" ]]; then
  cmd="CUDA_VISIBLE_DEVICES=${GPUS} python train.py --gpu_id ${GPUS} --condition MultiAgentSync_call2 \
    --output_dir ${output_dir} --l2_curr 0.1 --train_iter 4000 --resume ${resume} --miss_reward -0.1" 

# memory, non-coop phase
elif [[ "${training_phase}" == "memory_non_coop" ]]; then
  cmd="CUDA_VISIBLE_DEVICES=${GPUS} python train.py --gpu_id ${GPUS} --condition MultiAgentSing_memory \
    --output_dir ${output_dir} --l2_curr 0.1 --train_iter 4000 --resume ${resume} " 

# memory, coop phase
elif [[ "${training_phase}" == "memory_coop" ]]; then
  cmd="CUDA_VISIBLE_DEVICES=${GPUS} python train.py --gpu_id ${GPUS} --condition MultiAgentSync_memory \
    --output_dir ${output_dir} --l2_curr 0.1 --train_iter 4000 --resume ${resume} --pretrain --miss_reward -0.1" 

fi

if [[ -n "$load_policy" ]]; then
  cmd+=" --load_policy ${load_policy}"
fi

eval $cmd

