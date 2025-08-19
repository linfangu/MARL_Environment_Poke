#!/bin/bash

training_phase=$1 
output_dir=${2:-"./results/attention/single1"}
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

fi

if [[ -n "$load_policy" ]]; then
  cmd+=" --load_policy ${load_policy}"
fi

eval $cmd

