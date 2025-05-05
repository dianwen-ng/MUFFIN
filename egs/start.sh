#!/bin/bash
source path.sh
set -e

log_root="model75hz"
config="configs/config_24k_512codes_75hz.json"
input_training_file="data/train" 
input_validation_file="data/valid"

mode=train

if [ "${mode}" == "debug" ]; then
  ## debug
  echo "Debug"
  log_root=${log_root}_debug
  export CUDA_VISIBLE_DEVICES=0
  python ${MAIN_ROOT}/train.py \
    --config ${config} \
    --checkpoint_path ${log_root} \
    --input_training_file ${input_training_file} \
    --input_validation_file ${input_validation_file} \
    --checkpoint_interval 100 \
    --summary_interval 10 \
    --validation_interval 100 \

elif [ "$mode" == "train" ]; then
  ## train
  echo "Train model..."
  export CUDA_VISIBLE_DEVICES=0,1
  python ${MAIN_ROOT}/train.py \
    --config ${config} \
    --checkpoint_path ${log_root} \
    --input_training_file ${input_training_file} \
    --input_validation_file ${input_validation_file} \
    --checkpoint_interval 1250 \
    --summary_interval 100 \
    --validation_interval 2500
fi
