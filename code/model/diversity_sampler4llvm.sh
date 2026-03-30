#!/bin/bash


INPUT_JSON="all_ll_files_ori.json"
OUTPUT_JSON="all_ll_files.json"

TARGET_COUNT=100000
INITIAL_COUNT=5000
CANDIDATE_COUNT=25000
SELECT_RATIO=0.2
BATCH_SIZE=12
MODEL_PATH="./codegen-2B"
CHECKPOINT_PATH="checkpointSearch4sample/"
#CHECKPOINT_PATH="checkpointSearch/"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file zero2infer.yaml diversity_sampler.py \
    --input_json ${INPUT_JSON} \
    --output_json ${OUTPUT_JSON} \
    --target_count ${TARGET_COUNT} \
    --initial_count ${INITIAL_COUNT} \
    --candidate_count ${CANDIDATE_COUNT} \
    --select_ratio ${SELECT_RATIO} \
    --batch_size ${BATCH_SIZE} \
    --model_path ${MODEL_PATH} \
    --checkpoint_path ${CHECKPOINT_PATH}
