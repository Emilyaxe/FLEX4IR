#!/bin/bash

for i in {0..9}; do
	python3 getdata4llvm.py $i
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file train.yaml train.py
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file zero2infer.yaml infer2.py
	python3 ../llvm_exp_script/runLLVMMultiple.py --iterator $i --max_generated_file_id 8
done