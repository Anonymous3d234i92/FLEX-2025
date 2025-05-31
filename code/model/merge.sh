#!/bin/bash

for i in {0..100}; do
	python3 getdata.py $i
	accelerate launch --config_file train.yaml train.py
	accelerate launch --config_file zero2infer.yaml infer2.py
	python3 ../exp_srcipt/runMLIRMultiply.py --iterator $i --max_generated_file_id 2
done
