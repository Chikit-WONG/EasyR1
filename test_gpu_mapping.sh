#!/bin/bash

# Test script to verify multi-GPU setup
# This script will test the GPU mapping with 2 GPUs

MODEL_PATH=/hpc2hdd/home/ckwong627/workdir/models/Qwen2.5-VL-7B-Instruct

echo "Testing GPU mapping with 2 GPUs..."
echo "=================================="

# Test with distributed launcher
torchrun --nproc_per_node=2 /hpc2hdd/home/ckwong627/workdir/EasyR1/test_gpu_mapping.py
