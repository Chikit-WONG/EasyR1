#!/bin/bash

# Simple script to check GPU allocation
# Run this before starting multi-GPU training

echo "================================"
echo "GPU Allocation Check"
echo "================================"

# Check total GPUs
echo "Total GPUs available:"
nvidia-smi -L | wc -l
echo ""

# Check GPU memory
echo "GPU Memory Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader
echo ""

# Check compute capability (for CUDA support)
echo "GPU Compute Capability:"
nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader
echo ""

# Check NCCL communication support
echo "Checking NCCL support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} (Compute Capability: {props.major}.{props.minor})')
"
echo ""

# Check if distributed backend is available
echo "Checking distributed backend..."
python3 -c "
import torch.distributed as dist
print(f'NCCL available: {dist.is_available()}')
print(f'Default backend: {dist.get_backend()}')
"
