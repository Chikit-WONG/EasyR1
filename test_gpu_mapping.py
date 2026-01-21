#!/usr/bin/env python3
"""
Test script to verify GPU mapping for multi-GPU training
"""
import os
import torch
import torch.distributed as dist
from datetime import datetime

def test_gpu_mapping():
    """Test GPU mapping by printing device info"""
    
    # Print environment variables
    print(f"[{datetime.now()}] === Environment Variables ===")
    print(f"RANK: {os.getenv('RANK', 'not set')}")
    print(f"WORLD_SIZE: {os.getenv('WORLD_SIZE', 'not set')}")
    print(f"LOCAL_RANK: {os.getenv('LOCAL_RANK', 'not set')}")
    print(f"RAY_LOCAL_RANK: {os.getenv('RAY_LOCAL_RANK', 'not set')}")
    print(f"LOCAL_WORLD_SIZE: {os.getenv('LOCAL_WORLD_SIZE', 'not set')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    # Print CUDA device info
    print(f"\n[{datetime.now()}] === CUDA Device Info ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Current Device Name: {torch.cuda.get_device_name()}")
    
    # Print memory info
    if torch.cuda.is_available():
        print(f"\n[{datetime.now()}] === GPU Memory Info ===")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            print(f"GPU {i}: {props.name}")
            print(f"  Total Memory: {total_memory:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
    
    # Test distributed setup
    print(f"\n[{datetime.now()}] === Distributed Setup Test ===")
    try:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            print("NCCL process group initialized successfully")
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Distributed Rank: {rank}")
        print(f"Distributed World Size: {world_size}")
        
        # Test barrier
        dist.barrier()
        print(f"Barrier test passed at rank {rank}")
        
    except Exception as e:
        print(f"Distributed setup error: {e}")
    
    print(f"\n[{datetime.now()}] === Test Complete ===")

if __name__ == "__main__":
    test_gpu_mapping()
