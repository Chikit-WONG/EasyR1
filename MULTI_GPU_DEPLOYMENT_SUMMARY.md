# EasyR1 多卡部署修复总结

## 问题描述

用户想要使用2张GPU进行多卡部署，但从日志中可以看到系统只使用了GPU 0，导致内存溢出错误。

**症状**：
- NCCL警告显示两个rank都在使用GPU 0
- 内存溢出：`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.23 GiB`
- GPU 0已分配43.26 GiB，只有2.95 GiB可用

## 根本原因

在 `verl/single_controller/base/worker.py` 中，Worker类的 `__init__` 方法只为AMD GPU设置了设备，但对于NVIDIA GPU没有进行设置，导致所有分布式rank都默认使用GPU 0。

## 实施的修复

### 修改文件
📄 **文件**: `verl/single_controller/base/worker.py` (第125-147行)

**关键改动**：
```python
# Get local rank from environment variables
local_rank = int(os.getenv("LOCAL_RANK", os.getenv("RAY_LOCAL_RANK", "0")))

if "AMD" in torch.cuda.get_device_name():
    # ... existing AMD code ...
else:
    # For NVIDIA GPUs, set device based on local rank
    torch.cuda.set_device(local_rank)
```

**工作原理**：
1. 从环境变量中获取当前进程的local rank
2. 对于NVIDIA GPU，调用 `torch.cuda.set_device(local_rank)` 将进程绑定到对应的GPU
3. 这样rank 0绑定到GPU 0，rank 1绑定到GPU 1，等等

## 验证步骤

### 1. 快速检查GPU设置
```bash
cd /hpc2hdd/home/ckwong627/workdir/EasyR1
bash check_gpu_setup.sh
```

### 2. 测试GPU映射
```bash
bash test_gpu_mapping.sh
```

应该看到类似输出：
```
=== Environment Variables ===
RANK: 0 (and 1 in separate process)
WORLD_SIZE: 2
LOCAL_RANK: 0 (and 1)
RAY_LOCAL_RANK: 0 (and 1)
...
=== CUDA Device Info ===
Current Device: 0 (and 1 in separate process)
```

### 3. 运行多卡训练
```bash
export MODEL_PATH=/hpc2hdd/home/ckwong627/workdir/models/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files=hiyouga/geometry3k@train \
  data.val_files=hiyouga/geometry3k@test \
  worker.actor.model.model_path=$MODEL_PATH \
  trainer.experiment_name=qwen2_5_vl_7b_geo_grpo \
  trainer.n_gpus_per_node=2
```

**预期结果**：
- 两个rank都应该成功初始化
- GPU内存应该在两张卡之间分散
- NCCL barrier不再报错关于GPU映射的问题
- 不再出现OOM错误

## 提供的工具脚本

### 1. `test_gpu_mapping.py`
测试脚本，显示GPU映射和分布式训练的详细信息。

### 2. `test_gpu_mapping.sh`
使用 `torchrun` 运行GPU映射测试的脚本。

### 3. `check_gpu_setup.sh`
检查系统GPU配置的脚本。

### 4. `MULTI_GPU_FIX.md`
详细的修复文档和部署建议。

## 其他注意事项

### Flash Attention dtype警告
日志中出现Flash Attention 2不支持float32的警告。建议在config中明确指定dtype：

```yaml
worker:
  actor:
    fsdp:
      torch_dtype: bfloat16  # or float16
```

### DataLoader worker数量
日志警告DataLoader创建了8个worker，在只有2张GPU时过多。可以调整配置中的 `num_workers` 参数。

### GPU内存管理
为了最大化内存效率，确保：
- `gpu_memory_utilization: 0.6` 已设置（config.yaml中已有）
- CPU offloading已启用（对于actor和ref model）
- FSDP full_shard已启用

## 文件清单

修改和创建的文件：
- ✅ `verl/single_controller/base/worker.py` - 主要修复
- ✨ `MULTI_GPU_FIX.md` - 详细修复文档
- ✨ `test_gpu_mapping.py` - GPU映射测试脚本
- ✨ `test_gpu_mapping.sh` - 测试运行脚本
- ✨ `check_gpu_setup.sh` - GPU设置检查脚本
- 📄 `MULTI_GPU_DEPLOYMENT_SUMMARY.md` - 本文档

## 下一步

1. 运行 `bash check_gpu_setup.sh` 确认GPU硬件配置
2. 运行 `bash test_gpu_mapping.sh` 验证GPU映射修复
3. 使用修复后的代码进行多卡训练
4. 监控GPU使用情况确保两张卡都被充分利用

如有问题，请查阅 `MULTI_GPU_FIX.md` 获取更多详细信息。
