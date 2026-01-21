# 多卡部署GPU映射问题修复指南

## 问题分析

从错误日志中可以看到，尽管配置了 `trainer.n_gpus_per_node=2`，但系统仍然只使用了GPU 0：

```
[rank1]:[W113 00:17:02.558000772 ProcessGroupNCCL.cpp:4115] [PG ID 0 PG GUID 0 Rank 1]  using GPU 0 to perform barrier
[rank0]:[W113 00:17:38.526420693 ProcessGroupNCCL.cpp:4115] [PG ID 0 PG GUID 0 Rank 0]  using GPU 0 to perform barrier
```

以及内存溢出错误：
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.23 GiB. GPU 0 has a total capacity of 47.40 GiB
```

**根本原因**：在 `verl/single_controller/base/worker.py` 中，Worker的 `__init__` 方法只为AMD GPU设置了 `torch.cuda.set_device()`，但对于NVIDIA GPU没有类似的设置，导致所有rank都默认使用GPU 0。

## 修复方案

### 修改文件：`verl/single_controller/base/worker.py`

在Worker的`__init__`方法中添加对NVIDIA GPU的device设置：

```python
def __init__(self, cuda_visible_devices=None) -> None:
    # construct a meta from envrionment variable
    world_size = int(os.getenv("WORLD_SIZE"))
    rank = int(os.getenv("RANK"))
    self._rank = rank
    self._world_size = world_size

    # Get local rank from environment variables
    local_rank = int(os.getenv("LOCAL_RANK", os.getenv("RAY_LOCAL_RANK", "0")))
    
    if "AMD" in torch.cuda.get_device_name():
        os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("ROCR_VISIBLE_DEVICES")
        os.environ["LOCAL_RANK"] = os.getenv("RAY_LOCAL_RANK")
        cuda_visible_devices = os.getenv("LOCAL_RANK", "0")
        torch.cuda.set_device(int(cuda_visible_devices))
    else:
        # For NVIDIA GPUs, set device based on local rank
        torch.cuda.set_device(local_rank)
    
    # ... rest of the code
```

关键变化：
1. 首先获取 `local_rank`（从 `LOCAL_RANK` 或 `RAY_LOCAL_RANK` 环境变量）
2. 对于NVIDIA GPU，在else分支中调用 `torch.cuda.set_device(local_rank)` 来正确映射rank到对应的GPU

## 验证修复

### 方式1：使用测试脚本

已创建了 `test_gpu_mapping.py` 和 `test_gpu_mapping.sh` 来验证GPU映射：

```bash
cd /hpc2hdd/home/ckwong627/workdir/EasyR1
bash test_gpu_mapping.sh
```

这个脚本会打印出：
- 环境变量（RANK, WORLD_SIZE, LOCAL_RANK等）
- CUDA设备信息
- 每张GPU的内存使用情况
- 分布式训练的rank和world_size

### 方式2：查看训练日志

在修复后运行多卡训练时，应该看到：
- 两个rank都在初始化，而不是只有一个
- NCCL的barrier警告应该消失或者rank应该映射到不同的GPU
- GPU内存应该在两张卡之间分散，而不是全部集中在GPU 0

## 部署建议

修复后，使用以下命令进行多卡训练：

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

## 其他优化建议

1. **减少DataLoader worker数量**：日志中提到 "This DataLoader will create 8 worker processes"，在只有2张GPU的情况下可能过多。考虑调整配置中的num_workers。

2. **Flash Attention dtype问题**：日志中提到Flash Attention 2不支持torch.float32。可以在配置中显式设置dtype为torch.float16或torch.bfloat16。

3. **监控GPU内存**：使用 `nvidia-smi` 或在训练过程中监控GPU内存使用，确保两张卡都被充分利用。

## 文件修改总结

- **修改文件**：`/hpc2hdd/home/ckwong627/workdir/EasyR1/verl/single_controller/base/worker.py`
- **修改行数**：第125-147行
- **改动内容**：为NVIDIA GPU添加了 `torch.cuda.set_device(local_rank)` 调用
