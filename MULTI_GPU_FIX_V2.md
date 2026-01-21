# 多卡部署GPU映射修复 - 第二版

## 问题分析

从新的错误日志中可以看到：
```
RuntimeError: CUDA error: invalid device ordinal
```

这是因为在Ray的placement group context中：
1. Ray通过 `placement_group_bundle_idx` 分配GPU资源
2. 但**不会**自动设置 `CUDA_VISIBLE_DEVICES` 或调用 `torch.cuda.set_device()`
3. 所有worker进程仍然能看到机器上的所有GPU
4. torch.distributed.init_process_group() 不知道应该使用哪个GPU

NCCL警告消息清楚地指出了问题：
```
[rank1]: using GPU 0 to perform barrier as devices used by this process are currently unknown.
Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.
```

## 修复方案

### 修改文件：`verl/workers/fsdp_workers.py`

在 `FSDPWorker.__init__()` 中，在 `dist.init_process_group()` 调用**之前**，根据 `LOCAL_RANK` 或 `RAY_LOCAL_RANK` 设置正确的GPU device：

```python
if not dist.is_initialized():
    # For multi-GPU training with Ray, set the device before init_process_group
    local_rank = int(os.getenv("LOCAL_RANK", os.getenv("RAY_LOCAL_RANK", "0")))
    device_count = torch.cuda.device_count()
    if local_rank < device_count:
        torch.cuda.set_device(local_rank)
    
    dist.init_process_group(backend="nccl")
```

**关键改进**：
1. 安全地获取local_rank（优先使用RAY_LOCAL_RANK，备用LOCAL_RANK）
2. **验证device_id有效性**（`if local_rank < device_count`）
3. 在init_process_group()之前调用set_device()
4. 这样NCCL会正确识别每个rank的GPU

### 相关文件改动
- 添加 `import os` 到导入列表

## 为什么是这个位置？

**关键观察**：
- Worker 的 `__init__` 太早调用set_device可能与Ray的GPU分配冲突
- FSDPWorker 的 `__init__` 是分布式训练真正开始的地方
- `dist.init_process_group()` 需要知道当前进程的GPU device才能正确初始化NCCL
- 在 `super().__init__()` 之后但在 `dist.init_process_group()` 之前是最佳位置

## 工作流程

```
Ray创建 FSDPWorker
    ↓
FSDPWorker.__init__()
    ↓
super().__init__() 初始化基础Worker
    ↓
获取 local_rank（从RAY_LOCAL_RANK）
    ↓
torch.cuda.set_device(local_rank)  ← 告诉torch要使用哪个GPU
    ↓
dist.init_process_group()  ← NCCL现在知道正确的GPU
    ↓
正常训练 ← 每个rank使用正确的GPU
```

## 验证修复

修复后，应该看到：
1. ✅ 没有"invalid device ordinal"错误
2. ✅ NCCL barrier不再显示GPU映射未知的警告
3. ✅ 两个rank正确分配到不同的GPU
4. ✅ 模型可以成功初始化而不是OOM

## 文件清单

修改的文件：
- `verl/workers/fsdp_workers.py` - 在FSDPWorker.__init__中添加device设置
- `verl/single_controller/base/worker.py` - 保持原样（不在这里设置device）

---
最后修改: 2026-01-13
