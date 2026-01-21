# 快速参考：多卡部署问题修复

## 🎯 问题
多卡部署时只有GPU 0被使用，导致OOM错误

## ✅ 解决方案已实施

### 修改内容
**文件**: `verl/single_controller/base/worker.py`

在Worker.__init__中为NVIDIA GPU添加device设置：
```python
torch.cuda.set_device(local_rank)
```

## 🚀 快速验证

### 1. 检查GPU (30秒)
```bash
cd /hpc2hdd/home/ckwong627/workdir/EasyR1
bash check_gpu_setup.sh
```

### 2. 测试多GPU映射 (2-3分钟)
```bash
bash test_gpu_mapping.sh
```

### 3. 开始训练 (修复已生效)
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

## 📊 预期改进

**修复前**：
- ❌ 所有rank都在GPU 0上运行
- ❌ GPU 0内存已满，其他GPU空闲
- ❌ OOM错误

**修复后**：
- ✅ Rank 0在GPU 0，Rank 1在GPU 1
- ✅ GPU内存分散使用
- ✅ 不再出现OOM错误

## 📁 相关文档

- `MULTI_GPU_DEPLOYMENT_SUMMARY.md` - 完整总结
- `MULTI_GPU_FIX.md` - 详细修复指南
- `test_gpu_mapping.py` - GPU映射测试脚本
- `check_gpu_setup.sh` - GPU配置检查脚本

## 💡 提示

如果仍有问题，请检查：
1. `nvidia-smi` 显示的GPU数量是否正确
2. CUDA版本和PyTorch版本兼容性
3. 日志中是否还有"using GPU 0 to perform barrier"的警告

## 📞 技术细节

修复原理：
- Ray Worker在每个GPU上创建一个进程
- 通过RAY_LOCAL_RANK环境变量标识本地rank
- torch.cuda.set_device(local_rank)将进程绑定到对应GPU
- NCCL通信时能正确识别每个进程的GPU

---
最后修改: 2026-01-13
