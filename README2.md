# EasyR1 项目结构与代码说明文档

## 📚 目录
- [项目概述](#项目概述)
- [整体架构](#整体架构)
- [目录结构详解](#目录结构详解)
- [核心模块说明](#核心模块说明)
- [配置文件说明](#配置文件说明)
- [使用流程](#使用流程)
- [多卡部署说明](#多卡部署说明)

---

## 项目概述

**EasyR1** 是一个高效、可扩展的多模态强化学习(RL)训练框架，专为大语言模型和视觉语言模型设计。它是原始 [veRL](https://github.com/volcengine/verl) 项目的清洁分支，增强了对视觉语言模型的支持。

### 核心特性
- **支持的模型类型**：
  - 语言模型：Llama3、Qwen2/2.5/3 系列
  - 视觉语言模型：Qwen2-VL/2.5-VL/3-VL
  - DeepSeek-R1 蒸馏模型

- **支持的算法**：
  - GRPO (Group Relative Policy Optimization)
  - DAPO (Data Augmented Policy Optimization)
  - Reinforce++
  - ReMax
  - RLOO (REINFORCE Leave One Out)
  - GSPO、CISPO (新增)

- **关键技术**：
  - **HybridEngine**：高性能分布式训练引擎
  - **vLLM SPMD模式**：支持高效的模型推理和采样
  - **Padding-free训练**：优化内存使用
  - **多种实验追踪**：Wandb、SwanLab、Mlflow、Tensorboard

---

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                   EasyR1 训练框架                        │
├─────────────────────────────────────────────────────────┤
│  Trainer Layer (训练器层)                               │
│  ├─ RayPPOTrainer: 主训练流程控制                       │
│  ├─ ResourcePoolManager: 资源池管理                    │
│  └─ DataLoader: 数据加载与批处理                        │
├─────────────────────────────────────────────────────────┤
│  Algorithm Layer (算法层)                               │
│  ├─ Core Algorithms: GRPO, DAPO, Reinforce++等        │
│  ├─ KL Controller: KL散度控制器                         │
│  └─ Advantage Estimator: 优势函数估计                  │
├─────────────────────────────────────────────────────────┤
│  Worker Layer (工作器层)                                │
│  ├─ FSDPWorker: 分布式训练worker (Actor/Critic)        │
│  ├─ RewardManager: 奖励函数计算                         │
│  └─ RolloutWorker: 策略展开与采样                      │
├─────────────────────────────────────────────────────────┤
│  Infrastructure Layer (基础设施层)                      │
│  ├─ Ray: 分布式调度框架                                 │
│  ├─ FSDP: 全分片数据并行                                │
│  ├─ vLLM: 高效推理引擎                                   │
│  └─ Protocol: 数据传输协议                              │
└─────────────────────────────────────────────────────────┘
```

---

## 目录结构详解

### 根目录文件
```
EasyR1/
├── README.md                              # 项目主说明文档
├── README2.md                             # 本文档：详细的结构与代码说明
├── LICENSE                                # Apache 2.0 许可证
├── pyproject.toml                         # Python项目配置（构建、代码风格等）
├── setup.py                               # 安装脚本
├── requirements.txt                       # Python依赖包列表
├── Makefile                               # 构建和测试命令
├── Dockerfile                             # Docker镜像构建文件
├── Dockerfile.legacy                      # 旧版Docker镜像
└── .pre-commit-config.yaml               # Git提交前的代码检查配置
```

### 核心源代码目录 - `verl/`
这是项目的核心代码目录，包含所有的训练逻辑和基础设施。

```
verl/
├── __init__.py                            # 包初始化
├── protocol.py                            # 定义数据传输协议 (DataProto类)
├── trainer/                               # 训练器模块
│   ├── main.py                           # 入口函数，启动Ray任务
│   ├── ray_trainer.py                    # Ray分布式训练器实现
│   ├── config.py                         # 训练配置类定义
│   ├── core_algos.py                     # RL算法核心实现 (GRPO/DAPO等)
│   ├── data_loader.py                    # 数据加载器
│   └── metrics.py                        # 训练指标计算
├── workers/                               # Worker节点实现
│   ├── fsdp_workers.py                   # FSDP分布式训练worker
│   ├── actor/                            # Actor模型相关
│   ├── critic/                           # Critic模型相关
│   ├── reward/                           # 奖励函数管理
│   ├── rollout/                          # 策略展开（生成采样数据）
│   └── sharding_manager/                 # 模型分片管理
├── single_controller/                     # 单控制器分布式框架
│   ├── base/                             # 基础类定义
│   │   ├── worker.py                     # Worker基类（GPU设备管理）
│   │   └── worker_group.py               # WorkerGroup管理类
│   └── ray/                              # Ray分布式实现
│       └── base.py                       # RayWorkerGroup实现
├── models/                                # 模型相关
│   ├── monkey_patch.py                   # 模型补丁修复
│   └── transformers/                     # Transformers模型适配
└── utils/                                 # 工具函数
    ├── tokenizer.py                      # 分词器工具
    ├── torch_functional.py               # PyTorch函数工具
    └── py_functional.py                  # Python函数工具
```

### 示例和脚本目录

```
examples/                                  # 示例配置和脚本
├── config.yaml                           # 默认配置文件
├── qwen2_5_vl_7b_geo3k_grpo.sh          # Qwen2.5-VL 7B GRPO训练脚本
├── qwen2_5_vl_7b_geo3k_dapo.sh          # Qwen2.5-VL 7B DAPO训练脚本
├── qwen3_vl_4b_geo3k_grpo.sh            # Qwen3-VL 4B训练脚本
├── format_prompt/                        # 提示词格式化模板
│   └── math.jinja                       # 数学问题的Jinja模板
├── reward_function/                      # 奖励函数定义
│   └── math_reward.py                   # 数学任务奖励函数
└── runtime_env.yaml                      # Ray运行时环境配置

scripts/                                   # 实用脚本
└── model_merger.py                       # 模型检查点合并工具

run_scripts/                               # 运行脚本
├── run_qwen2_5_vl_7b_geo3k_grpo.sh      # 快速运行脚本
└── run_qwen3_vl_4b_geo3k_grpo.sh        # 另一个快速运行脚本
```

### 测试和调试目录

```
tests/                                     # 单元测试
test_gpu_mapping.py                       # GPU映射测试脚本
test_gpu_mapping.sh                       # GPU映射测试Shell脚本
check_gpu_setup.sh                        # GPU设置检查脚本
```

### 文档目录

```
MULTI_GPU_DEPLOYMENT_SUMMARY.md           # 多GPU部署修复总结
MULTI_GPU_FIX.md                          # 多GPU问题修复文档V1
MULTI_GPU_FIX_V2.md                       # 多GPU问题修复文档V2
QUICK_FIX_GUIDE.md                        # 快速修复指南
```

### 其他目录

```
checkpoints/                               # 训练检查点存储目录
temp/                                      # 临时文件目录
assets/                                    # 资源文件（图片、文档等）
.git/                                      # Git版本控制
.github/                                   # GitHub配置（CI/CD等）
verl.egg-info/                            # Python包信息
```

---

## 核心模块说明

### 1. 训练入口 - `verl/trainer/main.py`

**功能**：整个训练流程的入口点

**核心类**：
- `Runner`: Ray远程任务类，负责初始化和运行训练

**主要流程**：
```python
def run(config: PPOConfig):
    1. 解析配置文件
    2. 初始化tokenizer和processor（分词器、处理器）
    3. 定义Worker类映射（Actor、Critic）
    4. 创建资源池管理器
    5. 初始化奖励函数管理器
    6. 创建数据加载器
    7. 实例化RayPPOTrainer
    8. 初始化所有workers
    9. 开始训练 (trainer.fit())
```

**使用方式**：
```bash
python3 -m verl.trainer.main config=examples/config.yaml [其他参数...]
```

### 2. 训练器 - `verl/trainer/ray_trainer.py`

**功能**：实现完整的PPO训练循环

**核心类**：
- `RayPPOTrainer`: 主训练器
- `ResourcePoolManager`: 管理GPU资源分配
- `Role`: 枚举不同的Worker角色（ActorRolloutRef, Critic）

**关键方法**：
- `init_workers()`: 初始化所有分布式workers
- `fit()`: 主训练循环
- `training_step()`: 单个训练步
- `validation()`: 验证步骤
- `save_checkpoint()`: 保存检查点

**训练流程**：
```
for epoch in epochs:
    for batch in train_dataloader:
        1. Rollout: Actor模型生成响应
        2. Compute Rewards: 计算奖励函数
        3. Compute Values: Critic模型估计值函数
        4. Compute Advantages: 计算优势函数
        5. Update Actor: 更新Actor模型
        6. Update Critic: 更新Critic模型
        7. Log Metrics: 记录训练指标
    
    if should_validate:
        validation()
    
    save_checkpoint()
```

### 3. 算法实现 - `verl/trainer/core_algos.py`

**功能**：实现各种RL算法的核心逻辑

**核心组件**：
- **KL控制器**：
  - `FixedKLController`: 固定KL系数
  - `AdaptiveKLController`: 自适应KL系数

- **优势估计器** (`AdvantageEstimator`):
  - `GAE`: 广义优势估计
  - `GRPO`: Group Relative Policy Optimization
  - `REINFORCE_PLUS_PLUS`: 增强版REINFORCE
  - `REMAX`: ReMax算法
  - `RLOO`: Leave-One-Out变体

**关键函数**：
```python
def compute_advantages(
    values,          # Critic预测的价值
    rewards,         # 环境返回的奖励
    adv_estimator   # 使用的估计器类型
) -> advantages    # 计算出的优势值
```

### 4. Worker实现 - `verl/workers/fsdp_workers.py`

**功能**：分布式训练的Worker节点

**核心类**：
- `FSDPWorker`: 继承自`Worker`基类，实现FSDP(Fully Sharded Data Parallel)训练

**主要功能**：
- 模型初始化和分片
- 前向传播计算
- 反向传播和梯度更新
- 模型权重同步

**关键方法**：
- `init_model()`: 初始化模型并应用FSDP
- `update_policy()`: 更新策略网络
- `compute_values()`: 计算价值函数
- `generate_sequences()`: 生成文本序列

### 5. 数据协议 - `verl/protocol.py`

**功能**：定义不同模块间的数据传输格式

**核心类**：
- `DataProto`: 数据协议类，基于TensorDict

**主要特性**：
- 支持批量数据传输
- 自动padding处理
- 支持序列化和反序列化
- 跨进程数据传递（Ray）

**数据结构示例**：
```python
DataProto({
    'input_ids': torch.Tensor,      # 输入token ids
    'attention_mask': torch.Tensor,  # 注意力掩码
    'labels': torch.Tensor,          # 标签
    'rewards': torch.Tensor,         # 奖励
    'advantages': torch.Tensor,      # 优势值
    # ... 更多字段
})
```

### 6. 分布式基础 - `verl/single_controller/`

#### `base/worker.py`
**功能**：Worker基类，处理GPU设备分配

**关键修复**（多GPU支持）：
```python
def __init__(self):
    # 获取local rank
    local_rank = int(os.getenv("LOCAL_RANK", os.getenv("RAY_LOCAL_RANK", "0")))
    
    if "AMD" in torch.cuda.get_device_name():
        # AMD GPU设备设置
        ...
    else:
        # NVIDIA GPU: 根据local_rank设置设备
        torch.cuda.set_device(local_rank)  # 修复多GPU部署问题
```

#### `ray/base.py`
**功能**：Ray分布式实现

**核心类**：
- `RayWorkerGroup`: 管理多个Ray远程Worker
- `RayResourcePool`: 资源池管理

### 7. 配置管理 - `verl/trainer/config.py`

**功能**：定义所有训练配置参数

**核心配置类**：
- `PPOConfig`: 顶层配置
- `AlgorithmConfig`: 算法配置
- `WorkerConfig`: Worker配置
- `DataConfig`: 数据配置
- `TrainerConfig`: 训练器配置

---

## 配置文件说明

### `examples/config.yaml`

这是主配置文件，包含所有训练参数：

```yaml
# 数据配置
data:
  train_files: hiyouga/math12k@train    # 训练数据集
  val_files: hiyouga/math12k@test       # 验证数据集
  prompt_key: problem                    # 问题字段名
  answer_key: answer                     # 答案字段名
  image_key: images                      # 图像字段名
  rollout_batch_size: 512               # Rollout批次大小
  format_prompt: ./examples/format_prompt/math.jinja  # 提示词模板

# 算法配置
algorithm:
  adv_estimator: grpo                    # 优势估计器类型
  kl_coef: 1.0e-2                       # KL散度系数
  use_kl_loss: true                     # 是否使用KL损失

# Worker配置
worker:
  actor:
    global_batch_size: 128               # 全局批次大小
    model:
      model_path: Qwen/Qwen2.5-7B-Instruct  # 模型路径
      enable_gradient_checkpointing: true    # 梯度检查点
    optim:
      lr: 1.0e-6                        # 学习率
      weight_decay: 1.0e-2              # 权重衰减
    fsdp:
      torch_dtype: amp_bf16             # 混合精度类型
  
  critic:
    # Critic配置（结构类似Actor）
    ...
  
  reward:
    reward_model: external_function     # 奖励模型类型
    reward_fn_path: examples/reward_function/math_reward.py  # 奖励函数路径

# 训练器配置
trainer:
  project_name: easyr1                  # 项目名称
  experiment_name: test_experiment      # 实验名称
  default_hdfs_dir: null                # HDFS存储路径
  n_gpus_per_node: 8                    # 每节点GPU数
  nnodes: 1                             # 节点数
  total_epochs: 5                       # 总训练轮数
  logger: [wandb]                       # 日志记录器
```

---

## 使用流程

### 1. 环境准备

#### 使用Docker（推荐）
```bash
# 拉取预构建镜像
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0

# 运行容器
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

#### 使用Apptainer（HPC环境）
```bash
# 转换镜像
apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0

# 启动容器
apptainer shell --nv --cleanenv --bind /mnt/your_dir:/mnt/your_dir easyr1.sif
```

#### 本地安装
```bash
git clone https://github.com/hiyouga/EasyR1.git
cd EasyR1
pip install -e .
```

### 2. 准备数据

数据格式示例（JSON Lines）：
```json
{"problem": "What is 2+2?", "answer": "4", "images": null}
{"problem": "Solve x^2=16", "answer": "x=4 or x=-4", "images": null}
```

对于视觉语言模型：
```json
{
  "problem": "What shape is shown in the image?",
  "answer": "Triangle",
  "images": ["path/to/image.jpg"]
}
```

### 3. 配置训练

编辑配置文件或通过命令行参数覆盖：
```bash
# 方法1: 修改 examples/config.yaml
vim examples/config.yaml

# 方法2: 命令行覆盖
python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files=your_dataset \
  worker.actor.model.model_path=/path/to/model \
  trainer.n_gpus_per_node=2
```

### 4. 启动训练

#### 单机多卡
```bash
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

#### 自定义训练
```bash
export MODEL_PATH=/path/to/your/model

python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files=hiyouga/geometry3k@train \
  data.val_files=hiyouga/geometry3k@test \
  worker.actor.model.model_path=$MODEL_PATH \
  trainer.experiment_name=my_experiment \
  trainer.n_gpus_per_node=4
```

### 5. 监控训练

训练日志会自动记录到配置的日志系统：
- **Wandb**: https://wandb.ai/
- **SwanLab**: 本地可视化
- **Tensorboard**: `tensorboard --logdir runs/`

### 6. 合并检查点

训练完成后，将FSDP分片模型合并为HuggingFace格式：
```bash
python3 scripts/model_merger.py \
  --local_dir checkpoints/easyr1/experiment_name/global_step_1000/actor
```

合并后的模型可以直接用于推理：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("checkpoints/.../merged")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/.../merged")
```

---

## 多卡部署说明

### 问题背景
早期版本在多GPU部署时存在bug：所有进程都默认使用GPU 0，导致：
- 内存溢出 (OOM)
- NCCL通信错误
- GPU利用率不均衡

### 修复方案
在 `verl/single_controller/base/worker.py` 中添加了NVIDIA GPU的设备绑定：

```python
# 获取local rank
local_rank = int(os.getenv("LOCAL_RANK", os.getenv("RAY_LOCAL_RANK", "0")))

if "AMD" in torch.cuda.get_device_name():
    # AMD GPU逻辑
    ...
else:
    # NVIDIA GPU: 根据rank设置设备
    torch.cuda.set_device(local_rank)
```

### 验证GPU设置

#### 1. 快速检查
```bash
bash check_gpu_setup.sh
```

#### 2. 详细测试
```bash
bash test_gpu_mapping.sh
```

预期输出：
```
=== Environment Variables ===
RANK: 0 (进程1), 1 (进程2)
LOCAL_RANK: 0 (进程1), 1 (进程2)
...
=== CUDA Device Info ===
Current Device: 0 (进程1), 1 (进程2)
```

### 多卡训练配置

```bash
# 2卡训练
trainer.n_gpus_per_node=2

# 4卡训练
trainer.n_gpus_per_node=4

# 多节点训练 (2节点，每节点8卡)
trainer.nnodes=2
trainer.n_gpus_per_node=8
```

### 内存优化建议

如果遇到OOM错误，可以尝试：
1. 减小batch size:
   ```yaml
   data.rollout_batch_size: 256
   worker.actor.global_batch_size: 64
   ```

2. 启用gradient checkpointing:
   ```yaml
   worker.actor.model.enable_gradient_checkpointing: true
   ```

3. 使用BF16训练:
   ```yaml
   worker.actor.fsdp.torch_dtype: bf16
   worker.actor.optim.strategy: adamw_bf16
   ```

4. 减小序列长度:
   ```yaml
   data.max_prompt_length: 1024
   data.max_response_length: 1024
   ```

---

## 常见问题 (FAQ)

### Q1: 如何添加自定义数据集？
**A**: 准备JSON Lines格式文件，包含`prompt_key`、`answer_key`等字段，然后在配置中指定：
```yaml
data:
  train_files: path/to/your/train.jsonl
  prompt_key: question
  answer_key: solution
```

### Q2: 如何自定义奖励函数？
**A**: 在`examples/reward_function/`下创建Python文件，实现`compute_reward`函数：
```python
def compute_reward(prompts, responses, **kwargs):
    # 你的奖励逻辑
    rewards = ...
    return rewards
```

然后在配置中指定：
```yaml
worker.reward.reward_fn_path: examples/reward_function/your_reward.py
```

### Q3: 训练中断如何恢复？
**A**: 框架支持自动从检查点恢复，在配置中设置：
```yaml
trainer.load_checkpoint: true
trainer.checkpoint_path: checkpoints/easyr1/exp_name/global_step_1000
```

### Q4: 如何减少VRAM使用？
**A**: 参考[多卡部署说明](#内存优化建议)中的优化建议。

### Q5: 支持LoRA训练吗？
**A**: 当前版本不支持LoRA，团队正在开发中。

---

## 技术栈

- **深度学习框架**: PyTorch 2.8.0+
- **分布式框架**: Ray, FSDP
- **推理引擎**: vLLM 0.8.3+
- **模型库**: Transformers 4.54.0+
- **配置管理**: OmegaConf
- **加速库**: Flash-Attention 2.4.3+

---

## 相关文档

- [README.md](README.md) - 项目快速入门
- [MULTI_GPU_DEPLOYMENT_SUMMARY.md](MULTI_GPU_DEPLOYMENT_SUMMARY.md) - 多GPU部署详细说明
- [QUICK_FIX_GUIDE.md](QUICK_FIX_GUIDE.md) - 快速问题修复指南
- [HybridEngine论文](https://arxiv.org/abs/2409.19256) - 架构设计原理

---

## 贡献与支持

- **GitHub**: https://github.com/hiyouga/EasyR1
- **Issues**: 报告bug或提出功能请求
- **Twitter**: [@llamafactory_ai](https://twitter.com/llamafactory_ai)
- **原始项目**: [veRL by VolcEngine](https://github.com/volcengine/verl)

---

## 许可证

本项目采用 Apache License 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

---

**文档版本**: v1.0  
**最后更新**: 2026-01-21  
**维护者**: EasyR1 开发团队
