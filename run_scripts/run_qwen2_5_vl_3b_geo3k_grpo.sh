#!/bin/bash
#SBATCH -p debug        # 指定GPU队列
#SBATCH -o ./temp/output.txt  # 指定作业标准输出文件，%j为作业号  SBATCH -o output_%j.txt
#SBATCH -e ./temp/err.txt    # 指定作业标准错误输出文件  SBATCH -e err_%j.txt
#SBATCH -n 8            # 指定CPU总核心数
#SBATCH --gres=gpu:2    # 指定GPU卡数
#SBATCH -D .        # 指定作业执行路径为当前目录

# 加载CUDA模块（如果需要）
module load cuda/11.8

# 激活 Conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate easyr1

# # 设置 LD_LIBRARY_PATH，只使用 conda 下的 .so 文件
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# echo "[INFO] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# # 可选调试：查看 .so 是否在位
# echo "[DEBUG] Verifying libcudnn_graph.so presence:"
# ls -l $CONDA_PREFIX/lib/libcudnn_graph.so*

# Job 执行主体
echo "Job started at $(date)"
# export CUDA_VISIBLE_DEVICES=0,1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash examples/qwen2_5_vl_3b_geo3k_grpo.sh
echo "Job ended at $(date)"

# 退出环境
conda deactivate