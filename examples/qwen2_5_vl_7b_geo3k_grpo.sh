#!/bin/bash

set -x

# MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
MODEL_PATH=/hpc2hdd/home/ckwong627/workdir/models/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.rollout_batch_size=1 \
    data.mini_rollout_batch_size=1 \
    data.val_batch_size=1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo \
    trainer.n_gpus_per_node=8
