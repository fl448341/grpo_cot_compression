#!/bin/bash
# one grpo training on rtx 4090
# Param via ENV:  LENGTH_LAMBDA, TOTAL_STEPS ...

set -e
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HOME=$HOME/.cache/huggingface
export LENGTH_LAMBDA=${LENGTH_LAMBDA:-0.0}
export LENGTH_VARIANT=${LENGTH_VARIANT:-plain}
export LENGTH_TARGET=${LENGTH_TARGET:-0.0}

cd ~/rl/TinyZero
DATA_DIR=$HOME/data/gsm8k
EXP=grpo_lam${LENGTH_LAMBDA}_${LENGTH_VARIANT}_kl${KL_COEF:-0.001}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=200 \
    data.max_prompt_length=256 \
    data.max_response_length=${MAXRESP:-512} \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=${LR:-1e-5} \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF:-0.001} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.temperature=${TEMP:-0.7} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPUMEM:-0.2} \
    actor_rollout_ref.rollout.n=${GRPO_N:-8} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-50} \
    trainer.test_freq=${TEST_FREQ:-25} \
    trainer.total_training_steps=${TOTAL_STEPS:-150} \
    trainer.project_name=rl_proj7 \
    trainer.experiment_name=$EXP \
    trainer.default_local_dir=$HOME/rl/phase2/ckpts/$EXP \
    trainer.default_hdfs_dir=null \
    +trainer.val_before_train=False
