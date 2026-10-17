#!/usr/bin/env bash
# Qwen-Image LoRA OCR recipe (V1 trainer, sync mode + async reward on a dedicated GPU).
#
# This is the v1 counterpart of run_qwen_image_ocr_lora_async_reward.sh. It uses
# `verl_omni.trainer.main_diffusion_v1` with `trainer.v1.trainer_mode=sync` and
# moves reward scoring onto a dedicated GPU pool
# (`reward.reward_model.enable_resource_pool=true`) so it overlaps rollout
# generation instead of time-sharing the actor/rollout GPUs. Model, LoRA,
# pipeline, and SDE knobs match run_qwen_image_ocr_lora_v1.sh; the differences
# are the reward layout and a higher rollout concurrency.
#
# Layout: 4x 80GB GPUs for colocated actor/rollout + 1 dedicated reward GPU
# (5 GPUs total). On an 8-GPU node this still leaves 3 GPUs spare, which can be
# used to scale rollout further (see separate_async recipes).
#
# Reference:
#   v1 sync baseline:  run_qwen_image_ocr_lora_v1.sh
#   v0 async-reward:    run_qwen_image_ocr_lora_async_reward.sh
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

ocr_train_path=$WORKSPACE/data/ocr/qwen_image/train.parquet
ocr_test_path=$WORKSPACE/data/ocr/qwen_image/test.parquet

model_name=Qwen/Qwen-Image
reward_model_name=Qwen/Qwen3-VL-8B-Instruct
reward_function_path=verl_omni/utils/reward_score/genrm_ocr.py

# Actor + rollout share a colocated pool; reward runs on its own dedicated GPU.
NUM_GPUS_ACTOR_ROLLOUT=4
NUM_GPUS_REWARD=1
ROLLOUT_TP=1
REWARD_TP=1

ENGINE=vllm_omni
REWARD_ENGINE=vllm
# Request-level packing (mutually exclusive with step-wise continuous batching).
# Raised from the sync baseline's 8 to 32: the request-level safe bound at
# true_cfg_scale=4 (see docs/start/rollout_batching.md) is <=32, and the
# dedicated reward GPU frees the colocated pool from reward memory pressure.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
REQUEST_BATCH_MAX_WAIT_MS=${REQUEST_BATCH_MAX_WAIT_MS:-10}

# Micro batches match the v1 sync recipe.
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-16}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-32}

# Reproducibility knobs (see run_qwen_image_ocr_lora_v1.sh for details):
#   actor_rollout_ref.rollout.seed defaults to 42 (deterministic per prompt).
#   data.seed defaults to null / unseeded:
#     data.seed=42

python3 -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_batch_size=32 \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out','img_mlp.net.0.proj','img_mlp.net.2','txt_mlp.net.0.proj','txt_mlp.net.2']" \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=${MAX_NUM_SEQS} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.request_batch_max_wait_ms=${REQUEST_BATCH_MAX_WAIT_MS} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
    reward.num_workers=$((NUM_GPUS_REWARD / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.enable_resource_pool=True \
    reward.reward_model.nnodes=1 \
    reward.reward_model.n_gpus_per_node=$NUM_GPUS_REWARD \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.reward_model.rollout.gpu_memory_utilization=0.9 \
    reward.reward_model.rollout.free_cache_engine=False \
    reward.reward_model.rollout.enforce_eager=False \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=qwen_image_ocr_lora_v1_async_reward \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync "$@"
