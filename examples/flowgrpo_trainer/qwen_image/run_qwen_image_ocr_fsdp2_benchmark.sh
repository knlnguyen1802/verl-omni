#!/bin/bash
# Qwen-Image full-weight RL throughput benchmark for 8 x 80 GB GPUs
# (V1 trainer: TransferQueue + ReplayBuffer + sync mode).
#
# Uses the `verl_omni.trainer.main_diffusion_v1` entrypoint, which selects
# `PolicyGradientDiffusionTrainerV1Sync` via `trainer.v1.trainer_mode=sync`.
# TransferQueue is force-enabled inside the runner, so it does not need to be
# set on the CLI. Training, rollout, and reward knobs match the v0 benchmark
# this replaces; workload parity (512px, true_cfg_scale=1.0, SDE window) is
# kept so v0 vs v1 throughput is comparable.
#
# Reference (legacy v0 benchmark, FSDP1-era override wrapper):
# verl-omni/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr.sh
# Reference (v1 twin this recipe's v1 wiring follows):
# verl-omni/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_v1.sh
#
# FSDP2 is required because FSDP1 runs out of memory on this configuration.
# Rollout step execution and reward-model CUDA graphs/batching are enabled to
# improve throughput. FSDP2 forward prefetch is also enabled. The benchmark
# enables regional torch.compile while retaining Hub FA3 for actor training
# and rollout. Graph breaks are allowed so third-party attention preprocessing
# can stay eager without requiring coordinated compiler, Diffusers, and FA3
# patches. Append actor_rollout_ref.model.use_regional_compile=False to
# compare against eager execution. This benchmark disables checkpoint saving
# and periodic validation.
#
# Keep recompiles shared because regional compilation invokes torch.compile
# once per repeated transformer block. The 60 structurally identical Qwen Image
# blocks can then reuse compiled entries instead of compiling every graph-break
# region independently for every block.
# Compile dynamic shapes up front because Qwen Image derives its rotary-embedding
# length from each batch's text mask. Static compilation specializes every
# repeated block for each prompt length and exhausts Dynamo's accumulated
# recompile limit before the first training step completes.
set -euo pipefail
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

ocr_train_path=$WORKSPACE/data/ocr/qwen_image/train.parquet
ocr_test_path=$WORKSPACE/data/ocr/qwen_image/test.parquet

model_name=Qwen/Qwen-Image
reward_model_name=Qwen/Qwen3-VL-8B-Instruct
reward_function_path=verl_omni/utils/reward_score/genrm_ocr.py

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS:-8}
NUM_NODES=${NUM_NODES:-1}
ROLLOUT_TP=1
REWARD_TP=1
IMAGE_RESOLUTION=512

ENGINE=vllm_omni
REWARD_ENGINE=vllm
# Keep step-wise rollout and old-log-prob recomputation batch shapes aligned.
# Qwen-Image BF16 kernels are batch-shape sensitive; allowing rollout batches
# larger than recomputation batches can increase rollout-policy disagreement.
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-32}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-${LOG_PROB_MICRO_BATCH_SIZE}}
if [[ "${MAX_NUM_SEQS}" != "${LOG_PROB_MICRO_BATCH_SIZE}" ]]; then
    echo "WARNING: MAX_NUM_SEQS=${MAX_NUM_SEQS} differs from LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE}; this may hurt FlowGRPO convergence." >&2
fi

export TORCH_LOGS="${TORCH_LOGS:-graph_breaks,recompiles}"
echo "Using TORCH_LOGS=$TORCH_LOGS for torch.compile diagnostics."

python3 -m verl_omni.trainer.main_diffusion_v1 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_batch_size=32 \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.model.use_regional_compile=True \
    'actor_rollout_ref.model.regional_compile_options={backend:inductor,mode:default,fullgraph:false,dynamic:true}' \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.step_execution=true \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=${MAX_NUM_SEQS} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE} \
    reward.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.reward_model.rollout.enforce_eager=False \
    reward.reward_model.rollout.max_num_seqs=128 \
    reward.reward_model.rollout.max_model_len=8192 \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=qwen_image_ocr_8x80g_fsdp2_v1_benchmark \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / NUM_NODES)) \
    trainer.nnodes=$NUM_NODES \
    trainer.resume_mode=disable \
    trainer.save_freq=0 \
    trainer.test_freq=0 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync "$@"
