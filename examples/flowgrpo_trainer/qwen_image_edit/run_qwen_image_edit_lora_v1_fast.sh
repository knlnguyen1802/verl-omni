#!/usr/bin/env bash
# Qwen-Image-Edit-2511 LoRA RL with PickScore reward (V1 trainer, sync mode) —
# throughput-optimized variant of run_qwen_image_edit_lora_v1.sh.
#
# Baseline profile (wandb flow_grpo/l1qebcan, 8xA100-80GB, 32 prompts x n=16
# = 512 images/step): step 5223s = update_actor 2280s (44%) + gen 2078s (40%)
# + old_log_prob 633s (12%) + update_weights 96s + feed 35s.
#
# This launcher defaults to a sub-1000s profile. Per-GPU accounting of the
# baseline says where the time must come from: rollout forwards ran ~1.35s per
# single-image forward (batch-1, latency-bound; the engine never got
# max_num_seqs), and update_actor is eager-diffusers at ~25% MFU, so config
# fixes alone only halve it (~1100s at n=16). The batch shape therefore also
# shrinks, matching the OCR v1 recipe (n=8, SDE window 2):
#   n 16->8            halves both gen and update_actor
#   sde_window 3->2    another -33% off update_actor (per-timestep training)
# Budget at 8xA100: gen ~200-350s + update_actor ~400-550s + weights/feed ~200s
# = ~800-1000s/step. Escape hatches: ROLLOUT_N=16 SDE_WINDOW_SIZE=3 restores
# the baseline algorithm shape (~1700-2200s/step with the other fixes kept).
#
# What this launcher changes and why:
#
# 1. gen — request-level batching. The baseline never sets max_num_seqs, so the
#    diffusion engine falls back to near-serial generation. docs/start/
#    rollout_batching.md measures ~2x from max_num_seqs=32 (the documented safe
#    bound at true_cfg_scale=4 / 512px) on the same 32x16 workload. The mm
#    processor cache (0GB -> 2GB) reuses each condition image's VAE encode
#    across the 16 rollouts of a prompt instead of re-encoding every request.
# 2. old_log_prob — eliminated. calculate_log_probs=true makes the rollout
#    engine record per-sample log probs inside the SDE window at (near) zero
#    cost, and rollout_correction.bypass_mode=true substitutes them for
#    old_log_probs, skipping the 6-forward-per-image trainer recompute
#    (trainer_base.py "Bypass mode: skip old_log_prob recompute").
# 3. update_actor — FSDP param/optimizer offload OFF. With LoRA the resident
#    training set is a ~2.5GB/rank shard plus fp32 adapters, and the rollout
#    engines sleep through the training half of the step, so offload only adds
#    PCIe round-trips (docs/perf/diffusion_mfu.md: param_offload is the
#    largest MFU loss). Micro-batch stays at the n=8 mini-batch size (16/GPU);
#    raise to 32 if you go back to ROLLOUT_N=16. Gradient checkpointing stays
#    on: activations at micro 16-32 need it on 80GB.
#
# Algorithmic knobs default to the sub-1000s shape (n=8, window 2) but are
# env-overridable; the baseline shape is ROLLOUT_N=16 SDE_WINDOW_SIZE=3.
#
# Fallbacks: OFFLOAD=true restores the baseline memory profile; GRAD_CKPT=false
# plus PPO_MICRO_BATCH_PER_GPU=8 trades checkpoint recompute (~33% fwd cost)
# for recompute-free small batches if you have headroom to experiment.
# SKIP_OLD_LOGPROB=false restores the exact old_log_prob recompute.
set -x

# Enable reward model on GPU: Ray num_gpus=0 actors can still see CUDA devices.
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Offload PickScore to the host between scoring batches; scoring streams with
# rollout generation, so the rollout engine stays resident and each worker only
# holds the scorer on GPU while a batch is in flight.
export PICKSCORE_OFFLOAD=${PICKSCORE_OFFLOAD:-true}

model_name=${MODEL_PATH:-Qwen/Qwen-Image-Edit-2511}
reward_function_path=${REWARD_FUNCTION_PATH:-pkg://verl_omni.utils.reward_score.pickscore_reward}

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-8}
ACTOR_SP=${ACTOR_SP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-1}
REWARD_WORKERS=${REWARD_WORKERS:-4}
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen_image_edit_lora_pickscore_v1_fast}

# Perf profile (see header).
OFFLOAD=${OFFLOAD:-false}
GRAD_CKPT=${GRAD_CKPT:-true}
SKIP_OLD_LOGPROB=${SKIP_OLD_LOGPROB:-true}
# Batch shape: n=8 x window=2 targets <1000s/step; baseline is 16 / 3.
ROLLOUT_N=${ROLLOUT_N:-8}
SDE_WINDOW_SIZE=${SDE_WINDOW_SIZE:-2}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-12}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
REQUEST_BATCH_MAX_WAIT_MS=${REQUEST_BATCH_MAX_WAIT_MS:-10}
MM_PROCESSOR_CACHE_GB=${MM_PROCESSOR_CACHE_GB:-2}

# Micro-batch defaults assume 8 GPUs; fewer GPUs double the per-rank FSDP shard
# while activations stay per-rank, which OOMs the training forward. Halve them
# below 8 GPUs (gradient accumulation keeps the effective batch unchanged).
# At ROLLOUT_N=8 the n-scaled mini-batch is 16 images/GPU, so micro 16 is the
# effective cap; set 32 when running ROLLOUT_N=16.
if [ "$NUM_GPUS_ACTOR_ROLLOUT_REWARD" -ge 8 ]; then
    ppo_micro_default=16
    logprob_micro_default=32
else
    ppo_micro_default=8
    logprob_micro_default=16
fi
PPO_MICRO_BATCH_PER_GPU=${PPO_MICRO_BATCH_PER_GPU:-$ppo_micro_default}
LOGPROB_MICRO_BATCH_PER_GPU=${LOGPROB_MICRO_BATCH_PER_GPU:-$logprob_micro_default}

ENGINE=vllm_omni

WORKSPACE=${WORKSPACE:-$(cd "$(dirname "$0")/../../.." && pwd)}
train_path=${TRAIN_FILES:-$WORKSPACE/data/qwen_image_edit/train.parquet}
test_path=${VAL_FILES:-$WORKSPACE/data/qwen_image_edit/test.parquet}

output_dir=$WORKSPACE/outputs/qwen_image_edit_lora
checkpoint_dir=$output_dir/checkpoints
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log
rollout_data_dir=$output_dir/logs/$run_timestamp/rollout_images
val_data_dir=$output_dir/logs/$run_timestamp/val_images
# Per-step rollout/validation image dumps are opt-in: serializing each step's
# images to JSONL adds CPU and disk load beside the rollout workers.
dump_args=()
if [ "${DUMP_GENERATIONS:-false}" = "true" ]; then
    dump_args+=("+trainer.rollout_data_dir=$rollout_data_dir" "+trainer.validation_data_dir=$val_data_dir")
fi
mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1
echo "Logging to $log_file"

logprob_args=()
if [ "$SKIP_OLD_LOGPROB" = "true" ]; then
    logprob_args+=(
        actor_rollout_ref.rollout.calculate_log_probs=true
        algorithm.rollout_correction.bypass_mode=true
    )
fi

python3 -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=32 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.seed=42 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.attn_backend=_flash_3_varlen_hub \
    algorithm.global_std=false \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out','img_mlp.net.0.proj','img_mlp.net.2','txt_mlp.net.0.proj','txt_mlp.net.2']" \
    actor_rollout_ref.model.enable_gradient_checkpointing=$GRAD_CKPT \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=$ACTOR_SP \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=0.0001 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BATCH_PER_GPU \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enable_prompt_embed_cache=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.mm_processor_cache_gb=$MM_PROCESSOR_CACHE_GB \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=$MAX_NUM_SEQS \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.request_batch_max_wait_ms=$REQUEST_BATCH_MAX_WAIT_MS \
    actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=$NUM_INFERENCE_STEPS \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.algo.noise_level=0.7 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=$SDE_WINDOW_SIZE \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,6]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=40 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BATCH_PER_GPU \
    reward.num_workers=$REWARD_WORKERS \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_pickscore \
    "${logprob_args[@]}" \
    trainer.logger='["console", "tensorboard", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$checkpoint_dir \
    "${dump_args[@]}" \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=20 \
    trainer.total_training_steps=300 \
    trainer.total_epochs=100 \
    trainer.resume_mode=auto \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync "$@"
