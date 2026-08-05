#!/usr/bin/env bash
# SD3.5-Medium single-teacher On-Policy Distillation (SOPD) recipe.
#
# Reproduces the single-teacher OPD experiment from "DiffusionOPD" (arXiv:2605.15055)
# inside verl-omni: a frozen task-specific teacher LoRA (here, the Aesthetics teacher
# `quanhaol/Aes-Teacher`) is loaded onto the actor's own SD3.5-M backbone as a second
# PEFT adapter and queried at the student's rollout states. The student is trained
# purely against the teacher's per-step transition mean via the closed-form KL loss
# (`distill_kl`); no reward model / PPO / advantage is used (OPD-only mode).
#
# Default sampler is the deterministic ODE (noise_level=0), under which the per-step
# closed-form reverse-KL specializes to the squared-L2 surrogate 0.5*||mu_s - mu_t||^2
# (paper Eq. 12) — this is the SD3.5-M OPD default in the reference implementation.
#
# Prerequisites:
#   * Pre-download the Aes teacher LoRA: `huggingface-cli download quanhaol/Aes-Teacher`
#     and pass its local path via TEACHER_ADAPTER_PATH (or let HF hub resolve it).
#   * A parquet dataset of text prompts (e.g. Pick-a-Pic train split, which the Aes
#     teacher was trained on). Set PICKSCORE_TRAIN_PATH / PICKSCORE_VAL_PATH.
#
# Reference scripts:
#   - DiffusionOPD/scripts/single_node/sopd.sh           (reference implementation)
#   - run_sd35_medium_ocr_lora.sh                         (FlowGRPO SD3.5 base)
set -x

WORKSPACE=${OPD_WORKSPACE:-${WORKSPACE:-$HOME}}
pickscore_train_path=${PICKSCORE_TRAIN_PATH:-$WORKSPACE/data/pickscore/sd3/train.parquet}
pickscore_val_path=${PICKSCORE_VAL_PATH:-$WORKSPACE/data/pickscore/sd3/test.parquet}

model_name=stabilityai/stable-diffusion-3.5-medium
teacher_adapter_path=${TEACHER_ADAPTER_PATH:-quanhaol/Aes-Teacher}

NUM_GPUS_ACTOR_ROLLOUT=2
ROLLOUT_TP=1
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
ATTN_BACKEND=${ATTN_BACKEND:-native}

MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
REQUEST_BATCH_MAX_WAIT_MS=${REQUEST_BATCH_MAX_WAIT_MS:-10}
ROLLOUT_ATTN_BACKEND=${ROLLOUT_ATTN_BACKEND:-TORCH_SDPA}

if [ "${FA3:-0}" = "1" ]; then
    ATTN_BACKEND="_flash_3_varlen_hub"
    ROLLOUT_ATTN_BACKEND=FLASH_ATTN_3_HUB
fi

ENGINE=vllm_omni

# SD3 uses a joint CLIP-L/G + T5-XXL prompt encoding; the extra tokenizers mirror the
# FlowGRPO SD3.5 recipe so rollout and training-side prompt embeds stay consistent.
custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$pickscore_train_path \
    data.val_files=$pickscore_val_path \
    data.train_batch_size=8 \
    data.val_max_samples=32 \
    data.max_prompt_length=512 \
    data.truncation=error \
    data.seed=42 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.custom_chat_template="\"$custom_chat_template\"" \
    'actor_rollout_ref.model.extra_tokenizers={clip: {path: tokenizer, max_length: 77}, t5: {path: tokenizer_3, max_length: 256}}' \
    actor_rollout_ref.model.attn_backend=$ATTN_BACKEND \
    actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT_ATTN_BACKEND \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']" \
    actor_rollout_ref.model.teacher_adapter_path=$teacher_adapter_path \
    actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT_ATTN_BACKEND \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.use_distill_loss=True \
    actor_rollout_ref.actor.distill_loss_mode=distill_kl \
    actor_rollout_ref.actor.distill_loss_coef=1.0 \
    actor_rollout_ref.actor.opd_only=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=4.5 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.algo.sde_type="cps" \
    actor_rollout_ref.rollout.algo.sde_window_size=3 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=${MAX_NUM_SEQS} \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.request_batch_max_wait_ms=${REQUEST_BATCH_MAX_WAIT_MS} \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=40 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    reward.reward_model.enable=False \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=opd \
    trainer.experiment_name=sd35_medium_sopd_aes_lora \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    "$@"