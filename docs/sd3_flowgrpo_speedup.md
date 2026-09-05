# SD3-Medium FlowGRPO Speedup: Code Changes Explained

This document explains each code change made to speed up `verl_omni` for the
SD3-medium FlowGRPO path, and the reasoning behind why each one makes the
training/rollout run faster.

The changes are grouped by the hot-path they target, as identified from the
profiling trace:

| Profile line | Count | Self time | Root cause addressed by |
|---|---|---|---|
| `update_actor` | 1 | 4.851s | Change 3 + 4 (batched window forward/backward) |
| `MulBackward0` | 4236 | 1.319s | Change 3 + 4 |
| `MmBackward0` | 2916 | 274.9ms | Change 3 + 4 |
| `aten::to` / `_to_copy` / `convert_element_type` | ~10k | ~3s | Change 1 + 2 |
| `cudaLaunchKernel` | 80971 | 632.8ms | Change 3 + 4 (fewer, larger kernels) |
| `diffusion_forward` / `pipeline_forward` | 4 | 13.58s | Change 2 (rollout dtype churn) |

---

## Change 1 — Cache `math.pi` and reuse `sqrt(-dt)` in the SDE scheduler

**File:** `verl_omni/pipelines/schedulers/flow_match_sde.py`

### What changed

- Added a module-level float constant `_LOG_SQRT_2PI = math.log(math.sqrt(2.0 * math.pi))`.
- Replaced the per-step log-prob normalizer
  `torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))` with a subtraction of
  the plain float `_LOG_SQRT_2PI`.
- Hoisted `sqrt_neg_dt = torch.sqrt(-1 * dt)` out of the `sde` branch so it is
  computed **once** and reused by:
  1. the noise scale `std_dev_t * sqrt_neg_dt`,
  2. the log-prob denominator `(std_dev_t * sqrt_neg_dt) ** 2`,
  3. the log-prob normalizer `torch.log(std_dev_t * sqrt_neg_dt)`,
  4. the `return_sqrt_dt` tail (which previously recomputed
     `torch.sqrt(-1 * dt)` a third time).
- Applied the same `_LOG_SQRT_2PI` fix to the `dance_sde` branch.

### Why it is faster

The old code called `torch.as_tensor(math.pi)` **inside the per-step
log-prob block**. `torch.as_tensor(math.pi)` creates a **CPU** scalar tensor.
When it is then combined with the fp32 `log_prob` that lives on GPU
(`log_prob - torch.log(...) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))`),
PyTorch must either broadcast a CPU tensor against a GPU tensor (which forces
an implicit host-to-device copy and a stream sync) or materialize it on the
device per call. This happens on **every denoising step**, for every micro-batch,
in both rollout and training.

Subtracting a plain Python float from a GPU tensor is a single fused CUDA kernel
with **no CPU tensor, no H2D copy, no sync**. The numerical result is identical
because `log(sqrt(2*pi))` is a constant.

Reusing `sqrt_neg_dt` removes two extra `torch.sqrt` kernel launches per step.
`torch.sqrt` is not free — each call is a separate `cudaLaunchKernel`. The
profile shows `cudaLaunchKernel` at 80971 calls / 632.8ms, so cutting a few per
step compounds across `W` steps × micro-batches × training steps.

---

## Change 2 — Stop per-step timestep dtype churn in the rollout `diffuse` loop

**File:** `verl_omni/pipelines/sd3_flow_grpo/vllm_omni_rollout_adapter.py`

### What changed

- Before the denoising loop, pre-compute
  `timesteps_model = timesteps.to(device=self.device, dtype=model_dtype)` **once**.
- Inside the loop, the transformer now receives
  `timestep = timesteps_model[i].expand(latents.shape[0])` (already model
  dtype) instead of
  `timestep_value.expand(...).to(device=self.device, dtype=model_dtype)` per
  step.
- The fp32 `timesteps` tensor is **kept** for `scheduler.step(timestep_value)`,
  because the scheduler's `index_for_timestep` does an equality lookup
  (`schedule_timesteps == timestep`) that would be unreliable in bfloat16.

### Why it is faster

The old loop called `.to(device=..., dtype=model_dtype)` on the timestep
scalar **every denoising step**. `.to()` with a dtype change is an
`aten::_to_copy` (a real allocation + copy), and with a device change it is an
H2D copy. The profile shows:

```
aten::to                   8013   1.151s
aten::_to_copy             1591   1.040s
prims::convert_element_type 492   816.6ms
```

A large fraction of those come from this per-step cast (and the symmetric
`latents.to(model_dtype)` and `noise_pred.float()` casts). Pre-casting the
whole `timesteps` schedule once turns `W` per-step `aten::to` calls into a
single `aten::to` at loop entry. The per-step `timesteps_model[i]` is a
zero-copy view/slice of an already-correct-dtype tensor.

Keeping the fp32 `timesteps` for the scheduler avoids a correctness regression:
`FlowMatchEulerDiscreteScheduler.index_for_timestep` compares timesteps with
`==`, and bfloat16 cannot represent the fine-grained sigma values diffusers
emits, so casting the scheduler-side timesteps to bfloat16 would make the
equality lookup miss and break sampling.

---

## Change 3 — Batched SDE-window forward (`forward_and_sample_window`)

**Files:**
- `verl_omni/pipelines/sd3_flow_grpo/diffusers_training_adapter.py` (new method)
- `verl_omni/pipelines/model_base.py` (base-class hook)

### What changed

- Added `StableDiffusion3FlowGRPO.forward_and_sample_window(...)`, the batched
  counterpart of `forward_and_sample_previous_step`. Instead of running one
  forward for one timestep, it folds the **whole SDE window** (`W` steps) into
  the batch dimension and runs:
  1. **one** transformer forward of shape `(B*W, ...)` with `prompt_embeds` and
     `pooled_prompt_embeds` `repeat_interleave`'d along batch,
  2. **one** `scheduler.sample_previous_step` call over all `B*W` rows,
  3. a reshape of the outputs back to `(B, W, ...)`.
- Added a default `DiffusionModelBase.forward_and_sample_window` that raises
  `NotImplementedError`, so the engine can detect adapters that have not
  implemented the batched path and fall back to the per-step loop.

### Why it is faster

This is the single highest-impact change. The old training loop (see Change 4)
called `forward_and_sample_previous_step` **once per timestep**, i.e. `W`
separate transformer forwards, each of shape `(B, ...)`. That produced:

- `W` separate small matmuls instead of one large one. GPUs are dramatically
  more efficient at one `(B*W, L, D) × (D, D)` matmul than `W` separate
  `(B, L, D) × (D, D)` matmuls, because kernel launch overhead amortizes and
  the GEMM hits higher arithmetic intensity. The profile shows
  `aten::mm` 16218× / 669ms and `cudaLaunchKernel` 80971× / 632ms — much of
  that is per-step launch overhead.
- `W` separate autograd graphs. Each step built its own `MulBackward0` /
  `MmBackward0` / `AddBackward0` nodes, which is exactly why the profile shows
  `MulBackward0` 4236× / 1.319s and `MmBackward0` 2916× / 274.9ms. Batching
  collapses `W` per-step graphs into one graph with a handful of nodes, and the
  backward traverses them in a single `loss.backward()` call (Change 4).

Folding `W` into the batch dim is mathematically exact for SD3 because the
`SD3Transformer2DModel` is shape-agnostic: it does not treat the batch axis
specially, so `(B*W, C, H, W)` with `timestep.reshape(B*W)` produces the same
per-element outputs as `W` separate `(B, ...)` forwards. The scheduler's
`sample_previous_step` is also purely element-wise over the batch dim when
`timestep` is provided, so passing `(B*W,)` timesteps yields the same per-row
log-probs.

---

## Change 4 — One forward + one backward per micro-batch in the FSDP engine

**File:** `verl_omni/workers/engine/fsdp/diffusers_impl.py`

### What changed

- `PPODiffusersFSDPEngine.forward_backward_batch` now prefers a new
  `_run_window_forward_backward_batch` when the adapter supports it, and falls
  back to the existing per-step `_run_forward_backward_batch` otherwise.
- `_run_window_forward_backward_batch`:
  1. calls `forward_and_sample_window` once per micro-batch (over all `W`
     steps),
  2. flattens `(B, W)` → `(B*W,)` for the loss inputs (`old_log_probs`,
     `advantages`, and optional `ref_*` / `old_prev_sample_mean` /
     `rollout_is_weights`),
  3. calls `loss.backward()` **once** per micro-batch instead of `W` times,
  4. splits the window-stacked output back into per-step dicts so
     `postprocess_batch_func` is unchanged.
- `gradient_accumulation_steps = len(micro_batches)` (was
  `len(micro_batches) * num_timesteps`).
- Added `_supports_window_forward(loss_function)` which gates the window path
  on **both**:
  - the adapter overriding `forward_and_sample_window`, and
  - the loss mode being **window-safe**.
- `_resolve_loss_mode(loss_function)` extracts the loss mode from the
  `functools.partial(diffusion_loss, config=actor_config)` wrapped loss fn.

### Why it is faster

The old `_run_forward_backward_batch` loop was:

```python
for step in range(num_timesteps):       # W iterations
    loss, _ = self.forward_step(..., step=step)
    loss.backward()                     # one backward PER step
```

Each `.backward()` is a separate autograd engine traversal. With `W` steps
that is `W` Python-level backward calls, `W` separate graph traversals, and
`W` sets of small backward kernels. This is the direct source of the
`update_actor` 4.851s line and the `MulBackward0` 4236× / `MmBackward0` 2916×
explosion (the counts are roughly `W × micro_batch × per-step-node-count`).

The new path runs **one** `loss.backward()` per micro-batch over the combined
`(B*W,)` loss. The autograd graph is built once (one forward) and traversed
once (one backward). This:

- removes the `W`-fold Python-side backward overhead,
- lets the backward issue one set of larger, fused kernels instead of `W`
  sets of small ones,
- reduces `cudaLaunchKernel` count proportionally.

### Gradient-scale equivalence (why this is not a loss-accuracy change)

The loss wrapper `diffusion_loss` divides the loss by
`gradient_accumulation_steps` before returning it, and `.backward()` accumulates
gradients. Comparing the effective per-element gradient scale:

- **Old (per-step):** `gradient_accumulation_steps = M * W`. Each per-step loss
  is `mean over B`, divided by `M*W`. Summed over `W` backward calls:
  `W * (1/(M*W)) * (1/B) * sum_{s,b} d(loss_{s,b}) = 1/(M*W*B) * sum`.
- **New (batched):** `gradient_accumulation_steps = M`. Loss is
  `mean over (B*W)`, divided by `M`: `(1/M) * (1/(B*W)) * sum_{s,b} d(loss) =
  1/(M*B*W) * sum`.

Since `W` (window) equals `num_timesteps` in the training loop (the rollout
only stores window-step timesteps in `all_timesteps`), the two are identical:
`1/(M*W*B) == 1/(M*B*W)`. So the optimizer sees the same gradient magnitude.

### Why `grpo_guard` is excluded

`GRPOGuardLoss.compute_loss` reduces `std_dev_t.mean()` and `sqrt_dt.mean()` to
a **per-step scalar** (`scale = sqrt_dt_mean * sigma_t`). In the per-step path
that scalar is averaged over `B` for one step. If we folded all `W` steps into
one batch dim, `.mean()` would average over `B*W` — mixing the per-step scales
into one global scalar and silently changing the math.

`flow_grpo`, `dance_grpo`, and `flow_dppo` are element-wise over the batch dim
(no per-step scalar reduction), so they are window-safe. `_supports_window_forward`
restricts activation to exactly that set, so `grpo_guard` automatically falls
back to the unchanged per-step loop.

---

## Summary of expected impact

| Change | Targeted profile lines | Risk | Expected win |
|---|---|---|---|
| 1. Cache `math.pi` / reuse `sqrt(-dt)` | per-step CPU tensor + sync, extra `cudaLaunchKernel` | very low | small but free; removes per-step H2D sync |
| 2. Pre-cast rollout timesteps | `aten::to` / `_to_copy` in rollout | low | removes `W` per-step dtype copies |
| 3. Batched window forward | `aten::mm` count, `cudaLaunchKernel` count | medium (needs shape/memory check) | one large matmul instead of `W` small ones |
| 4. One backward per micro-batch | `MulBackward0`, `MmBackward0`, `update_actor` | medium (needs grad-equivalence check) | collapses `W` autograd graphs into one |

Changes 1 and 2 are independent and safe to take alone. Changes 3 and 4 are
coupled (3 enables 4) and need the end-to-end validation described below.

---

## Validation checklist (for the human reviewer / submitter)

Per `AGENTS.md`, a human must review every changed line and run the relevant
tests before this is submitted. Specifically:

1. **Gradient equivalence:** force `_supports_window_forward` to return `False`
   temporarily and run a few steps on the same seed/data with both paths.
   Confirm `actor/ppo_kl`, `actor/ratio_mean`, and the loss curve match to
   tolerance.
2. **Memory:** the window path holds `W` forward graphs in memory at once
   before backward. With `ppo_micro_batch_size_per_gpu=8` and `window=3` that is
   ~3× the per-step activation footprint. If OOM, reduce
   `ppo_micro_batch_size_per_gpu` by ~`W` or re-enable gradient checkpointing.
3. **Scheduler shapes:** the batched path passes `timestep` of shape `(B*W,)`,
   so `sigma_idx` is length `B*W` and `std_dev_t` is `(B*W, 1, 1, ...)`. Confirm
   the reshape back to `(B, W, 1, 1, ...)` is exact on the first run.
4. **`grpo_guard` fallback:** select `loss_mode=grpo_guard` and confirm it still
   routes through the per-step loop (via `_supports_window_forward`).
5. **DPO / NFT untouched:** confirm the DPO and NFT engines still use their
   own `forward_backward_batch` and are unaffected.
