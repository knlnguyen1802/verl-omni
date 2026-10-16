(kernel_replacement_rfc)=
# RFC: Profiling and replacing hot-path tensor code with fused CUDA/Triton kernels in diffusion RL training

Last updated: 10/08/2026.

Status: **Draft — Phase 1 (profiling) not yet executed.** All cost figures below are
static-analysis estimates from a code audit; they must be confirmed by Phase 1 before any
kernel is written.

Working doc for the `knlnguyen1802/verl-omni` fork. Not intended for upstream docs inclusion
until Phase 2 produces numbers worth publishing.

---

## 1. Summary

A static audit of `verl_omni/` (Appendix A) found ~13 hot-path sites where the diffusion
RL step loses time to (a) CPU↔GPU synchronization inside per-(micro-batch × timestep)
loops, (b) long chains of small elementwise kernels over full latents, or (c) Python-level
per-row loops that launch hundreds of tiny kernels.

This RFC defines a two-phase, evidence-first plan:

- **Phase 1 — attribute cost.** Instrument one training step of the reference benchmark
  (`examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_fsdp2_benchmark.sh`) with
  per-site profiler markers and measure, for each candidate site: calls/step, CPU enqueue
  time, GPU kernel time, kernel-launch count, sync count, and sync-stall duration. *No
  behavior changes.*
- **Phase 2 — isolated A/B per site.** Implement exactly one site's replacement (stock-op
  vectorization where possible, else a single Triton kernel) behind an env-var gate, run
  baseline vs candidate on identical config/seed, and report the end-to-end and per-phase
  percentage delta for that site alone. Repeat per site, then a combined run.

The ordering principle: **profile first, kernelize last.** Several top candidates need no
custom kernel at all (one-line de-syncs and stock-op vectorizations); those are tested
first because they are the most update-proof and the cheapest to validate.

## 2. Why this design (constraints the plan must respect)

1. **verl-omni main drifts weekly** and the `verl/` / `vllm-omni/` pins move with it
   (`.github/verl_pin.txt`, `.github/vllm_omni_pin.txt`). Everything introduced here must
   live in `verl_omni/` and consume **plain tensors and scalars only** — no imports from
   `verl` or `vllm_omni` internals, no dependency on their Python APIs.
2. **No new dependencies.** Triton ships inside every CUDA build of PyTorch ≥2.x; Triton
   JIT kernels compile from source at first call with no build step and no ABI coupling.
   CUDA C++ / `cpp_extension` is explicitly rejected (nvcc + arch + torch-version coupling
   is exactly the "depends on updates" failure mode).
3. **NPU is a supported platform** in this repo (Ascend recipes exist under
   `examples/flowgrpo_trainer/*/`*_npu.sh`). Every replacement must keep the current eager
   path intact and default; Triton paths activate only on CUDA tensors with an eager
   fallback. NPU behavior must be bit-for-bit unchanged.
4. **The repo already has the measurement machinery.** `marked_timer` phases
   (`gen`, `old_log_prob`, `ref`, `teacher`, `adv`, `update_actor`, …) are emitted as
   `timing_s/{phase}` metrics
   (`verl_omni/trainer/diffusion/diffusion_metric_utils.py:111`); the profiler subsystem
   from verl (`global_profiler.tool = nsys | torch | torch_memory | npu`, step-scoped) is
   documented in `docs/perf/profiler.md`; and
   `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_fsdp2_benchmark_nsys.sh`
   already captures controller + actor-rank nsys reports with Python sampling, NVTX, and
   CUDA-sync backtraces (`cudabacktrace=sync:100000`) over steps 2..N. The plan reuses all
   of it and adds only site-level markers.

## 3. Goals / Non-goals

**Goals**

- G1: A per-site cost table for one steady-state training step of the reference workload,
  with each site's share of step time (GPU-busy, CPU-enqueue, sync-stall split).
- G2: An isolated, reproducible A/B percentage delta per replaced site, on the same
  benchmark, with parity gates green before any timing counts.
- G3: A decision record: which sites become upstream PRs, which stay behind a flag, which
  are dropped (with numbers).
- G4: All new code confined to `verl_omni/utils/kernels/` + thin dispatch hooks at the call
  sites, gated by env var, default-off.

**Non-goals**

- No changes to the `verl/` or `vllm-omni/` checkouts or pins; anything that belongs there
  is surfaced upstream separately.
- No model-forward/attention/MoE kernel work (owned by vllm-omni / flash-attention).
- No config-schema changes during experiments (env gates only; a proper config knob is a
  post-decision follow-up, and if added must follow the diffusion config-plumbing rule of
  touching all 4 files + docs).
- No NPU kernel coverage; no mixed-precision changes; no algorithmic changes (advantage
  normalization, clip formulas, etc. stay numerically identical).

## 4. Phase 0 — Environment lock-in

Record in the experiment log (Section 11) before any run:

- verl-omni commit hash + branch; contents of `.github/verl_pin.txt` and
  `.github/vllm_omni_pin.txt`; torch / triton versions
  (`python -c "import torch, triton; print(torch.__version__, triton.__version__)"`).
- GPU model, driver, `nvidia-smi -q | grep -i "persistence\|clocks"` policy; disable
  autoboost if available, otherwise record it as a variance source.
- One fresh baseline timing run to confirm the benchmark reproduces (steps within ±5%
  across 3 repeats) before comparing anything.

The Windows workstation in this workspace has no torch/GPU; **all runs happen on the Linux
8×80GB GPU box**. The local checkout only hosts the RFC, markers, and analysis scripts.

## 5. Phase 1 — Per-site cost attribution in one training step

### 5.1 Reference workload

```bash
NUM_GPUS=8 bash examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_fsdp2_benchmark.sh \
    trainer.total_training_steps=8 trainer.save_freq=0 trainer.test_freq=0
```

This is the repo's own tuned throughput recipe (FSDP2, regional torch.compile, rollout
step-execution, 8×80GB). 8 steps = 2 warmup + 6 measured. The same command with
`trainer.total_training_steps=4` is the minimum for the nsys variant. If 8 GPUs are not
available, `NUM_GPUS=4` with the base `run_qwen_image_ocr.sh` is the fallback; record the
deviation.

### 5.2 Site markers (the one piece of new instrumentation)

Add `verl_omni/utils/kernels/site_markers.py`:

```python
import os
import torch

_RF_SITES = set(filter(None, os.environ.get("VERL_OMNI_KERNEL_RF_MARKERS", "").split(",")))

def site_marker(site_id: str):
    """Context manager labeling a candidate site in torch-profiler traces and NVTX.

    No-op (zero cost) unless the site is listed in VERL_OMNI_KERNEL_RF_MARKERS.
    """
    if site_id not in _RF_SITES or not torch.cuda.is_available():
        import contextlib
        return contextlib.nullcontext()
    return torch.profiler.record_function(f"kernelrf:{site_id}")
```

`torch.profiler.record_function` ranges appear in chrome traces *and* are exported as NVTX
ranges, so a single wrapper serves both the torch-profiler and nsys paths. Wrap each
Appendix-A site body with `with site_marker("S1_sde_step"):` (etc.). The wrap diff is
mechanical, default-off, and can be reverted file-by-file after Phase 2.

### 5.3 Measurement layers (cheapest first)

**L1 — phase budget (already emitted, zero work).** Read `timing_s/{step,gen,feed,reward,
old_log_prob,rollout_corr,ref,wait_prev_teacher,teacher,adv,update_actor}` medians for
steps 3..8 from console/TensorBoard. Output: which phase owns the step (expect
`gen` + `update_actor` + `old_log_prob` + `teacher` to dominate).

**L2 — torch-profiler per-site attribution.** Same run with
`global_profiler.tool=torch global_profiler.steps='[3,4,5]'` (per `docs/perf/profiler.md`).
Then:

```bash
python tools/analyze_kernelrf_trace.py <trace.json>   # proposed, see Deliverables
```

The analyzer groups trace events by enclosing `kernelrf:*` CPU range and reports per site:
CPU time inside the range (enqueue cost), summed GPU kernel time of child kernels, kernel
launch count, and the duration of any `cudaStreamSynchronize` / `cudaMemcpyAsync`(D2H) /
`cudaStreamWaitEvent` API calls inside the range (sync stalls). Run this on the actor rank
only; profiler overhead is acceptable here because L2 runs are not used for wall-clock A/B.

**L3 — nsys cross-check + gap analysis.** One run of the existing

```bash
TRAINING_STEPS=6 PROFILE_RANKS="[0]" \
  bash examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_fsdp2_benchmark_nsys.sh
```

which already collects, on the actor rank: CUDA + OS runtime events, CUDA **sync backtraces**
(`cudabacktrace=sync:100000` — this directly answers "which Python line stalled on a D2H"),
Python stacks at 20 Hz, and CUDA memory usage. Use `nsys stats --report cuda_api_sum` and
`--report cuda_gpu_trace` to (a) confirm L2's sync-stall totals and (b) compute the
GPU-idle gap fraction between kernels inside each phase. The report header documents
TSC-alignment caveats; steps 3..N-1 are the steady-state intervals.

**L4 — dynamic sync count.** One 4-step run with `torch.cuda.set_sync_debug_mode("warn")`
and a warnings-capturing wrapper, attributing warning counts to sites by nesting the
markers (the sync happens inside the `with site_marker(...)` block). This yields the
"syncs/step" column and validates the static estimates in Appendix A.

### 5.4 Phase-1 output and gate

Fill the Phase-1 table in the experiment log:

| site | calls/step | CPU ms | GPU ms | launches | syncs | stall ms | % of step |
|---|---|---|---|---|---|---|---|

**Gate:** a site enters Phase 2 if **any** of: (a) ≥2% of steady-state step time
(GPU-busy + stall combined), (b) ≥5 hard syncs/step, (c) it is S3 (the one-line engine
de-sync) which is always worth testing. Everything else is recorded and dropped.

## 6. Phase 2 — Isolated per-candidate replacement experiments

### 6.1 Gating

`VERL_OMNI_KERNEL_RF="S1,S2"` (comma list, default empty). Each site's dispatch lives in
`verl_omni/utils/kernels/`:

```python
# verl_omni/utils/kernels/__init__.py
_rf_enabled = lambda sid: sid in set(filter(None, os.environ.get("VERL_OMNI_KERNEL_RF", "").split(",")))
```

Call sites change from `foo(...)` to `kernel_rf("S1", foo, foo_fused)(...)` — one line, so
the final upstream diff per site stays reviewable and the revert is trivial.

### 6.2 Per-site experiment protocol

For each gated site Sx, exactly one variable changes:

1. `git diff` contains only: the dispatch line at the call site + the kernel/vectorization
   module. No other edits.
2. Run baseline and candidate back-to-back on the same box, same config, same seed:
   `trainer.total_training_steps=12`, discard steps 1–2 (engine/compile warmup), compare
   steps 3–12. **3 repeats each** (interleave B,C,B,C,B,C to decorrelate box drift).
   Timed runs are profiler-free; one extra profiler-overlay run per variant (L2 setup)
   provides launch-count deltas.
3. Collect per step index: `timing_s/*`, `perf/mfu/actor`, images/s, and (overlay runs)
   launch count + GPU-busy. Report **paired** deltas (same step index across runs),
   median and range.
4. **Microbench** each new Triton kernel vs its eager op sequence with
   `triton.testing.do_bench` at representative shapes (Qwen-Image latents fp32:
   B=16×C=16×64×64 at 512-res and ×128×128 at 1024-res), reporting µs, achieved GB/s vs
   peak, and max/rel numeric error on identical inputs. This separates "the kernel is
   good" from "the site was exposed end-to-end".

### 6.3 Parity gates (all must pass before timing counts)

| Gate | How | Threshold |
|---|---|---|
| Unit parity | fixed-seed random inputs, fused vs eager path | S1 `log_prob`: rel err ≤ 1e-5 (fp32 asserted upstream); metric reductions: rel ≤ 1e-6, clipfrac/ratio-count metrics bitwise equal; S3 loss: bitwise identical |
| Sync-clean | `torch.cuda.set_sync_debug_mode("error")` over 3 steps with gate on | zero new warnings vs baseline |
| Trajectory parity | 12-step run, gate on vs off, same seed | per-step loss/clipfrac/KL metric trajectories within run-to-run noise (quantified from Phase-0 repeats) |
| Eager fallback | gate unset + `import triton` failure simulated | code path identical to main; NPU device → eager always |
| Repo gates | `py -V:3.11 -m ruff check` locally; full pytest/pre-commit honestly reported as CI-only (this workspace's venvs cannot run them) | clean |

Note for S1: the fused kernel must take `variance_noise` as an **input** (generated by the
existing `randn_tensor(..., generator=...)` call before the kernel) — randomness stays in
the eager RNG stream so seeded runs remain reproducible across gate on/off.

### 6.4 Reporting

Per site, into the experiment log:

| metric | baseline (med of 3) | candidate (med of 3) | Δ% |
|---|---|---|---|

with rows: step time, each `timing_s/` phase touched by the site, images/s, launches,
sync stalls, max memory; plus the microbench table. Then one **combined run** with all
winning sites gated on simultaneously (overlaps can eat individual wins — Amdahl; the
combined number is what gets upstreamed in the PR body).

## 7. Decision matrix

| Outcome (single site, 8×80GB reference) | Action |
|---|---|
| End-to-end step time ≥3% faster, parity green | Upstream PR (per-site), kernel keeps eager fallback + env default-on via config follow-up |
| Phase-level ≥20% faster AND that phase ≥10% of step | Upstream PR allowed even if end-to-end <3% (correctness-adjacent sync removals land here) |
| Microbench ≥2× but end-to-end <1% | Keep behind `VERL_OMNI_KERNEL_RF`, document, drop from PR plan |
| Parity red or fragile on any gate | Revert; record in log; no retry without a numerics plan |

Aspirational hypothesis to validate (not a promise): combined wins ≥10% steady-state step
time on the reference workload, dominated by S3 + S1 + S2.

## 8. Implementation notes

- Layout: `verl_omni/utils/kernels/{__init__.py, site_markers.py,
  flow_match_sde_step.py, metric_pack.py, jagged_pack.py, …}` — one module per site, each
  exposing `<site>_eager(...)` (the moved original code) and `<site>_fused(...)`, plus the
  dispatcher. Import of `triton` is inside the fused functions with a
  `try/except ImportError` falling back to eager, so a CPU/NPU env can never crash on
  import.
- Triton version guard: assert `torch.version.cuda is not None` and Triton ≥ the bundled
  minimum at first fused call; log once and fall back otherwise. No pinned triton extra in
  `pyproject.toml` (it ships with torch).
- Every fused kernel's docstring states: input dtypes/shapes, the exact eager op sequence
  it replaces, and the tolerance used in its parity test.
- fp32 is already asserted at the S1 boundary (`flow_match_sde.py:196-198`); kernels may
  assume fp32 inputs and must assert it.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Removing a stall just moves the bottleneck to the next stall (Amdahl) | Phase 1 measures all sites on the same baseline; combined run in §6.4 is the decision number |
| Benchmark variance masking small wins | Phase-0 repeatability check (±5%), 3 interleaved repeats, paired per-step deltas, GPU clocks recorded |
| Fused numerics drift compounding over a long run | §6.3 trajectory-parity gate + 12-step metric curves, not just per-op tolerance |
| torch/triton version drift breaking kernels | eager fallback + import-time feature check; kernels use only stable Triton language features (no experimental APIs) |
| Markers/gates leaking into upstream diffs | markers and dispatch are one-line wraps; `grep VERL_OMNI_KERNEL_RF` must list only `verl_omni/utils/kernels/` + call sites before opening a PR |
| Pin drift invalidating Phase-1 numbers mid-study | pins + commit recorded in §4; if pins move, rerun Phase 0 baseline and renormalize all deltas to the new baseline |

## 10. Deliverables and PR plan

1. **PR-A (infra, no behavior change):** `site_markers.py`, `tools/analyze_kernelrf_trace.py`
   (chrome-trace/`cuda_api_sum` → per-site table), marker wraps at Appendix-A sites,
   `docs/perf/kernel_replacement_rfc.md` results section. Complies with
   `references/upstream-contribution-rules.md` (title tag `[diffusion, perf]`, honest
   CI-statement in the body).
2. **PR-B (always-worth one-liner):** S3 engine de-sync
   (`loss.detach().item()` → keep GPU scalar, single `.item()` in
   `postprocess_batch_func`; it is all-reduced + `.item()`'d downstream anyway) with its A/B
   numbers.
3. **PR-C…:** one PR per Phase-2-winning site, each carrying: its Phase-1 cost row, its
   §6.4 table, microbench, parity-test file under `tests/` (CPU-runnable eager-parity part
   only), and a `docs/perf/tuning_guide.md` note if user-visible.
4. **Experiment log** (Section 11 of this file, filled in-place on the fork) with every
   run's config, seeds, raw medians, and the final decision record.

## 11. Experiment log (fill as you go)

```
## Environment
date:            verl-omni commit:
verl pin:        vllm-omni pin:
torch:           triton:
GPU:             driver:
notes:

## Phase 1 — per-site cost table (8 steps, median of steps 3..8)
| site | calls/step | CPU ms | GPU ms | launches | syncs | stall ms | % of step |
| S3_engine_loss_item | | | | | | | |
| S1_sde_step         | | | | | | | |
| ...

## Phase 2 — A/B results (12 steps, median of steps 3..12, 3 interleaved repeats)
| site | step Δ% | phase Δ% | launches Δ | sync Δ | microbench µs (eager→fused) | parity | decision |
```

## Appendix A — Candidate inventory (static audit; costs pending Phase 1)

Type key: **SYNC** = pure de-sync (no kernel), **VEC** = stock-op vectorization (no
custom kernel), **FUSE** = one Triton kernel.

| ID | Site (file:line @ main `0fb3c91`) | Type | Frequency | Replacement sketch |
|---|---|---|---|---|
| S1 | `pipelines/schedulers/flow_match_sde.py:207,214-240,313` (cps/dance variants `:244-310`) | FUSE | per sample × per SDE step, in `gen` AND `old_log_prob`/`teacher` replay | `searchsorted` sigma lookup + single kernel: std_dev_t, prev_sample_mean, prev_sample (noise passed in), Gaussian log-prob reduced to `[B]`; replaces ~15-20 elementwise launches |
| S2 | `trainer/diffusion/diffusion_algos.py:334-341` (flow_grpo), `:426-442` (flow_dppo, 12 `.item()`s), `:559-565` (grpo_guard), `:849-856` (nft), `:731` (dpo, `any().item()` control flow) | FUSE | per micro-batch × per timestep inside `update_actor` (`workers/engine/fsdp/diffusers_impl.py:1029-1046`) | fused metric kernel → packed `[n]` metrics buffer, one D2H; ~6-12 syncs → 1 per call |
| S3 | `workers/engine/fsdp/diffusers_impl.py:1172` (also `:1358`, `:1487`), `workers/engine/veomni/diffusion_impl.py:479` | SYNC | every forward+backward of the T×M loop | keep loss a GPU tensor; single `.item()` in `postprocess_batch_func` (`:592-595`) |
| S4 | `flow_match_sde.py:207`; `pipelines/utils.py:171-181` (`get_sigmas`) | VEC | per sample per step (S1) / per sample per noising step (DPO) | `torch.searchsorted` / broadcast-compare+argmax; kills per-sample `nonzero().item()` |
| S5 | `diffusion_algos.py:240-263` (FlowGRPO adv), `:902-926` + `:928-944` (NFT adv) | VEC (scatter_reduce; FUSE optional) | per step, timed `adv` phase | segmented mean/std via scatter_add/gather; replaces Python groupby (hundreds of small launches + G syncs) |
| S6 | `workers/utils/padding.py:41-64` (`_to_nested`); prompt-embed packs: `pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py:104-118`, `pipelines/qwen_image_flow_grpo/common.py:86-100`, `pipelines/qwen_image_dpo/vllm_omni_rollout_adapter.py:280+`, `pipelines/qwen_image_edit_flow_grpo/vllm_omni_rollout_adapter.py:179+` | FUSE (shared kernel) | 3-5× per step (every logprob/ref/update dispatch) + 2× per rollout batch | one jagged/masked-pack kernel (lengths→cumsum offsets→gather); 6 call-site copies deleted |
| S7 | `pipelines/ltx2_flow_grpo/vllm_omni_rollout_adapter.py:319-364`; `pipelines/minimax_h3_flow_grpo/vllm_omni_rollout_adapter.py:407-434` (`:330`,`:344` setup syncs) | VEC (precompute indices) | per denoising step | precompute gather/scatter index vectors once; removes per-step `.item()`, masked selects, full clones |
| S8 | trajectory collection: `pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py:445-447,506-517` (step-mode `:549,:568-570`); same pattern in `boogu` `:353-355`, `sd3` `:269-341`, `flux_dance` `:326-341` | VEC (prealloc) | per step × per sample, full-latent clone | preallocated `(B, W+1, C, H, Wd)` buffer written in-place per step (or by S1's kernel); removes nested double-stack |
| S9 | `workers/rollout/vllm_rollout/vllm_omni_diffusion_strategy.py:48-56` | FUSE (tiny) | per generated sample | `uint8(clamp(x,0,1)*255+0.5)` kernel with folded finite-check flag; removes `bool(isfinite().all())` sync |
| S10 | `pipelines/qwen3_tts/rollout_utils.py:40-46`; `pipelines/qwen3_tts/talker_forward.py:210-249` | FUSE / VEC | per TTS sample / per actor forward | sliding-window match kernel; batched alignment scatter instead of per-row slice copies + syncs |
| S11 | `trainer/diffusion/rollout_correction.py:207-215`; `trainer/diffusion/diffusion_metric_utils.py:52-93` | FUSE (small) | per step | packed scalar reductions (mean/max/min × rewards/adv/returns + per-timestep means) → one D2H; replaces ~15 `.item()`s + NumPy groupby std |
| S12 | `diffusion_algos.py:946-964` (`_select_train_timesteps`) | VEC | per NFT step | `argsort(rand(B,T))` replaces per-row `randperm` loop |
| S13 | `workers/rollout/vllm_rollout/vllm_omni_ar_strategy.py:400-403` | SYNC (CPU) | per generated token | vectorize logprob extraction (CPU-side); measure via nsys Python sampling, not kernels |

Related non-kernel cleanups found in the audit (fix opportunistically, no experiments
needed): per-sample `torch.equal` checks in `reward_loop/reward_manager/media.py:50-56`
(use `(a != b).any()`), `torch.unique`-per-step in
`pipelines/minimax_h3_flow_grpo/diffusers_training_adapter.py:220-236`,
`online-DPO pair selection` full-batch `.cpu().tolist()` in
`diffusion_algos.py:597-625` (scatter_reduce argmax/argmin instead).
