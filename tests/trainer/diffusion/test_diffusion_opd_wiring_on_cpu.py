# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for the OPD-only training path (DiffusionOPD, arXiv:2605.15055).

Covers:
- ``DistillKLLoss`` ODE regime (``std_dev_t == 0`` -> squared-L2 surrogate, paper Eq. 12).
- ``diffusion_loss`` OPD-only short-circuit: primary policy-gradient loss skipped, only the
  distillation term contributes, and no ``old_log_probs`` / ``advantages`` are required.
- Config validation: ``opd_only=True`` requires ``use_distill_loss=True``.
- ``DiffusionModelConfig`` accepts ``teacher_adapter_path`` / ``teacher_adapter_name``.
"""

import os

import pytest
import torch

from verl_omni.trainer.diffusion.diffusion_algos import DistillKLLoss
from verl_omni.workers.config import DiffusionModelConfig

# ---------------------------------------------------------------------------
# DistillKLLoss ODE regime (std_dev_t == 0)
# ---------------------------------------------------------------------------


class TestDistillKLOdeRegime:
    def test_ode_zero_variance_uses_squared_l2_surrogate(self):
        """When std_dev_t == 0 the KL degenerates to 0.5 * ||mu_s - mu_t||^2 (paper Eq. 12)."""
        mean = torch.zeros(4, 16, 3)
        teacher_mean = torch.full((4, 16, 3), 2.0)
        std_dev_t = torch.zeros(4, 1, 1)

        loss, _ = DistillKLLoss.compute_loss(
            prev_sample_mean=mean,
            teacher_prev_sample_mean=teacher_mean,
            std_dev_t=std_dev_t,
        )

        # 0.5 * (2.0)^2 = 2.0
        assert loss.item() == pytest.approx(2.0, rel=1e-5)

    def test_ode_does_not_return_nan_or_inf(self):
        mean = torch.randn(4, 16, 3)
        teacher_mean = torch.randn(4, 16, 3)
        std_dev_t = torch.zeros(4, 1, 1)

        loss, _ = DistillKLLoss.compute_loss(
            prev_sample_mean=mean,
            teacher_prev_sample_mean=teacher_mean,
            std_dev_t=std_dev_t,
        )

        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_ode_identical_means_gives_zero(self):
        mean = torch.randn(4, 16, 3)
        std_dev_t = torch.zeros(4, 1, 1)

        loss, _ = DistillKLLoss.compute_loss(
            prev_sample_mean=mean,
            teacher_prev_sample_mean=mean.clone(),
            std_dev_t=std_dev_t,
        )

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_sde_unchanged_for_nonzero_variance(self):
        """The SDE closed-form (paper Eq. 11) is preserved when std_dev_t > 0."""
        mean = torch.zeros(4, 16, 3)
        teacher_mean = torch.full((4, 16, 3), 2.0)
        std_dev_t = torch.ones(4, 1, 1)

        loss, _ = DistillKLLoss.compute_loss(
            prev_sample_mean=mean,
            teacher_prev_sample_mean=teacher_mean,
            std_dev_t=std_dev_t,
        )

        # KL = ||Δ||^2 / (2 σ²) = 4 / 2 = 2.0
        assert loss.item() == pytest.approx(2.0, rel=1e-5)

    def test_mixed_batch_sde_and_ode(self):
        """A batch mixing SDE (σ>0) and ODE (σ=0) rows routes each correctly."""
        mean = torch.zeros(2, 16, 3)
        teacher_mean = torch.full((2, 16, 3), 2.0)
        std_dev_t = torch.tensor([[[1.0]], [[0.0]]])  # row 0 SDE, row 1 ODE

        loss, _ = DistillKLLoss.compute_loss(
            prev_sample_mean=mean,
            teacher_prev_sample_mean=teacher_mean,
            std_dev_t=std_dev_t,
        )

        # mean over batch: (2.0 + 2.0) / 2 = 2.0
        assert loss.item() == pytest.approx(2.0, rel=1e-5)


# ---------------------------------------------------------------------------
# diffusion_loss OPD-only short-circuit
# ---------------------------------------------------------------------------


def _compose_actor_config(overrides):
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    import verl_omni

    config_dir = os.path.join(os.path.dirname(verl_omni.__file__), "trainer/config/diffusion/actor")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="dp_diffusion_actor",
            overrides=["strategy=fsdp", "ppo_micro_batch_size_per_gpu=4", *overrides],
        )
    return omega_conf_to_dataclass(cfg)


class TestOpdOnlyLoss:
    def _build_batch_no_pg_signals(self):
        """Batch without old_log_probs / advantages (OPD-only)."""
        from verl.utils import tensordict_utils as tu

        torch.manual_seed(0)
        model_output = {
            "prev_sample_mean": torch.randn(4, 16, 3),
            "std_dev_t": torch.zeros(4, 1, 1),  # ODE regime
        }
        data = tu.get_tensordict(
            {
                "teacher_prev_sample_mean": torch.randn(4, 16, 3),
            }
        )
        tu.assign_non_tensor(data, gradient_accumulation_steps=1, sp_size=1)
        return model_output, data

    def test_opd_only_returns_only_distill_term(self):
        from verl_omni.workers.utils.losses import diffusion_loss

        actor_cfg = _compose_actor_config(
            overrides=[
                "use_distill_loss=true",
                "distill_loss_mode=distill_kl",
                "distill_loss_coef=1.0",
                "opd_only=true",
            ]
        )
        assert actor_cfg.opd_only is True
        model_output, data = self._build_batch_no_pg_signals()

        total_loss, metrics = diffusion_loss(actor_cfg, model_output, data)

        distill_loss_value, _ = DistillKLLoss.compute_loss(
            prev_sample_mean=model_output["prev_sample_mean"],
            teacher_prev_sample_mean=data["teacher_prev_sample_mean"],
            std_dev_t=model_output["std_dev_t"],
        )
        assert total_loss.item() == pytest.approx(distill_loss_value.item(), rel=1e-5)
        assert "actor/distill_kl_loss" in metrics

    def test_opd_only_works_without_old_log_probs(self):
        """OPD-only must not require old_log_probs / advantages on the batch."""
        from verl_omni.workers.utils.losses import diffusion_loss

        actor_cfg = _compose_actor_config(
            overrides=["use_distill_loss=true", "opd_only=true"]
        )
        model_output, data = self._build_batch_no_pg_signals()

        # Must not raise despite the missing policy-gradient signals.
        total_loss, _ = diffusion_loss(actor_cfg, model_output, data)
        assert torch.isfinite(total_loss)

    def test_non_opd_still_requires_old_log_probs(self):
        """Sanity: without opd_only, the default flow_grpo path still needs the PG signals
        (this guards against accidentally making the primary loss optional in normal mode)."""
        from verl.utils import tensordict_utils as tu

        from verl_omni.workers.utils.losses import diffusion_loss

        actor_cfg = _compose_actor_config(overrides=[])
        assert actor_cfg.opd_only is False

        torch.manual_seed(1)
        model_output = {
            "log_probs": torch.randn(4),
            "prev_sample_mean": torch.randn(4, 16, 3),
            "std_dev_t": torch.ones(4, 1, 1),
        }
        data = tu.get_tensordict(
            {
                "old_log_probs": torch.randn(4),
                "advantages": torch.randn(4),
            }
        )
        tu.assign_non_tensor(data, gradient_accumulation_steps=1, sp_size=1)

        total_loss, _ = diffusion_loss(actor_cfg, model_output, data)
        assert torch.isfinite(total_loss)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestOpdConfigValidation:
    def test_opd_only_requires_distill_loss(self):
        from verl_omni.workers.config.diffusion import FSDPDiffusionActorConfig

        with pytest.raises(ValueError, match="opd_only=True requires use_distill_loss"):
            FSDPDiffusionActorConfig(
                strategy="fsdp",
                rollout_n=1,
                opd_only=True,
                use_distill_loss=False,
            )

    def test_opd_only_accepts_valid_config(self):
        from verl_omni.workers.config.diffusion import FSDPDiffusionActorConfig

        cfg = FSDPDiffusionActorConfig(
            strategy="fsdp",
            rollout_n=1,
            opd_only=True,
            use_distill_loss=True,
            distill_loss_mode="distill_kl",
        )
        assert cfg.opd_only is True


# ---------------------------------------------------------------------------
# DiffusionModelConfig teacher fields
# ---------------------------------------------------------------------------


class TestTeacherModelConfigFields:
    def _make_model_dir(self, tmp_path):
        import json

        model_dir = tmp_path / "sd35"
        model_dir.mkdir()
        (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusion3Pipeline"}))
        transformer_dir = model_dir / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(json.dumps({"in_channels": 16}))
        return model_dir

    def test_defaults(self, tmp_path):
        model_dir = self._make_model_dir(tmp_path)
        cfg = DiffusionModelConfig(
            path=str(model_dir),
            algorithm="flow_grpo",
            load_tokenizer=False,
            attn_backend="native",
        )
        assert cfg.teacher_adapter_path is None
        assert cfg.teacher_adapter_name == "teacher"

    def test_settable(self, tmp_path):
        model_dir = self._make_model_dir(tmp_path)
        cfg = DiffusionModelConfig(
            path=str(model_dir),
            algorithm="flow_grpo",
            load_tokenizer=False,
            attn_backend="native",
            teacher_adapter_path="quanhaol/Aes-Teacher",
            teacher_adapter_name="teacher",
        )
        assert cfg.teacher_adapter_path == "quanhaol/Aes-Teacher"
        assert cfg.teacher_adapter_name == "teacher"