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
"""CPU tests for multi-task On-Policy Distillation (DiffusionOPD, arXiv:2605.15055).

Covers:
- ``MultiTaskRLDataset``: per-sample ``task_id`` / ``teacher_name`` tagging and global
  index <-> (task, local index) mapping.
- ``RoundRobinBatchSampler``: every batch (MOPD round) covers all tasks with equal
  ``batch_size_per_task`` representation.
- Config validation: ``mopd=True`` requires ``opd_only=True`` / ``use_distill_loss=True``
  and ``DiffusionModelConfig`` accepts the ``teacher_adapters`` list.
- ``DistillKLLoss`` with per-task teacher means (the MOPD multi-teacher loss path).
"""

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from verl_omni.utils.dataset.multi_task_dataset import MultiTaskRLDataset, RoundRobinBatchSampler


class _PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "dummy_tensor": torch.tensor([0], dtype=torch.uint8)}


# ---------------------------------------------------------------------------
# MultiTaskRLDataset
# ---------------------------------------------------------------------------


class TestMultiTaskDataset:
    def _make_dataset(self):
        tasks = [_PromptDataset([f"a_{i}" for i in range(10)]), _PromptDataset([f"b_{i}" for i in range(6)])]
        return MultiTaskRLDataset(tasks, task_names=["aes", "ocr"])

    def test_len_and_task_tags(self):
        ds = self._make_dataset()
        assert len(ds) == 16
        # global 0..9 -> task 0, global 10..15 -> task 1
        item0 = ds[0]
        assert item0["task_id"] == 0 and item0["teacher_name"] == "aes"
        item_mid = ds[9]
        assert item_mid["task_id"] == 0
        item_last = ds[15]
        assert item_last["task_id"] == 1 and item_last["teacher_name"] == "ocr"
        # underlying prompt is preserved
        assert ds[10]["prompt"] == "b_0"

    def test_local_index_mapping(self):
        ds = self._make_dataset()
        task_id, local = ds.local_index(12)
        assert task_id == 1 and local == 2
        task_id, local = ds.local_index(7)
        assert task_id == 0 and local == 7

    def test_requires_two_tasks(self):
        with pytest.raises(AssertionError):
            MultiTaskRLDataset([_PromptDataset(["x"])])


class TestRoundRobinSampler:
    def _make_dataset(self):
        tasks = [_PromptDataset([f"a_{i}" for i in range(10)]), _PromptDataset([f"b_{i}" for i in range(6)])]
        return MultiTaskRLDataset(tasks, task_names=["aes", "ocr"])

    def test_batch_covers_all_tasks_evenly(self):
        ds = self._make_dataset()
        sampler = RoundRobinBatchSampler(ds, batch_size_per_task=4, seed=42)
        indices = list(sampler)
        # 2 tasks * 4 samples * 2 rounds (max_task_len=10 // 4 = 2)
        assert len(indices) == 2 * 4 * 2

        # Every round contains exactly 4 samples from each task.
        round_size = 2 * 4
        for r in range(2):
            round_indices = indices[r * round_size : (r + 1) * round_size]
            round_task_ids = [ds.task_of(idx) for idx in round_indices]
            assert round_task_ids.count(0) == 4
            assert round_task_ids.count(1) == 4

        # All returned indices are in range.
        assert all(0 <= idx < len(ds) for idx in indices)

    def test_deterministic_given_seed(self):
        ds = self._make_dataset()
        a = list(RoundRobinBatchSampler(ds, batch_size_per_task=4, seed=7))
        b = list(RoundRobinBatchSampler(ds, batch_size_per_task=4, seed=7))
        assert a == b


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestMopdConfigValidation:
    def test_mopd_requires_opd_only(self):
        from verl_omni.workers.config.diffusion import FSDPDiffusionActorConfig

        with pytest.raises(ValueError, match="mopd=True requires opd_only=True"):
            FSDPDiffusionActorConfig(
                strategy="fsdp",
                rollout_n=1,
                mopd=True,
                opd_only=False,
                use_distill_loss=True,
            )

    def test_mopd_accepts_valid_config(self):
        from verl_omni.workers.config.diffusion import FSDPDiffusionActorConfig

        cfg = FSDPDiffusionActorConfig(
            strategy="fsdp",
            rollout_n=1,
            mopd=True,
            opd_only=True,
            use_distill_loss=True,
            distill_loss_mode="distill_kl",
        )
        assert cfg.mopd is True and cfg.opd_only is True


class TestTeacherAdaptersModelConfig:
    def _make_model_dir(self, tmp_path):
        import json

        model_dir = tmp_path / "sd35"
        model_dir.mkdir()
        (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusion3Pipeline"}))
        transformer_dir = model_dir / "transformer"
        transformer_dir.mkdir()
        (transformer_dir / "config.json").write_text(json.dumps({"in_channels": 16}))
        return model_dir

    def test_teacher_adapters_defaults_empty(self, tmp_path):
        from verl_omni.workers.config import DiffusionModelConfig

        model_dir = self._make_model_dir(tmp_path)
        cfg = DiffusionModelConfig(
            path=str(model_dir),
            algorithm="flow_grpo",
            load_tokenizer=False,
            attn_backend="native",
        )
        assert cfg.teacher_adapters == []

    def test_teacher_adapters_settable(self, tmp_path):
        from verl_omni.workers.config import DiffusionModelConfig

        model_dir = self._make_model_dir(tmp_path)
        cfg = DiffusionModelConfig(
            path=str(model_dir),
            algorithm="flow_grpo",
            load_tokenizer=False,
            attn_backend="native",
            teacher_adapters=[
                {"name": "aes", "path": "quanhaol/Aes-Teacher", "guidance_scale": 4.5},
                {"name": "ocr", "path": "some/ocr-teacher", "guidance_scale": 4.5},
                {"name": "geneval", "path": "some/geneval-teacher", "guidance_scale": 1.0},
            ],
        )
        assert len(cfg.teacher_adapters) == 3
        assert cfg.teacher_adapters[0]["name"] == "aes"
        assert cfg.teacher_adapters[2]["guidance_scale"] == 1.0


# ---------------------------------------------------------------------------
# DistillKLLoss with per-task teacher means (MOPD multi-teacher loss path)
# ---------------------------------------------------------------------------


class TestMopdDistillLoss:
    def test_per_task_teacher_means_match_student_for_one_task(self):
        """When the student already matches the teacher of one task, the loss is low."""
        from verl_omni.trainer.diffusion.diffusion_algos import DistillKLLoss

        torch.manual_seed(0)
        bsz, timesteps, dim = 8, 16, 3
        student_mean = torch.randn(bsz, timesteps, dim)

        # Two task groups in one batch; the task-0 teacher matches the student, the
        # task-1 teacher differs, simulating the MOPD round-robin batch.
        teacher_prev_sample_mean = student_mean.clone()
        teacher_prev_sample_mean[4:] = teacher_prev_sample_mean[4:] + 2.0
        std_dev_t = torch.zeros(bsz, 1, 1)  # ODE regime -> squared-L2 surrogate

        loss, metrics = DistillKLLoss.compute_loss(
            prev_sample_mean=student_mean,
            teacher_prev_sample_mean=teacher_prev_sample_mean,
            std_dev_t=std_dev_t,
        )

        # ODE surrogate: 0.5 * ||mu_s - mu_t||^2. Rows 0-3 contribute 0; rows 4-7 -> 2.0 each.
        assert loss.item() == pytest.approx(1.0, rel=1e-5)
        assert "actor/distill_kl_loss" in metrics

    def test_loss_finite_for_mopd_batch(self):
        from verl_omni.trainer.diffusion.diffusion_algos import DistillKLLoss

        torch.manual_seed(1)
        bsz, timesteps, dim = 24, 10, 4  # M=3 tasks x 8 prompts x 10 denoise steps
        student_mean = torch.randn(bsz, timesteps, dim)
        teacher_prev_sample_mean = student_mean + 0.1 * torch.randn_like(student_mean)
        std_dev_t = torch.zeros(bsz, 1, 1)

        loss, _ = DistillKLLoss.compute_loss(
            prev_sample_mean=student_mean,
            teacher_prev_sample_mean=teacher_prev_sample_mean,
            std_dev_t=std_dev_t,
        )
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# MOPD teacher-inference reassembly helper (mask scatter/gather)
# ---------------------------------------------------------------------------


class TestMopdTeacherReassembly:
    def test_scatter_back_preserves_row_order(self):
        """Per-task teacher means computed on row subsets must scatter back to the
        original row order (the logic inside ``_compute_mopd_teacher_log_prob``)."""
        torch.manual_seed(3)
        bsz, timesteps, dim = 12, 8, 3
        full_teacher_mean = torch.randn(bsz, timesteps, dim)

        task_ids = np.array([0, 0, 1, 1, 1, 0, 2, 2, 0, 1, 2, 2])
        per_task = {}
        for task_id in np.unique(task_ids):
            mask = task_ids == task_id
            # simulate the per-task teacher output on the subset rows
            per_task[int(task_id)] = (mask, full_teacher_mean[mask].clone())

        reassembled = torch.zeros_like(full_teacher_mean)
        for _, (mask, prev_sample_mean) in per_task.items():
            reassembled[torch.from_numpy(mask)] = prev_sample_mean

        assert torch.allclose(reassembled, full_teacher_mean)
