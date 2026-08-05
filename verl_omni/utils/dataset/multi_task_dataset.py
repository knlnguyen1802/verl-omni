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
"""Multi-task round-robin data for DiffusionOPD (arXiv:2605.15055).

Stage 2 of DiffusionOPD samples prompts from every task's dataset in a round-robin
fashion: each MOPD round visits each task exactly once (``batch_size_per_task`` samples
per task), so the training step covers a full task cycle before a single optimizer step
(gradient accumulation G = M, paper Algorithm 1).

This module provides:
- :class:`MultiTaskRLDataset`: concatenates M per-task RL datasets and tags every sample
  with its ``task_id`` (int) and ``teacher_name`` (str). The collate_fn turns these into
  object arrays in ``DataProto.non_tensor_batch`` so the trainer can route each sample to
  its task-specific teacher during on-policy distillation.
- :class:`RoundRobinBatchSampler`: yields one batch per round containing equal
  ``batch_size_per_task`` samples from every task.
"""

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, Sampler

from verl_omni.utils.dataset.rl_dataset import create_rl_dataset

__all__ = ["MultiTaskRLDataset", "RoundRobinBatchSampler", "create_multi_task_rl_datasets"]


class MultiTaskRLDataset(Dataset):
    """Concatenation of M task datasets with per-sample task tags."""

    def __init__(self, task_datasets: list[Dataset], task_names: list[str] | None = None):
        assert len(task_datasets) >= 2, "MultiTaskRLDataset requires at least two task datasets."
        self.task_datasets = task_datasets
        self.num_tasks = len(task_datasets)
        if task_names is None:
            task_names = [f"task_{i}" for i in range(self.num_tasks)]
        assert len(task_names) == self.num_tasks, (
            f"task_names ({len(task_names)}) must match the number of tasks ({self.num_tasks})."
        )
        self.task_names = list(task_names)
        self.task_lens = [len(ds) for ds in task_datasets]
        self.cum_lens = np.cumsum([0] + self.task_lens)
        if any(length <= 0 for length in self.task_lens):
            raise ValueError(f"All task datasets must be non-empty, got lengths {self.task_lens}.")

    def __len__(self) -> int:
        return int(self.cum_lens[-1])

    def task_of(self, idx: int) -> int:
        """Return the task id of a global index."""
        return int(np.searchsorted(self.cum_lens, idx, side="right") - 1)

    def local_index(self, idx: int) -> tuple[int, int]:
        """Map a global index to ``(task_id, local_index_within_task)``."""
        task_id = self.task_of(idx)
        return task_id, int(idx - self.cum_lens[task_id])

    def __getitem__(self, idx: int):
        task_id, local_idx = self.local_index(idx)
        item = self.task_datasets[task_id][local_idx]
        # Task tags flow through the default collate_fn as object arrays
        # (``DataProto.non_tensor_batch``) and are consumed by the trainer's
        # per-task teacher inference.
        item = {**item, "task_id": task_id, "teacher_name": self.task_names[task_id]}
        return item


class RoundRobinBatchSampler(Sampler):
    """Yield samples in a round-robin order over tasks for DiffusionOPD.

    One MOPD round emits ``batch_size_per_task`` samples from every task (task 0, then
    task 1, ...), so a dataloader configured with ``batch_size = M * batch_size_per_task``
    produces exactly one batch per round, i.e. one full task cycle per training step
    (paper Algorithm 1: gradient accumulation over a full round-robin cycle).
    """

    def __init__(
        self,
        dataset: MultiTaskRLDataset,
        batch_size_per_task: int,
        seed: int = 42,
    ):
        assert isinstance(dataset, MultiTaskRLDataset)
        self.dataset = dataset
        self.num_tasks = dataset.num_tasks
        self.batch_size_per_task = int(batch_size_per_task)
        assert self.batch_size_per_task >= 1, "batch_size_per_task must be >= 1"
        self.seed = seed
        self.epoch = 0

        # Base the number of rounds on the longest task so every task is visited on
        # every round; shorter tasks are sampled with replacement across rounds.
        max_task_len = max(dataset.task_lens)
        self.num_rounds = max(1, max_task_len // self.batch_size_per_task)

    def __len__(self) -> int:
        # Total number of samples the sampler emits; the dataloader then chunks them
        # into ``num_rounds`` batches of size ``M * batch_size_per_task``.
        return self.num_rounds * self.num_tasks * self.batch_size_per_task

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        for _ in range(self.num_rounds):
            for t in range(self.num_tasks):
                task_len = self.dataset.task_lens[t]
                if task_len >= self.batch_size_per_task:
                    chosen = torch.randperm(task_len, generator=g)[: self.batch_size_per_task].tolist()
                else:
                    chosen = torch.randint(0, task_len, (self.batch_size_per_task,), generator=g).tolist()
                yield from (self.dataset.cum_lens[t] + i for i in chosen)


def create_multi_task_rl_datasets(
    task_files: list[list[str]],
    data_config: DictConfig,
    tokenizer,
    processor,
    task_names: list[str] | None = None,
    max_samples: int = -1,
    is_train: bool = True,
) -> MultiTaskRLDataset:
    """Build one RL dataset per task and wrap them in a :class:`MultiTaskRLDataset`.

    Args:
        task_files: One list of data file paths per task (``data.task_train_files``).
        data_config: The ``data`` hydra config.
        tokenizer / processor: Passed through to ``create_rl_dataset``.
        task_names: Optional per-task names (defaults to the teacher adapter names).
        max_samples: Optional cap per task dataset.
        is_train: Whether these are training (True) or validation (False) datasets.
    """
    datasets = [
        create_rl_dataset(
            files,
            data_config,
            tokenizer,
            processor,
            is_train=is_train,
            max_samples=max_samples,
        )
        for files in task_files
    ]
    return MultiTaskRLDataset(datasets, task_names=task_names)
