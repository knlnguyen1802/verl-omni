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
"""Colocate-async v1 policy-gradient diffusion trainer.

Mirrors upstream ``PPOTrainerColocateAsync`` hook semantics, adapted to
verl-omni diffusion rollout:

1. Trainer and rollout are colocated on one shared GPU pool.
2. Partial rollout is enabled. In-flight samples aborted at a mode transition
   are retried as whole samples by ``DiffusionWholeSampleRetryLLMServerClient``;
   there is no token-append resume for diffusion.
3. Warmup batches keep generation in flight across the first step boundary.

Diffusion-specific compute (reward, old/ref log-prob, Flow-GRPO advantage,
actor update, metrics, dumping) lives in ``PolicyGradientDiffusionTrainerV1``;
this subclass only defines the mode lifecycle hooks.
"""

import logging
import os

from verl.utils.debug import marked_timer

from verl_omni.trainer.diffusion.v1.trainer_base import (
    PolicyGradientDiffusionTrainerV1,
    register_diffusion_trainer,
)
from verl_omni.workers.rollout.diffusion_llm_server import DiffusionWholeSampleRetryLLMServerClient

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_diffusion_trainer("colocate_async")
class PolicyGradientDiffusionTrainerV1ColocateAsync(PolicyGradientDiffusionTrainerV1):
    """Asynchronous policy-gradient diffusion trainer (v1) with colocated rollout.

    Hook behavior:

    - ``on_init_end``: update rollout weights at the current ``global_steps``.
    - ``on_train_begin``: submit ``trainer.v1.colocate_async.num_warmup_batches``
      warmup batches so generation is already in flight when the first step
      starts sampling.
    - ``on_sample_end``: abort in-flight requests, then sleep the replicas to
      free weight memory for the actor update. Aborted samples are retried as
      whole samples by the client once generation resumes.
    - ``on_step_end``: update rollout weights from the freshly trained actor,
      then resume generation. The update must come first: waking clears the
      sleeping tags while admission is still paused, so retried requests never
      observe a slept engine.
    """

    def get_llm_client(self):
        """Get the diffusion whole-sample-retry client for the colocated rollout."""
        return self.llm_server_manager.get_client(client_cls=DiffusionWholeSampleRetryLLMServerClient)

    def on_init_end(self):
        # update weights after loading checkpoint
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_train_begin(self):
        num_warmup_batches = self.config.trainer.v1.colocate_async.num_warmup_batches
        for _ in range(num_warmup_batches):
            self._add_batch_to_generate()
        logger.info(f"Added {num_warmup_batches} warmup batches to the agent loop manager")

    def on_sample_end(self):
        # abort all unfinished requests and pause generation
        self.checkpoint_manager.abort_replicas()
        # sleep all replicas to discard weights and (no-op) KV cache
        self.checkpoint_manager.sleep_replicas()

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights from the freshly-trained actor
            self.checkpoint_manager.update_weights(self.global_steps)
            # resume generation
            self.checkpoint_manager.resume_generation_replicas()
