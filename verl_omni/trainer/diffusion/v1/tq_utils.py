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

import logging
import os
from typing import Any

import numpy as np
import torch
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.protocol import DataProto
from verl.utils import tensordict_utils as tu

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _unwrap_non_tensor_item(item: Any) -> Any:
    """Unwrap common non-tensor wrappers (e.g. NonTensorData) to raw values."""
    value = item
    # Some TransferQueue paths return wrapper objects with a ``data`` attribute.
    # Unwrap a few levels defensively until reaching a plain python value.
    for _ in range(4):
        if isinstance(value, str | bytes | bytearray | dict | list | tuple | np.ndarray | torch.Tensor):
            break
        data_attr = getattr(value, "data", None)
        if data_attr is None or data_attr is value:
            break
        value = data_attr
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_object_array(value: Any) -> np.ndarray:
    """Normalize non-tensor TransferQueue values to object ndarray."""
    if isinstance(value, np.ndarray) and value.dtype == object:
        items = value.tolist()
    elif isinstance(value, np.ndarray):
        items = value.tolist()
    elif isinstance(value, str | bytes | dict):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    items = [_unwrap_non_tensor_item(item) for item in items]
    arr = np.empty(len(items), dtype=object)
    arr[:] = items
    return arr


def _stack_field(value: Any, padding: float = 0.0) -> torch.Tensor | None:
    """Normalize a TransferQueue-returned field into a stacked tensor.

    Diffusion worker outputs are pre-padded to fixed shapes (prompt_length for
    token ids, fixed C/H/W for images, max_prompt_embed_length for embeds), so
    most fields come back already stacked. Variable-length fallbacks are padded
    so the downstream diffusion compute path receives uniform batch tensors.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor) and value.is_nested:
        return value.to_padded_tensor(padding=padding)
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "to_padded_tensor"):
        return value.to_padded_tensor(padding=padding)
    if isinstance(value, list | tuple):
        tensors = [v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in value]
        if not tensors:
            return None
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding)
    return torch.as_tensor(value)


_SHARED_PROMPT_FIELDS = (
    "prompt_embeds",
    "negative_prompt_embeds",
    "prompt_embeds_mask",
    "negative_prompt_embeds_mask",
)


def _merge_prompt_shared_fields(
    data: dict[str, Any],
    keys: list[str],
    partition_id: str,
) -> dict[str, Any]:
    """Expand per-uid shared fields stored in ``{partition_id}_prompt_shared`` onto every row.

    The agent loop worker writes session-invariant tensors (frozen text-encoder
    embeddings/masks) once per prompt uid in a dedicated partition. This merges
    them back so each trajectory row carries the same embeds, without having
    serialized ``rollout.n`` copies through TransferQueue.
    """
    uids = [key.rsplit("_", 2)[0] for key in keys]
    unique_uids: list[str] = []
    uid_to_row: dict[str, int] = {}
    for uid in uids:
        if uid not in uid_to_row:
            uid_to_row[uid] = len(unique_uids)
            unique_uids.append(uid)
    if not unique_uids:
        return data

    shared_partition_id = f"{partition_id}_prompt_shared"
    try:
        # kv_batch_get returns a TensorDict; normalize to a plain dict so the
        # emptiness check and key/field access below are well-defined.
        shared = dict(
            tq.kv_batch_get(keys=unique_uids, partition_id=shared_partition_id)
        )
    except Exception as e:
        logger.warning(f"Failed to fetch shared prompt fields from {shared_partition_id}: {e}")
        return data
    if not shared:
        return data

    row_to_uid = [uid_to_row[uid] for uid in uids]
    for field in _SHARED_PROMPT_FIELDS:
        if field not in shared or shared[field] is None:
            continue
        value = shared[field]
        # Nested tensors don't support advanced indexing; pad first (embeds are
        # fixed-shape so this is a no-op in practice, matching ``_stack_field``).
        if isinstance(value, torch.Tensor) and value.is_nested:
            value = value.to_padded_tensor(padding=0.0)
        if isinstance(value, torch.Tensor):
            data[field] = value[row_to_uid]
        else:
            data[field] = [value[i] for i in row_to_uid]
    return data


def diffusion_tq_batch_to_dataproto(
    batch_meta: KVBatchMeta,
    pad_token_id: int = 0,
) -> DataProto:
    """Read selected TQ rows and assemble a diffusion ``DataProto``.

    Args:
        batch_meta: ``KVBatchMeta`` returned by ``ReplayBuffer.sample``.
        pad_token_id: Padding token id for variable-length prompt token tensors.

    Returns:
        ``DataProto`` whose ``batch`` carries diffusion tensors (prompts,
        responses, rollout_log_probs, rm_scores, embeds, ...) and whose
        ``non_tensor_batch`` carries uid/reward_model/data_source/extra_fields.
    """
    keys = list(batch_meta.keys)
    partition_id = batch_meta.partition_id

    data = dict(
        tq.kv_batch_get(
            keys=keys,
            partition_id=partition_id,
        )
    )
    # Expand per-prompt shared fields (text-encoder embeds) onto every row.
    data = _merge_prompt_shared_fields(data, keys, partition_id)

    batch_dict: dict[str, torch.Tensor] = {}
    non_tensor_batch: dict[str, Any] = {}
    for field, value in data.items():
        padding = float(pad_token_id) if field == "prompts" else 0.0
        stacked = None
        try:
            stacked = _stack_field(value, padding=padding)
        except (TypeError, ValueError, RuntimeError):
            # Non-numeric/object fields are forwarded via non_tensor_batch.
            stacked = None

        if stacked is not None and isinstance(stacked, torch.Tensor):
            batch_dict[field] = stacked
            continue
        non_tensor_batch[field] = _to_object_array(value)

    # Unpack extra_fields dict rows into top-level non_tensor_batch keys so the
    # diffusion compute path can read min/max_global_steps and reward_extra_info.
    extra_fields_arr = non_tensor_batch.pop("extra_fields", None)
    if extra_fields_arr is not None:
        for i, extra in enumerate(extra_fields_arr.tolist()):
            if not isinstance(extra, dict):
                continue
            for k, v in extra.items():
                if k not in non_tensor_batch:
                    non_tensor_batch[k] = np.empty(len(extra_fields_arr), dtype=object)
                non_tensor_batch[k][i] = v

    batch = TensorDict(batch_dict, batch_size=len(keys))
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def put_dataproto_fields_to_tq(
    batch_meta: KVBatchMeta,
    data: DataProto,
    fields: list[str],
) -> None:
    """Write selected ``DataProto`` batch fields back to TransferQueue.

    Args:
        batch_meta: ``KVBatchMeta`` whose ``keys`` and ``partition_id`` target
            the rows to update.
        data: ``DataProto`` containing the computed tensor fields.
        fields: Batch field names to persist; missing fields are skipped.
    """
    output: dict[str, Any] = {}
    for field in fields:
        if field not in data.batch:
            continue
        output[field] = data.batch[field]
    if not output:
        return
    tq.kv_batch_put(
        keys=list(batch_meta.keys),
        partition_id=batch_meta.partition_id,
        fields=tu.get_tensordict(output),
    )


def sort_diffusion_tq_keys(keys: list[str]) -> list[int]:
    """Return indices that sort TransferQueue keys in rollout order.

    Args:
        keys: TransferQueue row keys from ``KVBatchMeta.keys``.

    Returns:
        Permutation indices that reorder ``keys`` by ``(uid, rollout, output)``.
    """
    sort_keys = []
    for key in keys:
        parts = key.rsplit("_", 2)
        if len(parts) == 3:
            sort_keys.append((parts[0], int(parts[1]), int(parts[2])))
        else:
            sort_keys.append((key, 0, 0))
    return sorted(range(len(keys)), key=lambda i: sort_keys[i])
