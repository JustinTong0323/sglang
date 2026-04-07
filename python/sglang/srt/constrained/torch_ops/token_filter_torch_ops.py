# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""Torch impl for token filter operations."""

from typing import List

import torch


def _to_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    if value >= (1 << 31):
        value -= 1 << 32
    return value


def set_token_filter_torch(
    vocab_mask: torch.Tensor,
    token_ids: List[int],
    batch_idx: int,
    is_allowed: bool = True,
    reset_vocab_mask: bool = True,
):
    if reset_vocab_mask:
        mask_val = -1 if (not is_allowed) else 0
        vocab_mask[batch_idx].fill_(mask_val)

    if not token_ids:
        return

    aggregated_masks = {}
    for token_id in token_ids:
        element_idx = token_id // 32
        bit_idx = token_id % 32
        aggregated_masks[element_idx] = aggregated_masks.get(element_idx, 0) | (
            1 << bit_idx
        )

    row = vocab_mask[batch_idx]
    element_indices = torch.tensor(
        list(aggregated_masks.keys()), dtype=torch.long, device=row.device
    )
    bitmasks = torch.tensor(
        [
            _to_int32(mask if is_allowed else ~mask)
            for mask in aggregated_masks.values()
        ],
        dtype=row.dtype,
        device=row.device,
    )

    if is_allowed:
        row[element_indices] = torch.bitwise_or(row[element_indices], bitmasks)
    else:
        row[element_indices] = torch.bitwise_and(row[element_indices], bitmasks)
