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

    for token_id in token_ids:
        element_idx = token_id // 32
        bit_idx = token_id % 32
        current_value = vocab_mask[batch_idx, element_idx].item()

        if is_allowed:
            new_value = current_value | (1 << bit_idx)
        else:
            new_value = current_value & (~(1 << bit_idx) & 0xFFFFFFFF)
        vocab_mask[batch_idx, element_idx] = torch.tensor(
            new_value, dtype=torch.int64
        ).to(torch.int32)
