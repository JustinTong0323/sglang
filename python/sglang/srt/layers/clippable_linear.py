# Copyright 2025 SGLang Team
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
"""TP-sharded linear wrappers with per-tensor activation clamping.

Used by the Gemma 4 vision and audio encoders.  Each wrapper owns a parallel
linear and four scalar clip buffers (``input_min/max``, ``output_min/max``)
that default to ±inf (no-op) and are populated from the checkpoint.

For fused projections (QKV, GateUp), input bounds are shared (the checkpoint
stores identical copies per projection — last write wins during loading) and
output bounds are per-projection.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import add_prefix


_INF = float("inf")


class ClippableRowParallelLinear(nn.Module):
    """``RowParallelLinear`` with input/output activation clamping.

    Checkpoint weight at ``<name>.weight`` is remapped to ``<name>.linear.weight``
    by the model's ``load_weights``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.linear = RowParallelLinear(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear", prefix),
        )
        self.input_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)
        self.input_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)
        self.output_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)
        self.output_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, self.input_min, self.input_max)
        x, _ = self.linear(x)
        x = torch.clamp(x, self.output_min, self.output_max)
        return x


class ClippableColumnParallelLinear(nn.Module):
    """``ColumnParallelLinear`` with input/output activation clamping."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.linear = ColumnParallelLinear(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("linear", prefix),
        )
        self.input_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)
        self.input_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)
        self.output_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)
        self.output_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, self.input_min, self.input_max)
        x, _ = self.linear(x)
        x = torch.clamp(x, self.output_min, self.output_max)
        return x


class ClippableQKVParallelLinear(nn.Module):
    """Fused QKV projection with per-projection activation clamping.

    Owns a single ``QKVParallelLinear`` for the fused matmul.  Clip bounds
    are stored as flat buffers: shared ``input_min/max`` (applied before the
    matmul) and per-projection ``q/k/v_output_min/max`` (applied after split).
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_attention_tp_size()
        self.q_size = (total_num_heads // tp_size) * head_size
        self.kv_size = (total_num_kv_heads // tp_size) * head_size

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_size,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.input_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)
        self.input_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)
        self.q_output_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)
        self.q_output_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)
        self.k_output_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)
        self.k_output_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)
        self.v_output_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)
        self.v_output_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.clamp(hidden_states, self.input_min, self.input_max)
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = torch.clamp(q, self.q_output_min, self.q_output_max)
        k = torch.clamp(k, self.k_output_min, self.k_output_max)
        v = torch.clamp(v, self.v_output_min, self.v_output_max)
        return q, k, v


class ClippableGateUpParallelLinear(nn.Module):
    """Fused gate/up projection with per-projection activation clamping.

    Same pattern as ``ClippableQKVParallelLinear``: owns a single
    ``MergedColumnParallelLinear`` for the fused matmul, with shared input
    bounds and per-projection output bounds as flat buffers.
    """

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        *,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_attention_tp_size()
        self.proj_size = intermediate_size // tp_size

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=input_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.input_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)
        self.input_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)
        self.gate_output_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)
        self.gate_output_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)
        self.up_output_min = nn.parameter.Buffer(torch.tensor(-_INF), persistent=False)
        self.up_output_max = nn.parameter.Buffer(torch.tensor(_INF), persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.clamp(x, self.input_min, self.input_max)
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.split([self.proj_size, self.proj_size], dim=-1)
        gate = torch.clamp(gate, self.gate_output_min, self.gate_output_max)
        up = torch.clamp(up, self.up_output_min, self.up_output_max)
        return gate, up
