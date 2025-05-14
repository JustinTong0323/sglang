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
import copy
from typing import Iterable, Optional, Sequence, Set, Tuple

import torch
from torch import nn
from transformers import ROPE_INIT_FUNCTIONS, AutoModel, PreTrainedModel
from transformers.activations import ACT2FN

from sglang.srt.configs import Gemma3p5TextConfig
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import GeluAndMul
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.utils import add_prefix
from sglang.utils import logger


# Aligned with HF's implementation, using sliding window inclusive with the last token
# SGLang assumes exclusive
def get_attention_sliding_window_size(config):
    return config.sliding_window - 1


# Adapted from:
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma3.py
def extract_layer_index(prefix: str) -> int:
    """Extract the layer index from a prefix string."""
    parts = prefix.split(".")
    for part in parts:
        if part.startswith("layers."):
            layer_str = part.split(".")[-1]
            try:
                return int(layer_str)
            except ValueError:
                continue
    return -1


class Gemma3p5MLP(nn.Module):
    def __init__(
        self,
        config: Gemma3p5TextConfig,
        layer_idx: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Raise error if hidden_activation is not gelu_pytorch_tanh
        if config.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma3p5 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_activation` to "
                "`gelu_pytorch_tanh`."
            )

        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # self.gate_up_proj = MergedColumnParallelLinear(
        #     hidden_size,
        #     [intermediate_size] * 2,
        #     bias=False,
        #     quant_config=quant_config,
        #     prefix=add_prefix("gate_up_proj", prefix),
        # )

        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )

        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )

        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = ACT2FN[config.hidden_activation]
        if config.activation_sparsity_pattern is not None:
            self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]
        else:
            self.activation_sparsity = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        if self.activation_sparsity > 0.0:
            gate = self._gaussian_topk(gate)
        activations = self.act_fn(gate)
        up, _ = self.up_proj(x)
        x, _ = self.down_proj(activations * up)
        return x

    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
        target_sparsity_tensor = torch.tensor(
            self.activation_sparsity, dtype=torch.float32, device=inputs.device
        )
        # normal_dist and std_multiplier are adapted from jax.scipy.stats.norm.ppf().
        #
        # References:
        #   *   https://docs.jax.dev/en/latest/_autosummary/jax.scipy.stats.norm.ppf.html
        #   *   https://pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal
        #   *   https://pytorch.org/docs/stable/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.icdf
        normal_dist = torch.distributions.normal.Normal(0, 1)
        std_multiplier: torch.Tensor = normal_dist.icdf(target_sparsity_tensor)
        std_multiplier = std_multiplier.type(inputs.dtype)
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = inputs_mean + inputs_std * std_multiplier
        return nn.functional.relu(inputs - cutoff_x)


class Gemma3p5RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        scale_shift: float = 1.0,
        with_scale: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.scale_shift = scale_shift
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._guard_against_excess_precision(x)

        scale = self.weight if self.weight is not None else torch.tensor(1.0)
        if self.scale_shift != 0.0:
            scale += self.scale_shift

        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * scale.float()
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def _guard_against_excess_precision(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(ryanmullins): Implement Torch equivalent to jax.lax.reduce_precision
        return x


class Gemma3p5LaurelBlock(nn.Module):
    """Learned Augmented Residual Layer"""

    def __init__(
        self,
        config: Gemma3p5TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config

        self.linear_left = ColumnParallelLinear(
            input_size=self.config.hidden_size,
            output_size=self.config.laurel_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix(prefix, "linear_left"),
        )

        self.linear_right = RowParallelLinear(
            input_size=self.config.laurel_rank,
            output_size=self.config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix(prefix, "linear_right"),
        )

        self.post_laurel_norm = Gemma3p5RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # laurel_x adapated from two einsums:
        # jax.numpy.einsum("bld,dr->blr", ...)
        # jax.numpy.einsum("blr,rd->bld", ...)

        laurel_x, _ = self.linear_left(x)
        laurel_x, _ = self.linear_right(laurel_x)
        normed_laurel_x = self.post_laurel_norm(laurel_x)
        return x + normed_laurel_x


class Gemma3p5Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Gemma3p5TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        num_layers_that_compute_kv: Optional[int] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.is_sliding = bool((layer_id + 1) % config.sliding_window_pattern)
        self.config = config
        self.layer_idx = layer_id
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True

        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.is_sliding else None

        if num_layers_that_compute_kv is None:
            self.is_kv_shared_layer = False
        else:
            self.is_kv_shared_layer = layer_id >= num_layers_that_compute_kv

        self.qkv_norm = Gemma3p5RMSNorm(
            dim=config.head_dim,
            eps=config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=False,
        )
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0

        hidden_size = config.hidden_size

        head_dim = getattr(
            config, "head_dim", hidden_size // config.num_attention_heads
        )
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim

        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.query_pre_attn_scalar**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        # Determine if layer uses sliding window based on pattern
        self.is_sliding = bool((layer_id + 1) % config.sliding_window_pattern)

        # Initialize the rotary embedding.
        if self.is_sliding:
            # Local attention. Override the values in config.json.
            self.rope_theta = config.rope_local_base_freq
            self.rope_scaling = {"rope_type": "default"}
            self.sliding_window = get_attention_sliding_window_size(config)
        else:
            # Global attention. Use the values in config.json.
            self.rope_theta = config.rope_theta
            self.rope_scaling = config.rope_scaling
            self.sliding_window = None

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=0.0,
            # Module must also define `get_attention_sliding_window_size` to correctly initialize
            # attention backend in `ForwardBatch`.
            sliding_window_size=self.sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]

        qkv, _ = self.qkv_proj(hidden_states)
        # [s, h * head_dim]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # [s, h, head_dim]
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        # -> [b, h, s, head_dim]
        q = q.transpose(0, 1).unsqueeze(0)
        q = self.qkv_norm(q)

        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        # -> [b, h, s, head_dim]
        k = k.transpose(0, 1).unsqueeze(0)
        k = self.qkv_norm(k)

        v = self.qkv_norm(v)

        cos, sin = position_embeddings

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # [b, h, s, head_dim] ->  [b, s, h, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        attn_output = self.attn(
            q=q,
            k=k,
            v=v,
            forward_batch=forward_batch,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output, _ = self.o_proj(attn_output)

        return output


class Gemma3p5DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Gemma3p5TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        num_layers_that_compute_kv: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_id
        self.self_attn = Gemma3p5Attention(
            config=config,
            layer_id=layer_id,
            num_layers_that_compute_kv=num_layers_that_compute_kv,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.mlp = Gemma3p5MLP(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.input_layernorm = Gemma3p5RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Gemma3p5RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma3p5RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma3p5RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.is_sliding = self.self_attn.is_sliding
        self.sliding_window = config.sliding_window

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.act_fn = ACT2FN[config.hidden_activation]

        self.altup = Gemma3p5AltUp(config)
        self.laurel = Gemma3p5LaurelBlock(config)
        self.per_layer_input_gate = nn.Linear(
            self.hidden_size, self.hidden_size_per_layer_input, bias=False
        )
        self.per_layer_projection = nn.Linear(
            self.hidden_size_per_layer_input, self.hidden_size, bias=False
        )
        self.post_per_layer_input_norm = Gemma3p5RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.post_laurel_norm = Gemma3p5RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        per_layer_input: torch.Tensor,
        forward_batch: ForwardBatch,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        predictions = self.altup.predict(hidden_states)
        active_prediction = predictions[self.config.altup_active_idx]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_hidden_states = self.laurel(active_prediction_normed)
        laurel_normed = self.post_laurel_norm(laurel_hidden_states)
        laurel_output = active_prediction_normed + laurel_normed

        # apply global RoPE to non-sliding layer only
        if self.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        attn = self.self_attn(
            hidden_states=active_prediction_normed,
            position_embeddings=position_embeddings,
            forward_batch=forward_batch,
            attention_mask=attention_mask,
        )
        attn = self.post_attention_layernorm(attn)

        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) * torch.rsqrt(torch.tensor(2.0))

        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm
        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)

        first_prediction = corrected_predictions[0]
        if self.config.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction)

        # per_layer_input_gate adapted from jax.numpy.einsum("btd,dp->btp", ...)
        first_prediction = self.per_layer_input_gate(
            first_prediction.type(next(self.per_layer_input_gate.parameters()).dtype)
        )
        first_prediction = self.act_fn(first_prediction)
        first_prediction = torch.multiply(first_prediction, per_layer_input)

        # per_layer_projection adapted from jax.numpy.einsum("btp,pd->btd", ...)
        first_prediction = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)

        for i in range(1, len(corrected_predictions)):
            corrected_predictions[i] += first_prediction

        return corrected_predictions


class Gemma3p5RotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma3p5TextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Gemma3p5TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: Optional[float] = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale


class Gemma3p5AltUp(nn.Module):
    """Alternating Updates (AltUp)

    The AltUp module wraps transformer layers. The `predict` step modifies the
    input to the transformer layer, and the `correct` step propagates the output
    of the transformer layer to the sparsely updated dimensions.

    See more in the research paper:

    https://proceedings.neurips.cc/paper_files/paper/2023/file/f2059277ac6ce66e7e5543001afa8bb5-Paper-Conference.pdf
    """

    def __init__(self, config: Gemma3p5TextConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.correct_output_scale = nn.Parameter(
            torch.zeros(self.config.hidden_size, dtype=torch.float32)
        )
        self.correction_coefs = nn.Linear(
            self.config.altup_num_inputs, self.config.altup_num_inputs, bias=False
        )
        self.prediction_coefs = nn.Linear(
            self.config.altup_num_inputs, self.config.altup_num_inputs**2, bias=False
        )
        self.modality_router = nn.Linear(
            self.config.hidden_size, self.config.altup_num_inputs, bias=False
        )

        self.router_norm = Gemma3p5RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )

    def compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor:
        x_norm: torch.Tensor = self.router_norm(x)
        router_inputs: torch.Tensor = x_norm * self.config.hidden_size**-1.0
        # routed adapted from jax.numpy.einsum("btf,fd->btd", ...)
        routed: torch.Tensor = self.modality_router(router_inputs)
        return torch.tanh(routed).float()

    def predict(self, x: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        modalities = self.compute_router_modalities(
            x[self.config.altup_active_idx]
        ).type(next(self.prediction_coefs.parameters()).dtype)

        if self.config.altup_coef_clip is not None:
            self.prediction_coefs.weight.clamp_(
                -self.config.altup_coef_clip, self.config.altup_coef_clip
            )

        # all_coefs adapted from jax.numpy.einsum("...p,pij->...ij", ...)
        all_coefs: torch.Tensor = self.prediction_coefs(modalities)
        all_coefs = all_coefs.reshape(
            *modalities.shape[:-1],
            self.config.altup_num_inputs,
            self.config.altup_num_inputs,
        )

        outputs: list[torch.Tensor] = [
            torch.zeros_like(x[0])
        ] * self.config.altup_num_inputs
        for i in range(self.config.altup_num_inputs):
            output = outputs[i]

            for j in range(self.config.altup_num_inputs):
                coef = torch.unsqueeze(all_coefs[..., i, j], dim=-1)
                output += coef * x[j]

            x_i = x[i]
            outputs[i] = (x_i + output).type(x_i.dtype)

        return outputs

    def correct(
        self, predictions: Sequence[torch.Tensor], activated: torch.Tensor
    ) -> list[torch.Tensor]:
        modalities = self.compute_router_modalities(activated).type(
            next(self.correction_coefs.parameters()).dtype
        )

        if self.config.altup_coef_clip is not None:
            self.correction_coefs.weight.clamp_(
                -self.config.altup_coef_clip, self.config.altup_coef_clip
            )

        # all_coefs adapted from jax.numpy.einsum("...p,pi->...i", ...)
        all_coefs: torch.Tensor = self.correction_coefs(modalities)
        active_x = predictions[self.config.altup_active_idx]
        innovation = activated - active_x

        corrected = [torch.zeros_like(predictions[0])] * self.config.altup_num_inputs
        for i in range(self.config.altup_num_inputs):
            coef = torch.unsqueeze(all_coefs[..., i] + 1, dim=-1)
            corrected[i] = (predictions[i] + coef * innovation).type(activated.dtype)

        return corrected

    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:
        scale = self.correct_output_scale if self.config.altup_correct_scale else 1.0
        return corrected * scale

    def forward(
        self, x: Sequence[torch.Tensor], activated: torch.Tensor
    ) -> Sequence[torch.Tensor]:
        predictions = self.predict(x)
        corrected = self.correct(predictions=predictions, activated=activated)
        return corrected


class Gemma3p5TextModel(PreTrainedModel):
    def __init__(
        self,
        config: Gemma3p5TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        # KV Cache is partially shared once, as defined by Gemma3p5TextConfig.frac_shared_layers.
        # The following computes the number of initial layers that compute KV before it is shared.
        attention_pattern_length = self.config.sliding_window_pattern
        frac_unshared_layers = 1 - self.config.frac_shared_layers
        num_unshared_layers: int = round(
            self.config.num_hidden_layers * frac_unshared_layers
        )

        if num_unshared_layers >= attention_pattern_length:
            numerator = num_unshared_layers + attention_pattern_length - 1
            num_unshared_layers = (
                attention_pattern_length * numerator // attention_pattern_length
            )
        else:
            logger.warning_once(
                "Not rounding unshared layers. round_up_to_nearest_attention_block is"
                " False or num_unshared_layers is less than attention_pattern_length."
            )
        self.num_layers_that_compute_kv = num_unshared_layers
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3p5 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3p5TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [
                Gemma3p5DecoderLayer(
                    config=config,
                    layer_id=layer_idx,
                    quant_config=quant_config,
                    num_layers_that_compute_kv=self.num_layers_that_compute_kv,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma3p5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3p5RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # TODO (raushan): Fix this after RoPE refactor. For now we hack it by
        # reassigning thetas when we want to create a local RoPE layer. Config
        # defaults should hold values for global RoPE.
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3p5RotaryEmbedding(config=config)

        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input

        self.embed_tokens_per_layer = Gemma3p5TextScaledWordEmbedding(
            config.vocab_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            self.padding_idx,
            embed_scale=config.hidden_size_per_layer_input**0.5,
        )

        self.per_layer_model_projection = nn.Linear(
            self.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )

        self.per_layer_projection_norm = Gemma3p5RMSNorm(
            dim=config.hidden_size_per_layer_input,
            eps=config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )

        # KV Cache is partially shared once, as defined by Gemma3p5TextConfig.frac_shared_layers.
        # The following computes the number of initial layers that compute KV before it is shared.
        attention_pattern_length = self.config.sliding_window_pattern
        frac_unshared_layers = 1 - self.config.frac_shared_layers
        num_unshared_layers: int = round(
            self.config.num_hidden_layers * frac_unshared_layers
        )

        if num_unshared_layers >= attention_pattern_length:
            numerator = num_unshared_layers + attention_pattern_length - 1
            num_unshared_layers = (
                attention_pattern_length * numerator // attention_pattern_length
            )
        else:
            logger.warning_once(
                "Not rounding unshared layers. round_up_to_nearest_attention_block is"
                " False or num_unshared_layers is less than attention_pattern_length."
            )
        self.num_layers_that_compute_kv = num_unshared_layers

        self.altup_projections = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                for _ in range(1, self.config.altup_num_inputs)
            ]
        )

        self.altup_unembed_projections = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                for _ in range(1, self.config.altup_num_inputs)
            ]
        )

    def get_per_layer_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        per_layer_inputs_mask = torch.logical_and(
            input_ids >= 0, input_ids < self.vocab_size
        )
        tokens = torch.where(
            per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids)
        )
        return self.embed_tokens_per_layer(tokens).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_layer_projection = self.per_layer_model_projection(inputs_embeds).reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        if per_layer_projection.shape != per_layer_inputs.shape:
            # per-layer inputs are sometimes padded with zeros, slice the relevant embeddings.
            per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]

        return (per_layer_projection + per_layer_inputs) * torch.rsqrt(
            torch.tensor(2.0)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ) -> LogitsProcessor:

        inputs_embeds = self.embed_tokens(input_ids)
        per_layer_inputs = self.get_per_layer_inputs(input_ids)

        per_layer_inputs = self.project_per_layer_inputs(
            inputs_embeds, per_layer_inputs
        )

        # embed positions
        hidden_states_0 = inputs_embeds

        positions = positions.unsqueeze(0)
        # Initialize RoPE embeddings
        position_embeddings_global = self.rotary_emb(hidden_states_0, positions)
        position_embeddings_local = self.rotary_emb_local(hidden_states_0, positions)

        # Expand hidden_states to support per-layer inputs
        target_magnitude: torch.Tensor = (
            torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
        )
        epsilon_tensor = torch.tensor(torch.finfo().min)

        hidden_states: list[torch.Tensor] = [
            hidden_states_0
        ] * self.config.altup_num_inputs

        for i in range(1, self.config.altup_num_inputs):
            # altup_proj adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_proj: torch.Tensor = self.altup_projections[i - 1](hidden_states[i])
            hidden_states[i] = altup_proj.type(hidden_states_0.dtype)
            new_magnitude = (
                torch.mean(hidden_states[i] ** 2, dim=-1, keepdim=True) ** 0.5
            )
            hidden_states[i] *= target_magnitude / torch.maximum(
                new_magnitude, epsilon_tensor
            )

        # decoder layers
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            per_layer_input = per_layer_inputs[:, decoder_layer.layer_idx, :]

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                per_layer_input=per_layer_input,
                forward_batch=forward_batch,
                position_ids=positions,
            )

            hidden_states = layer_outputs

        # Per-layer inputs to single output
        target_magnitude = (
            torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True) ** 0.5
        )
        for i in range(1, self.config.altup_num_inputs):
            # altup_unembed_projections adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_unemb_proj: torch.Tensor = self.altup_unembed_projections[i - 1](
                hidden_states[i]
            )
            hidden_states[i] = altup_unemb_proj.type(hidden_states_0.dtype)
            new_magnitude = (
                torch.mean(hidden_states[i] ** 2, dim=-1, keepdim=True) ** 0.5
            )
            hidden_states[i] *= target_magnitude / torch.maximum(
                new_magnitude, epsilon_tensor
            )

        hidden_states = torch.mean(torch.stack(hidden_states), dim=0)
        hidden_states = self.norm(hidden_states)

        return hidden_states


class AltUP_Projection(nn.Module):
    def __init__(self):
        super().__init__()
        # FIXME: shape not decided yet
        self.weight = nn.Parameter(torch.zeros(0))

    def forward(self, x):
        return x * self.weight

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class AltUp(nn.Module):
    def __init__(self, d, K, selection_strategy="alternating", transformer_layer=None):
        super().__init__()
        self.d = d
        self.K = K
        self.selection_strategy = selection_strategy
        self.p = nn.Parameter(torch.randn(K, K))
        self.g = nn.Parameter(torch.randn(K))
        self.transformer_layer = transformer_layer
        if transformer_layer is None:
            raise ValueError("A transformer layer must be provided.")

    def predict(self, x_old):
        # x_old shape: [batch_size, seq_len, d * K]
        batch_size, seq_len, dK = x_old.shape
        x_old_reshaped = x_old.view(batch_size, seq_len, self.K, self.d)
        # x_old_reshaped shape: [batch_size, seq_len, K, d]
        p_expanded = self.p.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        x_hat = torch.matmul(p_expanded, x_old_reshaped.transpose(2, 3)).transpose(2, 3)
        # x_hat shape: [batch_size, seq_len, K, d]
        return x_hat.view(batch_size, seq_len, dK)

    def compute(self, x_old, layer_index=None):
        batch_size, seq_len, dK = x_old.shape
        x_old_reshaped = x_old.view(batch_size, seq_len, self.K, self.d)
        if self.selection_strategy == "same":
            j_star = 0
        elif self.selection_strategy == "alternating":
            if layer_index is None:
                raise ValueError(
                    "Layer index must be provided for 'alternating' strategy."
                )
            j_star = layer_index % self.K
        else:
            raise ValueError(f"Invalid selection strategy: {self.selection_strategy}")

        x_old_j_star = x_old_reshaped[:, :, j_star, :]
        x_tilde_j_star = self.transformer_layer(x_old_j_star)
        return x_tilde_j_star, j_star

    def correct(self, x_hat, x_tilde_j_star, j_star):
        batch_size, seq_len, dK = x_hat.shape
        x_hat_reshaped = x_hat.view(batch_size, seq_len, self.K, self.d)
        x_hat_j_star = x_hat_reshaped[:, :, j_star, :]
        delta = x_tilde_j_star - x_hat_j_star
        g_expanded = self.g.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, K, 1]
        delta_expanded = delta.unsqueeze(2)  # [batch_size, seq_len, 1, d]
        correction = g_expanded * delta_expanded
        x_new = x_hat_reshaped + correction
        return x_new.view(batch_size, seq_len, dK)

    def forward(self, x, layer_index):
        x_hat = self.predict(x)
        x_tilde_j_star, j_star = self.compute(x, layer_index)
        x_new = self.correct(x_hat, x_tilde_j_star, j_star)
        return x_new


class Gemma3p5ForCausalLM(PreTrainedModel):
    config_class = Gemma3p5TextConfig

    def __init__(
        self,
        config: Gemma3p5TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config)
        self.model = Gemma3p5TextModel(config)
        self.vocab_size = config.vocab_size
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config)

    def configure_subnetwork(self, flag):
        """Configure the subnetwork for all layers based on the flag."""
        for layer_idx in range(len(self.model.layers)):
            self.model.layers[layer_idx].mlp.configure_subnetwork(flag)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_attention_sliding_window_size(self):
        return get_attention_sliding_window_size(self.config)

    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ) -> LogitsProcessor:

        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, **kwargs
        )

        return self.logits_processor(
            input_ids, hidden_states, self.model.embed_tokens, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            # (".gate_up_proj", ".gate_proj", 0),
            # (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            for param_name, shard_name, shard_id in stacked_params_mapping:
                # if param_name in name:
                # print(f"{param_name} is already in {name}")
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # lm_head is not used in vllm as it is tied with embed_token.
                # To prevent errors, skip loading lm_head.weight.
                if "lm_head.weight" in name:
                    continue
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            logger.warning(
                "Some weights are not initialized from checkpoints: %s", unloaded_params
            )
        return loaded_params


EntryClass = Gemma3p5ForCausalLM
AutoModel.register(Gemma3p5TextConfig, Gemma3p5ForCausalLM, exist_ok=True)
