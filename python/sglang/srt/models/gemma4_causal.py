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

import logging
import re
from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn
from transformers import (
    AutoModel,
    Gemma4TextConfig,
    PretrainedConfig,
    PreTrainedModel,
)

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.layernorm import Gemma4RMSNorm, GemmaRMSNorm, RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.utils import add_prefix, make_layers
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.server_args import get_global_server_args
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.gemma3_causal import Gemma3MLP, Gemma3TextScaledWordEmbedding
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


# Aligned with HF's implementation, using sliding window inclusive with the last token
# SGLang assumes exclusive
def get_attention_sliding_window_size(config):
    return config.sliding_window - 1


Gemma4MLP = Gemma3MLP
Gemma4TextScaledWordEmbedding = Gemma3TextScaledWordEmbedding


class Gemma4PerLayerEmbedding(nn.Module):
    """Per-Layer Embedding (PLE) system for Gemma 4.

    Gemma 4 uses a secondary embedding stream that provides layer-specific
    token embeddings. These are combined with the main hidden states via
    a gating mechanism in each decoder layer.

    The PLE embedding stores embeddings for all layers packed together:
    (vocab_size, hidden_size_per_layer_input * num_hidden_layers)
    """

    def __init__(
        self,
        vocab_size_per_layer_input: int,
        hidden_size_per_layer_input: int,
        hidden_size: int,
        num_hidden_layers: int,
        rms_norm_eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size_per_layer_input
        self.hidden_size_per_layer = hidden_size_per_layer_input
        self.hidden_size = hidden_size
        self.num_layers = num_hidden_layers

        # Packed embedding: (vocab_size, hidden_size_per_layer * num_layers)
        # We store embeddings for ALL layers together
        total_embed_dim = hidden_size_per_layer_input * num_hidden_layers
        self.embed_tokens_per_layer = VocabParallelEmbedding(
            vocab_size_per_layer_input,
            total_embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens_per_layer",
        )

        # Projection from PLE space to hidden space
        # (hidden_size_per_layer * num_layers, hidden_size)
        self.per_layer_model_projection = nn.Linear(
            total_embed_dim,
            hidden_size,
            bias=False,
        )

        # Normalization for PLE output
        # JAX uses scale_plus_one=False for this norm (x * scale, not x * (1+scale))
        self.per_layer_projection_norm = RMSNorm(
            self.hidden_size_per_layer,
            eps=rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute per-layer embeddings and project to hidden size.

        Args:
            input_ids: Token IDs (batch_size, seq_len)

        Returns:
            Per-layer input tensor (batch_size, seq_len, hidden_size)
        """
        # Get packed per-layer embeddings
        per_layer_embeds = self.embed_tokens_per_layer(input_ids)

        # Apply normalization (reshape to apply per-layer, then reshape back)
        # Original shape: (batch, seq, hidden_size_per_layer * num_layers)
        batch_size, seq_len, _ = per_layer_embeds.shape
        per_layer_embeds = per_layer_embeds.view(
            batch_size, seq_len, self.num_layers, self.hidden_size_per_layer
        )
        per_layer_embeds = self.per_layer_projection_norm(per_layer_embeds)
        per_layer_embeds = per_layer_embeds.view(batch_size, seq_len, -1)

        # Project to hidden size
        per_layer_input = self.per_layer_model_projection(per_layer_embeds)
        return per_layer_input


class Gemma4Router(nn.Module):
    """Router for Gemma4 MoE that preprocesses input before projection.

    Applies RMSNorm (no learned weight), root_size scaling
    (hidden_size^{-0.5}), then a learned per-dimension scale before
    projecting to expert logits.

    This preprocessing is applied ONLY to the router's input, not to
    the expert MLPs' input.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        # RMSNorm without learned weight — pure normalization only
        self.norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, with_scale=False)
        # Per-dimension learned scale, applied after norm + root_size
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        # Constant 1/sqrt(hidden_size) scaling factor
        self.register_buffer(
            "root_size",
            torch.tensor(self.hidden_size**-0.5),
            persistent=False,
        )
        # Project to expert logits; replicated across TP for consistent routing
        self.proj = ReplicatedLinear(
            self.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("proj", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw router logits [T, E]."""
        x = self.norm(x)
        x = x * self.root_size.to(x.dtype)
        x = x * self.scale.to(x.dtype)
        router_logits, _ = self.proj(x)
        return router_logits


class Gemma4MoE(nn.Module):
    """Mixture of Experts for Gemma4.

    Wraps MoE implementation with custom routing. The router projection is
    external (Gemma4Router) — this class only handles expert dispatch.

    Gemma4 routing: softmax over ALL experts → top-k → renormalize.
    per_expert_scale is folded into routing weights for mathematical
    correctness with MoE's fused kernel.
    """

    def __init__(
        self,
        hidden_size: int,
        layer_id: int,
        config: Gemma4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.num_experts = config.num_experts
        self.tp_size = get_tensor_model_parallel_world_size()

        # Per-expert output scale folded into routing weights so that
        # MoE's fused kernel computes: Σ_e (expert_e * w_e * scale_e)
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

        # Gemma4 routing: softmax over ALL experts → top-k → renormalize.
        per_expert_scale = self.per_expert_scale

        def routing_function(
            hidden_states: torch.Tensor,
            gating_output: torch.Tensor,
            topk: int,
            renormalize: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            _, topk_ids = torch.topk(gating_output, k=topk, dim=-1)
            router_probabilities = torch.nn.functional.softmax(gating_output, dim=-1)
            indicator = torch.nn.functional.one_hot(
                topk_ids, num_classes=gating_output.size(-1)
            ).sum(dim=-2)
            gate_weights = indicator * router_probabilities
            renorm_factor = torch.sum(gate_weights, dim=-1, keepdim=True)
            renorm_factor = torch.where(renorm_factor > 0.0, renorm_factor, 1.0)
            dispatch_weights = gate_weights / renorm_factor

            topk_weights = dispatch_weights.gather(1, topk_ids)

            # Fold per_expert_scale into routing weights
            expert_scales = per_expert_scale[topk_ids].to(topk_weights.dtype)
            topk_weights = topk_weights * expert_scales

            return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

        self.topk = TopK(
            top_k=config.top_k_experts,
            layer_id=layer_id,
            custom_routing_function=routing_function,
        )

        experts_type = get_moe_impl_class(quant_config)

        self.experts = experts_type(
            num_experts=config.num_experts + get_global_server_args().ep_num_redundant_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.expert_intermediate_size,
            layer_id=layer_id,
            top_k=config.top_k_experts,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            activation="gelu",
            reduce_results=True,
        )

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        topk_output = self.topk(hidden_states, router_logits)
        hidden_states = self.experts(hidden_states, topk_output)
        return hidden_states.view(num_tokens, hidden_dim)

class Gemma4Attention(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: Gemma4TextConfig,
        head_dim: int,
        max_position_embeddings: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.layer_id = layer_id
        self.config = config
        tp_size = get_tensor_model_parallel_world_size()

        layer_type = config.layer_types[layer_id]
        self.sliding_window = config.sliding_window if layer_type == "sliding_attention" else None

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        if layer_type == "sliding_attention":
            self.total_num_kv_heads = getattr(
                config, "swa_num_key_value_heads", config.num_key_value_heads
            )
        else:
            self.total_num_kv_heads = config.num_key_value_heads

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        hidden_size = config.hidden_size
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

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

        self.q_norm = Gemma4RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.k_norm = Gemma4RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.v_norm = Gemma4RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=False
        )

        if layer_type in config.rope_parameters:
            rope_parameters = dict(config.rope_parameters[layer_type])
            if layer_type == "full_attention":
                global_prf = getattr(config, "global_partial_rotary_factor", 0.25)
                rope_parameters["partial_rotary_factor"] = global_prf
        else:
            rope_parameters = dict(
                rope_type="default",
                rope_theta=getattr(config, "rope_theta", 10000.0),
            )

        # KV sharing logic
        num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)
        first_kv_shared_layer_idx = (
            config.num_hidden_layers - num_kv_shared_layers
        )
        self.is_kv_shared_layer = layer_id >= first_kv_shared_layer_idx and num_kv_shared_layers > 0

        self.kv_shared_layer_index = None
        if num_kv_shared_layers > 0 and self.layer_id >= first_kv_shared_layer_idx:
            prev_layers = config.layer_types[:first_kv_shared_layer_idx]
            current_layer_type = config.layer_types[self.layer_id]
            self.kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(
                current_layer_type
            )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_parameters.get("rope_theta", 10000.0),
            rope_scaling={"rope_type": rope_parameters.get("rope_type", "default")},
            partial_rotary_factor=rope_parameters.get("partial_rotary_factor", 1.0),
            is_neox_style=True,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            1,  # scaling factor
            num_kv_heads=self.num_kv_heads,
            layer_id=(
                self.kv_shared_layer_index if self.is_kv_shared_layer else self.layer_id
            ),
            logit_cap=0.0,
            sliding_window_size=self.sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        # Check if we should use shared KV cache
        if self.is_kv_shared_layer and self.kv_shared_layer_index is not None:
            # For KV shared layers, we skip K/V computation and normalization
            # The RadixAttention will handle retrieving shared KV from cache
            k = None
            v = None
        else:
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k = self.k_norm(k)

            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = self.v_norm(v)

        # Apply rotary embedding
        if k is not None:
            k = k.flatten(-2, -1)
            q, k = self.rotary_emb(positions, q, k)
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        else:
            # For shared KV layers, create a dummy key for rotary embedding and discard it
            dummy_k = torch.zeros_like(
                q[:, : self.kv_size]
            )  # Create dummy key with same shape as needed
            q, _ = self.rotary_emb(positions, q, dummy_k)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        attn_output = self.attn(
            q,
            k,
            v,
            forward_batch=forward_batch,
            save_kv_cache=not self.is_kv_shared_layer,
        )
        if attn_output.dim() == 3:
            attn_output = attn_output.flatten(-2, -1)
        output, _ = self.o_proj(attn_output)

        return output


class Gemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", None
        ) or 0

        self.layer_id = layer_id

        # Gemma 4 uses different head dimensions for sliding vs full attention
        layer_type = config.layer_types[layer_id]
        self.is_full_attention = layer_type == "full_attention"
        if self.is_full_attention:
            head_dim = config.head_dim  # following sglang naming
        else:
            head_dim = getattr(config, "swa_head_dim", config.head_dim)

        self.self_attn = Gemma4Attention(
            layer_id=layer_id,
            config=config,
            max_position_embeddings=config.max_position_embeddings,
            head_dim=head_dim,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(
            config, "num_kv_shared_layers", 0
        )
        is_kv_shared_layer = self.layer_id >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = (
            getattr(config, "use_double_wide_mlp", False) and is_kv_shared_layer
        )
        layer_intermediate_size = config.intermediate_size * (
            2 if use_double_wide_mlp else 1
        )

        self.mlp = Gemma4MLP(
            hidden_size=self.hidden_size,
            intermediate_size=layer_intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Per-Layer Embedding (PLE) components — present in each decoder layer
        if self.hidden_size_per_layer_input > 0:
            # Gate: projects hidden_states → per-layer dim for gating
            self.per_layer_input_gate = ReplicatedLinear(
                self.hidden_size,
                self.hidden_size_per_layer_input,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("per_layer_input_gate", prefix),
            )
            # Projection: projects gated per-layer input back → hidden size
            self.per_layer_projection = ReplicatedLinear(
                self.hidden_size_per_layer_input,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("per_layer_projection", prefix),
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

        # Parallel MoE
        self.enable_moe_block = getattr(config, "enable_moe_block", False) or getattr(
            config, "use_second_mlp_block", False
        )
        if self.enable_moe_block:
            self.router = Gemma4Router(
                config,
                quant_config=quant_config,
                prefix=add_prefix("router", prefix),
            )
            self.moe = Gemma4MoE(
                hidden_size=self.hidden_size,
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("moe", prefix),
            )

            self.post_feedforward_layernorm_1 = GemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = GemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = GemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.router = None
            self.moe = None
            self.post_feedforward_layernorm_1 = None
            self.post_feedforward_layernorm_2 = None
            self.pre_feedforward_layernorm_2 = None

        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)
        self.prefix = prefix

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        # Gemma4 residual pattern following JAX implementation:
        # 1. input_norm(x) -> attn -> post_attn_norm -> ADD residual
        # 2. pre_ff_norm -> mlp -> post_ff_norm -> ADD residual
        residual = hidden_states

        # Apply input layernorm
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states

        if self.enable_moe_block:
            # Dense MLP branch
            hidden_states_1 = self.pre_feedforward_layernorm(hidden_states)
            hidden_states_1 = self.mlp(hidden_states_1)
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states_1)

            # MoE branch: router sees raw hidden_states (applies its own
            # norm + scale internally); experts see separately normed input
            router_logits = self.router(hidden_states)
            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states)
            hidden_states_2 = self.moe(hidden_states_2, router_logits)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            # Combine branches
            hidden_states = hidden_states_1 + hidden_states_2
        else:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        if (
            per_layer_input is not None
            and self.per_layer_input_gate is not None
            and self.per_layer_projection is not None
            and self.post_per_layer_input_norm is not None
        ):
            gate, _ = self.per_layer_input_gate(hidden_states)
            # PLE uses gelu activation for the gate
            # Note: GeluAndMul expects concatenated [gate, up] but here we
            # only have a single projection. Use F.gelu directly.
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            gated_per_layer = gate * per_layer_input
            per_layer_contribution, _ = self.per_layer_projection(gated_per_layer)
            per_layer_contribution = self.post_per_layer_input_norm(
                per_layer_contribution
            )
            hidden_states = hidden_states + per_layer_contribution
        
        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None


class Gemma4TextModel(PreTrainedModel):
    def __init__(
        self,
        config: Gemma4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.padding_idx = getattr(config, "pad_token_id", None)

        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size**0.5,  # embeded normalizer
        )

        # Per-layer input embeddings
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", None
        ) or 0
        self.vocab_size_per_layer_input = getattr(
            config, "vocab_size_per_layer_input", None
        ) or config.vocab_size

        if self.hidden_size_per_layer_input and self.hidden_size_per_layer_input > 0:
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                self.vocab_size_per_layer_input,
                config.num_hidden_layers * self.hidden_size_per_layer_input,
                self.padding_idx,
                embed_scale=self.hidden_size_per_layer_input**0.5,
            )
            
            self.per_layer_model_projection = ReplicatedLinear(
                self.hidden_size,
                config.num_hidden_layers * self.hidden_size_per_layer_input,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("per_layer_model_projection", prefix),
            )

            self.per_layer_projection_norm = RMSNorm(
                self.hidden_size_per_layer_input,
                config.rms_norm_eps,
            )
            self.per_layer_input_scale = torch.rsqrt(torch.tensor(2.0))
            self.per_layer_projection_scale = torch.tensor(
                config.hidden_size**-0.5,
            )
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None
            self.per_layer_input_scale = None
            self.per_layer_projection_scale = None

        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Gemma4DecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )

        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def get_per_layer_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        if self.embed_tokens_per_layer is None:
            return None

        # Handle out-of-vocab tokens for PLE (vocab_size_per_layer_input may
        # be smaller than the main vocab_size). Following Gemma3n pattern.
        per_layer_inputs_mask = torch.logical_and(
            input_ids >= 0,
            input_ids < self.vocab_size_per_layer_input,
        )
        per_layer_inputs_tokens = torch.where(
            per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids)
        )

        # Get packed per-layer embeddings: (num_tokens, total_ple_dim)
        per_layer_embeds = self.embed_tokens_per_layer(per_layer_inputs_tokens)

        # Apply embed_scale (sqrt of per-layer hidden dim)
        # Alreayd done in embedding layer
        # per_layer_embeds = per_layer_embeds * self.embed_scale_per_layer

        # Reshape to (num_tokens, num_layers, hidden_size_per_layer_input)
        per_layer_embeds = per_layer_embeds.reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        return per_layer_embeds

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project inputs_embeds and combine with per_layer_inputs.

        Following HF/Gemma3n reference:
        1. Project inputs_embeds: hidden_size → total_ple_dim
        2. Scale by hidden_size^{-0.5} (Gemma4ScaledLinear w_scale)
        3. Reshape to (num_tokens, num_layers, per_layer_dim)
        4. Normalize with per_layer_projection_norm
        5. Combine: (projection + per_layer_inputs) * 1/sqrt(2)
        """
        if self.per_layer_model_projection is None:
            return None

        # Project from hidden_size to total_ple_dim
        per_layer_projection, _ = self.per_layer_model_projection(inputs_embeds)

        # Apply w_scale (HF: Gemma4ScaledLinear with w_scale=hidden_size^{-0.5})
        per_layer_projection = per_layer_projection * self.per_layer_projection_scale

        # Reshape to (num_tokens, num_layers, hidden_size_per_layer_input)
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

        # Normalize
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        # Combine: (projection + per_layer_inputs) * scale
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (input_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if input_ids is not None:
            input_embeds = self.embed_tokens(input_ids)
            per_layer_inputs = self.get_per_layer_inputs(input_ids)
        per_layer_inputs = self.project_per_layer_inputs(input_embeds, per_layer_inputs)

        hidden_states = input_embeds

        for layer_idx, layer in enumerate(self.layers):
            if per_layer_inputs is not None:
                per_layer_input = per_layer_inputs[:, layer_idx, :]
            else:
                per_layer_input = None
            layer_outputs = layer(
                positions=positions,
                hidden_states=hidden_states,
                per_layer_input=per_layer_input,
                forward_batch=forward_batch,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            residual = layer_outputs[1] if len(layer_outputs) > 1 else None

        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Gemma4ForCausalLM(PreTrainedModel):
    config_class = Gemma4TextConfig
    base_model_prefix = "language_model"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # Gemma does not apply LoRA to the embedding layer.
    embedding_modules = {}
    embedding_padding_modules = []
    supports_lora = False

    def __init__(
        self,
        config: Gemma4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config
        self.model = Gemma4TextModel(
            config=config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.logits_processor = LogitsProcessor(config)

        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.post_init()

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
        per_layer_inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LogitsProcessor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            per_layer_inputs,
            **kwargs,
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def _get_k_eq_v_layers(self) -> set:
        """Return set of layer indices where attention_k_eq_v applies (full-attention layers)."""
        if not getattr(self.config, "attention_k_eq_v", False):
            return set()
        return {
            i
            for i, lt in enumerate(self.config.layer_types)
            if lt == "full_attention"
        }

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping_gemma4(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
        )
        num_experts = self.config.num_experts

        k_eq_v_layers = self._get_k_eq_v_layers()

        params_dict = dict(self.named_parameters())
        params_dict.update(dict(self.named_buffers()))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            name = name.replace("model.language_model.", "model.")

            if (
                ".moe." in name
                and "experts" not in name
                and "per_expert_scale" not in name
            ):
                name = name.replace(".moe.", ".moe.experts.")

            # attention_k_eq_v: full-attention layers have no v_proj in the
            # checkpoint (K and V share weights).  When we see a k_proj weight
            # for one of these layers, load it into both the "k" and "v" shards
            # of the fused QKV so the forward produces v_raw == k_raw.
            should_dup_k_to_v = (
                ".k_proj." in name
                and k_eq_v_layers
                and (m := re.search(r"layers\.(\d+)\.", name)) is not None
                and int(m.group(1)) in k_eq_v_layers
            )

            # Try stacked (fused) params first
            orig_name = name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                name = orig_name
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                if should_dup_k_to_v:
                    weight_loader(param, loaded_weight, "v")
                break
            else:
                for param_name, weight_name, shard_id in expert_params_mapping:
                    name = orig_name
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    for i in range(num_experts):
                        weight_loader(param, loaded_weight[i].T, name, shard_id, i)
                    break
                else:
                    name = orig_name
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    if name not in params_dict:
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


EntryClass = Gemma4ForCausalLM
