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
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import Gemma4VisionConfig

from sglang.srt.layers.attention.vision import QKV_BACKEND_IMPL, VisionAttention
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import Gemma4RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import add_prefix, get_device_capability, is_cuda

# ---------------------------------------------------------------------------
# 2-D Multidimensional RoPE (matches HF Gemma4RotaryEmbedding for vision)
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class Gemma4VisionRotaryEmbedding(nn.Module):
    """Compute 2-D multidimensional RoPE cos/sin for patch positions."""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        rope_params = config.rope_parameters.get("full_attention", {})
        self.rope_theta: float = rope_params.get("rope_theta", 100.0)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, patch_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq, hidden] – only used for device/dtype.
            patch_positions: [batch, num_patches, 2] – (x, y) coordinates.
        Returns:
            (cos, sin) each of shape [batch, num_patches, head_dim].
        """
        ndim = patch_positions.shape[-1]  # 2
        head_dim_per_dim = self.head_dim // ndim

        all_embs = []
        for d in range(ndim):
            dim_inv_freq = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(0, head_dim_per_dim, 2, device=x.device, dtype=torch.float)
                    / head_dim_per_dim
                )
            )
            dim_inv_freq_expanded = dim_inv_freq[None, :, None].expand(
                patch_positions.shape[0], -1, 1
            )
            dim_positions = patch_positions[:, :, d].float()
            dim_positions_expanded = dim_positions[:, None, :]

            dim_freqs = (dim_inv_freq_expanded @ dim_positions_expanded).transpose(1, 2)
            dim_emb = torch.cat((dim_freqs, dim_freqs), dim=-1)
            all_embs.append(dim_emb)

        emb = torch.cat(all_embs, dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin


def _apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply 2-D RoPE to x of shape [batch*seq, heads, head_dim].

    cos/sin have shape [batch, seq, head_dim]. We split along head_dim into
    ndim=2 parts and apply standard rotary to each independently.
    """
    ndim = 2
    chunk_size = x.shape[-1] // ndim
    x_parts = x.split(chunk_size, dim=-1)
    cos_parts = cos.split(chunk_size, dim=-1)
    sin_parts = sin.split(chunk_size, dim=-1)
    y_parts = [_apply_rotary(x_parts[k], cos_parts[k], sin_parts[k]) for k in range(ndim)]
    return torch.cat(y_parts, dim=-1)


# ---------------------------------------------------------------------------
# Vision Attention (TP-sharded via QKVParallelLinear + RowParallelLinear)
# ---------------------------------------------------------------------------


class Gemma4VisionAttention(nn.Module):
    """Multi-head attention for the Gemma 4 vision encoder.

    Uses SGLang's QKVParallelLinear and RowParallelLinear for tensor-parallel
    sharding, Gemma4RMSNorm for per-head QK/V normalization, and the same
    multimodal attention backends as VisionAttention.
    """

    def __init__(
        self,
        config: Gemma4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        tp_size = get_attention_tp_size()
        self.num_heads_per_partition = self.num_heads // tp_size
        self.num_kv_heads_per_partition = self.num_kv_heads // tp_size

        self.q_size = self.num_heads_per_partition * self.head_dim
        self.kv_size = self.num_kv_heads_per_partition * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=False
        )

        backend = self._select_backend()
        self.qkv_backend = QKV_BACKEND_IMPL[backend](
            head_dim=self.head_dim,
            num_heads=self.num_heads_per_partition,
            num_kv_heads=self.num_kv_heads_per_partition,
            dropout=0.0,
            flatten_batch=True,
            softmax_in_single_precision=False,
        )

    @staticmethod
    def _select_backend() -> str:
        """Mirror VisionAttention._determine_attention_backend for consistency."""
        from sglang.srt.server_args import get_global_server_args

        override = get_global_server_args().mm_attention_backend
        if override is not None:
            return override
        if is_cuda():
            major, _ = get_device_capability()
            if major == 9:
                from sglang.srt.utils import is_blackwell_supported

                if is_blackwell_supported():
                    return "triton_attn"
                return "fa3"
            return "triton_attn"
        return "sdpa"

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden_size]
            cos, sin: [batch, seq, head_dim] from Gemma4VisionRotaryEmbedding
            attention_mask: [batch, seq] — True = valid, False = padding
        """
        bsz, seq_len, _ = hidden_states.shape

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.reshape(bsz * seq_len, self.num_heads_per_partition, self.head_dim)
        k = k.reshape(bsz * seq_len, self.num_kv_heads_per_partition, self.head_dim)
        v = v.reshape(bsz * seq_len, self.num_kv_heads_per_partition, self.head_dim)

        # Per-head QK norm
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q.shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k.shape)
        v = self.v_norm(v.reshape(-1, self.head_dim)).reshape(v.shape)

        # 2-D RoPE: cos/sin are [batch, seq, head_dim]; broadcast to [batch*seq, 1, head_dim]
        cos_flat = cos.reshape(bsz * seq_len, 1, self.head_dim)
        sin_flat = sin.reshape(bsz * seq_len, 1, self.head_dim)
        q = _apply_multidimensional_rope(q, cos_flat, sin_flat)
        k = _apply_multidimensional_rope(k, cos_flat, sin_flat)

        # Build 4-D attention mask for backends that expect it
        if attention_mask is not None:
            attn_mask_4d = (
                attention_mask.unsqueeze(-1) * attention_mask.unsqueeze(1)
            ).unsqueeze(1)
        else:
            attn_mask_4d = None

        output = self.qkv_backend.forward(
            q=q, k=k, v=v,
            cu_seqlens=None,
            bsz=bsz, seq_len=seq_len,
            attention_mask=attn_mask_4d,
        )

        output = rearrange(output, "(b s) h d -> b s (h d)", b=bsz)
        output, _ = self.o_proj(output)
        return output


# ---------------------------------------------------------------------------
# Vision MLP (GeGLU, TP-sharded)
# ---------------------------------------------------------------------------


class Gemma4VisionMLP(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.intermediate_size, self.intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        from sglang.srt.layers.activation import SiluAndMul

        self.act_fn = SiluAndMul()  # GeGLU: GELU variant handled by weight init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# Encoder Layer
# ---------------------------------------------------------------------------


class Gemma4VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = Gemma4VisionAttention(
            config, quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Gemma4VisionMLP(
            config, quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        eps = config.rms_norm_eps
        hs = config.hidden_size
        self.input_layernorm = Gemma4RMSNorm(hs, eps=eps)
        self.post_attention_layernorm = Gemma4RMSNorm(hs, eps=eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(hs, eps=eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(hs, eps=eps)

        self.register_buffer("layer_scalar", torch.ones(()))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


# ---------------------------------------------------------------------------
# Vision Transformer (stack of encoder layers + RoPE)
# ---------------------------------------------------------------------------


class Gemma4VisionTransformer(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList([
            Gemma4VisionEncoderLayer(
                config, layer_idx=i, quant_config=quant_config,
                prefix=add_prefix(f"layers.{i}", prefix),
            )
            for i in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        patch_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs_embeds: [batch, seq, hidden_size]
            attention_mask: [batch, seq] — True = valid token
            patch_positions: [batch, seq, 2]
        Returns:
            last_hidden_state: [batch, seq, hidden_size]
        """
        cos, sin = self.rotary_emb(inputs_embeds, patch_positions)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attention_mask)
        return hidden_states


# ---------------------------------------------------------------------------
# Patch Embedder
# ---------------------------------------------------------------------------


class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.position_embedding_size = config.position_embedding_size

        self.input_proj = nn.Linear(3 * self.patch_size ** 2, self.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, self.hidden_size)
        )

    def _position_embeddings(
        self, patch_positions: torch.Tensor, padding_positions: torch.Tensor
    ) -> torch.Tensor:
        one_hot = F.one_hot(patch_positions, num_classes=self.position_embedding_size)
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
        position_embeddings = one_hot @ self.position_embedding_table
        position_embeddings = position_embeddings.sum(dim=1)
        position_embeddings = torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)
        return position_embeddings

    def _patchify(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        patchified_shape = (batch_size, num_channels, patch_height, self.patch_size, patch_width, self.patch_size)
        consolidated_shape = (batch_size, patch_height * patch_width, num_channels * self.patch_size ** 2)
        patches = pixel_values.reshape(patchified_shape).permute(0, 2, 4, 3, 5, 1).reshape(consolidated_shape)
        patches = 2 * (patches - 0.5)
        return self.input_proj(patches.to(self.input_proj.weight.dtype))

    def forward(
        self, pixel_values: torch.Tensor, patch_positions: torch.Tensor, padding_positions: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self._patchify(pixel_values)
        position_embeddings = self._position_embeddings(patch_positions, padding_positions)
        return hidden_states + position_embeddings


# ---------------------------------------------------------------------------
# Pooler
# ---------------------------------------------------------------------------


class Gemma4VisionPooler(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.default_output_length = config.default_output_length
        self.root_hidden_size = self.hidden_size ** 0.5

    def _avg_pool_by_positions(
        self, x: torch.Tensor, patch_positions: torch.Tensor, length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = x.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k ** 2
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {x.shape} to {length}: {k=}^2 times {length=} must be {input_seq_len}."
            )
        clamped_positions = patch_positions.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
        output = weights.transpose(1, 2).to(x.dtype) @ x
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output, mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        patch_positions: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (pooled_hidden_states, mask) where mask is True for valid tokens.
        """
        length = self.default_output_length
        if isinstance(length, (list, tuple)):
            length = length[0]
        if hidden_states.shape[1] == length:
            mask = padding_positions
        else:
            hidden_states, mask = self._avg_pool_by_positions(
                hidden_states, patch_positions, length
            )
        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, mask


# ---------------------------------------------------------------------------
# Top-level Vision Encoder (patch_embedder → transformer → pooler)
# ---------------------------------------------------------------------------


class Gemma4VisionEncoder(nn.Module):
    """Drop-in replacement for HF ``Gemma4VisionEncoder`` with TP support."""

    def __init__(
        self,
        config: Gemma4VisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.pooling_kernel_size = config.pooling_kernel_size
        self.default_output_length = config.default_output_length
        self.max_patches = self.default_output_length * self.pooling_kernel_size ** 2

        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionTransformer(
            config, quant_config=quant_config,
            prefix=add_prefix("encoder", prefix),
        )
        self.pooler = Gemma4VisionPooler(config)

    @property
    def device(self) -> torch.device:
        return self.patch_embedder.input_proj.weight.device

    def _num_real_patches(self, pixel_values: torch.Tensor) -> int:
        _, _, height, width = pixel_values.shape
        return (height // self.patch_size) * (width // self.patch_size)

    def _patch_positions(
        self, pixel_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = pixel_values.shape
        device = pixel_values.device
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        num_patches = patch_height * patch_width
        num_padding = self.max_patches - num_patches

        patch_grid = torch.meshgrid(
            torch.arange(patch_width, device=device),
            torch.arange(patch_height, device=device),
            indexing="xy",
        )
        stacked_grid = torch.stack(patch_grid, dim=-1)
        real_positions = stacked_grid.reshape(num_patches, 2).unsqueeze(0).repeat(batch_size, 1, 1)

        if num_padding > 0:
            pad_positions = torch.full(
                (batch_size, num_padding, 2), -1, device=device, dtype=torch.long
            )
            patch_positions = torch.cat([real_positions, pad_positions], dim=1)
        else:
            patch_positions = real_positions

        padding_positions = torch.zeros(batch_size, self.max_patches, device=device, dtype=torch.bool)
        if num_padding > 0:
            padding_positions[:, num_patches:] = True

        return patch_positions.long(), padding_positions

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode pixel_values into soft tokens.

        Args:
            pixel_values: [batch, channels, height, width]

        Returns:
            (hidden_states, pooler_mask) — hidden_states [batch, output_len, hidden],
            pooler_mask [batch, output_len] True = valid.
        """
        patch_positions, padding_positions = self._patch_positions(pixel_values)

        inputs_embeds = self.patch_embedder(
            pixel_values,
            patch_positions[:, : self._num_real_patches(pixel_values)],
            padding_positions[:, : self._num_real_patches(pixel_values)],
        )

        num_real = inputs_embeds.shape[1]
        num_padding = self.max_patches - num_real
        if num_padding > 0:
            pad_embeds = torch.zeros(
                inputs_embeds.shape[0], num_padding, inputs_embeds.shape[2],
                device=inputs_embeds.device, dtype=inputs_embeds.dtype,
            )
            inputs_embeds = torch.cat([inputs_embeds, pad_embeds], dim=1)

        last_hidden = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,
            patch_positions=patch_positions,
        )

        pooled, pooler_mask = self.pooler(last_hidden, patch_positions, padding_positions)
        return pooled, pooler_mask
