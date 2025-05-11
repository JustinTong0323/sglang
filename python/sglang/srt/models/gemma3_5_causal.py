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
from typing import Iterable, List, Optional, Set, Tuple

import einops
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, PreTrainedModel

from sglang.srt.configs import Gemma3p5Config, Gemma3p5TextConfig
from sglang.srt.layers.activation import GeluAndMul
from sglang.srt.layers.layernorm import Gemma3RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.gemma3_causal import (
    Gemma3Attention,
    Gemma3RotaryEmbedding,
    Gemma3TextScaledWordEmbedding,
)
from sglang.srt.utils import add_prefix, make_layers


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
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma3p5 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_activation` to "
                "`gelu_pytorch_tanh`."
            )
        self.act_fn = GeluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Gemma3p5LaurelLR(nn.Module):
    # LaurelLR for SGLang
    def __init__(
        self,
        hidden_size: int,
        laurel_rank: int,
        quant_config: Optional["QuantizationConfig"] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.laurel_rank = laurel_rank

        self.linear_left = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=laurel_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix(prefix, "linear_left"),
        )

        self.linear_right = RowParallelLinear(
            input_size=laurel_rank,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix(prefix, "linear_right"),
        )

    def forward(self, x_i: torch.Tensor) -> torch.Tensor:
        temp, _ = self.linear_left(x_i)
        output, _ = self.linear_right(temp)

        return output


class Gemma3p5DecoderLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: Gemma3p5TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma3Attention(
            layer_id=layer_id,
            config=config,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.hidden_size = config.hidden_size
        self.mlp = Gemma3p5MatFormerMLP(
            config=config,
            scale_factors=config.matformer_scale_factors,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.laurel_lr = Gemma3p5LaurelLR(config.hidden_size, config.laurel_rank)

        self.is_sliding = self.self_attn.is_sliding
        self.layer_id = layer_id

        self.altup = Gemma3p5AltUP(config=config)
        self.config = config

    def forward(
        self,
        positions: torch.Tensor,
        blocks: List[torch.Tensor],
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        # TODO: add PerLayerEmbedding and ....

        # 1. predict the updated representation for each blocks
        predicted_blocks = [self.altup.predict(block) for block in blocks]

        # 2. select a block and perform the regular forward
        # TODO: use `alternating` method described in the paper
        # assume altup_num_inputs is the number of sub-blocks: K
        selected_index = self.layer_id % self.config.altup_num_inputs
        hidden_states = blocks[selected_index]
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # apply global RoPE to non-sliding layer only
        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            forward_batch=forward_batch,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states + self.laurel_lr(residual)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states + self.laurel_lr(residual)

        # correct the predict of each blocks with computation result
        # propagates decoder output to predictions
        new_blocks = []
        selected_block_prediction_result = predicted_blocks[selected_index]
        for predicted in predicted_blocks:
            new_block = predicted + self.altup.correct(
                hidden_states, selected_block_prediction_result
            )
            new_block = self.altup.scale_corrected_output(new_block)
            # TODO:  called just before the block exits
            # where are blocks being merged ?
            new_blocks += [new_block]

        return new_blocks


# TODO: not tested
class Gemma3p5AltUP(nn.Module):
    def __init__(self, config: Gemma3p5TextConfig):
        super().__init__()
        self.d = config.hidden_size_per_layer_input
        self.K = config.altup_num_inputs
        self.active_idx = config.altup_active_idx
        self.coef_clip = config.altup_coef_clip
        self.correct_scale = config.altup_correct_scale
        self.lr_multiplier = config.altup_lr_multiplier

        self.prediction_coefs = nn.Parameter(
            torch.randn(self.K, self.K, self.d, self.d)
        )
        self.modality_router = nn.Linear(self.d * self.K, self.d)
        self.router_norm = nn.LayerNorm(self.d)
        self.correction_coefs = nn.Parameter(torch.randn(self.K, self.d))
        self.correct_output_scale = (
            nn.Parameter(torch.ones(1)) if self.correct_scale else None
        )

    def predict(self, x_old):
        """
        Predict the updated representation for each sub-blocks
        """
        batch_size, seq_len, dK = x_old.shape
        x_old_reshaped = x_old.view(batch_size, seq_len, self.K, self.d)
        # x_old_reshaped shape: [batch_size, seq_len, K, d]

        p_expanded = self.prediction_coefs.unsqueeze(0).unsqueeze(
            0
        )  # [1, 1, K, K, d, d]
        x_old_expanded = x_old_reshaped.unsqueeze(2).unsqueeze(
            -1
        )  # [batch_size, seq_len, K, 1, d, 1]

        # Perform prediction using matrix multiplication
        x_hat = torch.matmul(p_expanded, x_old_expanded).squeeze(
            -1
        )  # [batch_size, seq_len, K, K, d]
        x_hat = x_hat.sum(dim=3)  # [batch_size, seq_len, K, d]

        return x_hat.view(batch_size, seq_len, dK)

    def correct(self, x_hat, x_tilde):
        """
        Correct the passed prediction of sub-blocks with predicted results
        Args:
            x_hat: computation result for selected block
            x_tilde: predicted result
        """
        # TODO: use correct_output_scale & correction_coefs
        batch_size, seq_len, dK = x_hat.shape
        x_hat_reshaped = x_hat.view(batch_size, seq_len, self.K, self.d)

        # Correction
        delta = x_tilde.unsqueeze(2) - x_hat_reshaped[
            :, :, self.active_idx, :
        ].unsqueeze(
            2
        )  # [batch_size, seq_len, 1, d]
        g_expanded = (
            self.correction_coefs.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        )  # [1, 1, K, 1]
        correction = g_expanded * delta  # [batch_size, seq_len, K, d]
        x_new_reshaped = x_hat_reshaped + correction

        x_new = x_new_reshaped.view(batch_size, seq_len, dK)

        # Apply output scaling if enabled
        if self.correct_output_scale is not None:
            x_new = x_new * self.correct_output_scale

        return x_new

    def scale_corrected_output(self, corrected_output):
        if self.correct_output_scale:
            return corrected_output * self.correct_output_scale
        return corrected_output


class Gemma3p5PerLayerEmbedding(nn.Module):
    pass


class Gemma3p5TextModel(PreTrainedModel):
    def __init__(
        self,
        config: Gemma3p5TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size**0.5,
        )

        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # when we want to create a local RoPE layer. Config defaults should hold values for global RoPE
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config)

        self.altup_projections = nn.ModuleList(
            [
                AltUP_Projection()
                # TODO: check the layer count
                for _ in range(3)
            ]
        )

        self.altup_unembed_projections = nn.ModuleList(
            [
                AltUP_Projection()
                # TODO: check the layer count
                for _ in range(3)
            ]
        )

        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Gemma3p5DecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        if positions.dim() == 1:
            positions = einops.rearrange(positions, "s -> 1 s")

        position_embeddings_global = self.rotary_emb(hidden_states, positions)
        position_embeddings_local = self.rotary_emb_local(hidden_states, positions)

        # partitioning the vector into blocks
        blocks = hidden_states.split(split_size=self.config.altup_num_inputs)

        for layer in self.layers:
            hidden_states = layer(
                positions=positions,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                hidden_states=blocks,
                forward_batch=forward_batch,
                **kwargs,
            )

        # concat blocks
        hidden_states = torch.concat(hidden_states, dim=0)
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


# Copied and modified from https://github.com/devvrit/matformer/blob/main/modified_llama.py
# TODO: not modified for Gemma3p5 yet
class Gemma3p5MatFormerMLP(Gemma3p5MLP):
    def __init__(self, config, scale_factors, quant_config=None, prefix=""):
        super().__init__(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.intermediate_size = config.intermediate_size
        self.scale_factors = (
            scale_factors  # List of scale factors for 's', 'm', 'l', 'xl'
        )
        self.current_subset_hd = None

    def configure_subnetwork(self, flag):
        """Configure subnetwork size based on flag."""
        hd = self.intermediate_size
        if flag == "s":
            scale = self.scale_factors[0]  # hd/8
        elif flag == "m":
            scale = self.scale_factors[1]  # hd/4
        elif flag == "l":
            scale = self.scale_factors[2]  # hd/2
        else:  # 'xl'
            scale = self.scale_factors[3]  # hd

        self.current_subset_hd = int(hd * scale)

    def forward(self, x):
        if self.current_subset_hd is None:
            raise ValueError(
                "Subnetwork size not configured. Call `configure_subnetwork` first."
            )
        gate_proj = self.gate_proj.weight[: self.current_subset_hd]
        up_proj = self.up_proj.weight[: self.current_subset_hd]
        down_proj = self.down_proj.weight[:, : self.current_subset_hd]
        down_proj = F.linear(
            self.act_fn(F.linear(x, gate_proj) * F.linear(x, up_proj)), down_proj
        )
        self.current_subset_hd = None

        return down_proj


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

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config_class = Gemma3p5TextConfig
    base_model_prefix = "language_model"

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

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    # Gemma does not apply LoRA to the embedding layer.
    embedding_modules = {}
    embedding_padding_modules = []
    supports_lora = True

    def __init__(
        self,
        config: Gemma3p5Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config
        self.model = Gemma3p5TextModel(
            config=config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
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

        # config.matformer_scale_factors and config.matformer_flag are not seen in the config file
        scale_factors = config.matformer_scale_factors

        # Replace FFN in each layer with ModifiedFFN
        for layer_idx in range(config.num_hidden_layers):
            self.model.layers[layer_idx].mlp = Gemma3p5MatFormerMLP(
                config, scale_factors
            )

        self.configure_subnetwork(config.matformer_flag)

        self.post_init()

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
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
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
        # unloaded_params = params_dict.keys() - loaded_params
        # if unloaded_params:
        #     logger.warning(
        #         "Some weights are not initialized from checkpoints: %s", unloaded_params
        #     )
        return loaded_params


EntryClass = Gemma3p5ForCausalLM
AutoModel.register(Gemma3p5Config, Gemma3p5ForCausalLM, exist_ok=True)
