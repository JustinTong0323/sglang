# copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modular_gemma3.py
from typing import Any, Optional, Union

from transformers import PretrainedConfig, logging
from transformers.modeling_rope_utils import rope_config_validation

# from ...configuration_utils import PretrainedConfig
# from ...modeling_rope_utils import rope_config_validation
# from ...utils import logging
# from ..siglip import SiglipVisionConfig


logger = logging.get_logger(__name__)


class Gemma3p5TextConfig(PretrainedConfig):
    model_type = "gemma3p5_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=262_144,
        hidden_size=2048,
        intermediate_size=16384,
        num_hidden_layers=35,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=32_768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=1_000_000.0,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=512,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=None,
        cache_implementation="hybrid",
        rope_scaling=None,
        rope_local_base_freq=10_000.0,
        sliding_window_pattern=5,
        activation_sparsity_pattern=None,
        altup_active_idx=0,
        altup_coef_clip=120.0,
        altup_correct_scale=True,
        altup_lr_multiplier=1.0,
        altup_num_inputs=4,
        architectures=["Gemma3p5ForCausalLM"],
        frac_shared_layers=0.5,
        hidden_size_per_layer_input=256,
        laurel_rank=64,
        torch_dtype="bfloat16",
        transformers_version="4.52.0.dev0",
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.cache_implementation = cache_implementation
        self.rope_local_base_freq = rope_local_base_freq
        self.sliding_window_pattern = sliding_window_pattern
        self.rope_scaling = rope_scaling
        self.activation_sparsity_pattern = activation_sparsity_pattern
        self.altup_active_idx = altup_active_idx
        self.altup_coef_clip = altup_coef_clip
        self.altup_correct_scale = altup_correct_scale
        self.altup_lr_multiplier = altup_lr_multiplier
        self.altup_num_inputs = altup_num_inputs
        self.architectures = architectures
        self.frac_shared_layers = frac_shared_layers
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.laurel_rank = laurel_rank
        self.torch_dtype = torch_dtype
        self.transformers_version = transformers_version
        rope_config_validation(self)


class Gemma3p5AudioConfig(PretrainedConfig):
    model_type = "gemma3p5"

    def __init__(
        self,
        *args,
        hidden_size: int = 1536,
        embedding_norm_eps: float = 1e-6,
        vocab_size: int = 256_128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.embedding_norm_eps = embedding_norm_eps
        self.vocab_size = vocab_size


class Gemma3p5VisionConfig(PretrainedConfig):
    model_type = "gemma3p5"

    def __init__(
        self,
        *args,
        embedding_norm_eps: float = 1e-6,
        hidden_size: int = 2048,
        vocab_size: int = 128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.embedding_norm_eps = embedding_norm_eps
        self.vocab_size = vocab_size


class Gemma3p5Config(PretrainedConfig):
    model_type = "gemma3p5"
    sub_configs = {
        "text_config": Gemma3p5TextConfig,
        "vision_config": Gemma3p5VisionConfig,
        "audio_config": Gemma3p5AudioConfig,
    }

    def __init__(
        self,
        text_config: Optional[Union[Gemma3p5TextConfig, dict[str, Any]]] = None,
        vision_config: Optional[Union[Gemma3p5VisionConfig, dict[str, Any]]] = None,
        audio_config: Optional[Union[Gemma3p5AudioConfig, dict[str, Any]]] = None,
        audio_soft_tokens_per_image: int = 256,
        vision_soft_tokens_per_image: int = 256,
        boi_token_id: int = 255_999,
        eoi_token_id: int = 256_000,
        image_token_id: int = 262_144,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(text_config, dict):
            text_config = Gemma3p5TextConfig(**text_config)
        elif text_config is None:
            text_config = Gemma3p5TextConfig()
            logger.info("text_config is None. Using default Gemma3p5TextConfig.")

        if isinstance(vision_config, dict):
            vision_config = Gemma3p5VisionConfig(**vision_config)
        elif vision_config is None:
            logger.info("vision_config is None. Vision capabilities will not be used.")

        if isinstance(audio_config, dict):
            audio_config = Gemma3p5AudioConfig(**audio_config)
        elif audio_config is None:
            logger.info("audio_config is None. Audio capabilities will not be used.")

        self.text_config = text_config
        self.vision_config = vision_config
        self.audio_config = audio_config

        self.audio_soft_tokens_per_image = audio_soft_tokens_per_image
        self.vision_soft_tokens_per_image = vision_soft_tokens_per_image
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.image_token_id = image_token_id
        self.initializer_range = initializer_range


# register_processor(Gemma3p5Config, Gemma3SGLangImageProcessor)
# register_image_processor(Gemma3p5Config, Gemma3SGLangImageProcessor)
