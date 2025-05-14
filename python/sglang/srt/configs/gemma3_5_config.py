import fractions
from collections.abc import Sequence
from typing import Any, Optional, Union

from transformers import PretrainedConfig

from sglang.utils import logger

# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3p5/configuration_gemma3p5.py


class Gemma3p5TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Gemma3p5TextModel`]. It is used to instantiate an Gemma3p5Text
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Gemma3p5Text-4B.
    e.g. [google/gemma3p5_text-4b](https://huggingface.co/google/gemma3p5_text-4b) #TODO (sindhuraghuram): Update the link here
    Configuration objects inherit from [`Gemma3TextConfig`] and can be used to control the model outputs. Read the
    documentation from [`Gemma3TextConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 262208):
            Vocabulary size of the Gemma3p5Text model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Gemma3p5TextModel`]
        hidden_size (`int`, *optional*, defaults to 2304):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 9216):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 26):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 256):
            The attention head dimension.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the decoder. Will default to `"gelu_pytorch_tanh"`
            if not specified. `"gelu_pytorch_tanh"` uses an approximation of the `"gelu"` activation function.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        query_pre_attn_scalar (`float`, *optional*, defaults to 256):
            Scaling factor used on the attention scores
        sliding_window (`int`, *optional*, defaults to 4096): in Gemma3p5Text, every other layer uses sliding window attention. This is the
            size of the sliding window.
        final_logit_softcapping (`float`, *optional*):
            Scaling factor when applying tanh softcapping on the logits.
        attn_logit_softcapping (`float`, *optional*):
            Scaling factor when applying tanh softcapping on the attention scores.
        cache_implementation (`str`, *optional*, defaults to `"hybrid"`): the cache type to be used with `generate`.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings used in gloabl attention. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        rope_local_base_freq (float, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings for local attention.
        sliding_window_pattern (`int`, *optional*, defaults to 5):
            Pattern for the sliding window attention.

    TODO (sindhuraghuram): Update the list of configs

    ```python
    >>> from transformers import Gemma3p5TextModel, Gemma3p5TextConfig
    >>> # Initializing a Gemma3p5Text gemma3p5_text-4b style configuration
    >>> configuration = Gemma3p5TextConfig()
    >>> # Initializing a model from the gemma3p5_text-4b style configuration
    >>> model = Gemma3p5TextModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
        rope_local_base_freq (float, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings for local attention.
        sliding_window_pattern (`int`, *optional*, defaults to 5):
            Pattern for the sliding window attention.
    """

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
        vocab_size: int = 262_144,
        hidden_size: int = 2048,
        hidden_size_per_layer_input: int = 256,
        num_hidden_layers: int = 35,
        sliding_window: int = 512,
        intermediate_size: int = 16_384,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        hidden_activation: str = "gelu_pytorch_tanh",
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        cache_implementation: str = "hybrid",
        max_position_embeddings: int = 32_768,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        rope_theta: float = 1_000_000.0,
        rope_scaling: Optional[dict[str, Any]] = None,
        rope_local_base_freq: float = 10_000.0,
        query_pre_attn_scalar: int = 256,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window_pattern: int = 5,
        final_logit_softcapping: float = 30.0,
        attn_logit_softcapping: Optional[float] = None,
        altup_active_idx: int = 0,
        altup_coef_clip: float = 120.0,
        altup_correct_scale: bool = True,
        altup_lr_multiplier: float = 1.0,
        altup_num_inputs: int = 4,
        frac_shared_layers: Union[float, fractions.Fraction] = 0.5,
        laurel_rank: int = 64,
        activation_sparsity_pattern: Optional[Sequence[float]] = None,
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
        # For configuring HybridCache to work with 5:1 attention pattern
        self.sliding_window_pattern = sliding_window_pattern
        self.rope_scaling = rope_scaling
        # rope_config_validation(self)
        self.hidden_size_per_layer_input = hidden_size_per_layer_input

        self.altup_active_idx = altup_active_idx
        self.altup_coef_clip = altup_coef_clip
        self.altup_correct_scale = altup_correct_scale
        self.altup_lr_multiplier = altup_lr_multiplier
        self.altup_num_inputs = altup_num_inputs

        self.laurel_rank = laurel_rank

        self.frac_shared_layers = frac_shared_layers
        if (
            activation_sparsity_pattern is not None
            and (len_asp := len(activation_sparsity_pattern)) != num_hidden_layers
        ):
            raise ValueError(
                "activation_sparsity_pattern must have an explicit activation sparsity value for every layer."
                f"Expected {num_hidden_layers} values but got {len_asp}."
            )
        self.activation_sparsity_pattern = activation_sparsity_pattern


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
    r"""
    This is the configuration class to store the configuration of a [`Gemma3p5ForConditionalGeneration`]. It is used to instantiate an
    Gemma3p5ForConditionalGeneration according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the PaliGemma-2B.

    e.g. [google/gemma-3-4b](https://huggingface.co/google/gemma-3-4b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[Gemma3p5TextConfig, dict]`, *optional*):
            The config object of the text backbone.
        vision_config (`Union[AutoConfig, dict]`,  *optional*):
            Custom vision config or dict.
        mm_tokens_per_image (`int`, *optional*, defaults to 256):
            The number of tokens per image embedding.
        boi_token_id (`int`, *optional*, defaults to 255999):
            The begin-of-image token index to wrap the image prompt.
        eoi_token_id (`int`, *optional*, defaults to 256000):
            The end-of-image token index to wrap the image prompt.
        image_token_id (`int`, *optional*, defaults to 262144):
            The image token index to encode the image prompt.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.


    Example:

    ```python
    >>> from transformers import Gemma3p5ForConditionalGeneration, Gemma3p5Config, Gemma3p5TextConfig

    >>> # Initializing a MobileNet vision config
    >>> checkpoint = "timm/mobilenet_something_something"
    >>> vision_config = AutoConfig.from_pretrained(checkpoint)

    >>> # Initializing a Gemma3p5 Audio config
    >>> audio_config = Gemma3p5AudioConfig()

    >>> # Initializing a Gemma3p5 Text config
    >>> text_config = Gemma3p5TextConfig()

    >>> # Initializing a Gemma3p5 gemma-3-4b style configuration
    >>> configuration = Gemma3p5Config(text_config, vision_config, audio_config)

    >>> # Initializing a model from the gemma-3-4b style configuration
    >>> model = Gemma3p5TextConfig(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

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


__all__ = ["Gemma3p5Config", "Gemma3p5TextConfig"]
