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
"""Config loading utilities."""

from pathlib import Path
from typing import Optional

from transformers import PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from sglang.srt.connector import create_remote_connector
from sglang.srt.utils import is_remote_url, lru_cache_frozenset

from .compat import _ensure_gguf_version
from .common import (
    AutoConfig,
    DeepseekVLV2Config,
    _CONFIG_REGISTRY,
    _is_deepseek_ocr2_model,
    _is_deepseek_ocr_model,
    _is_mistral_model,
    _load_deepseek_v32_model,
    _load_mistral_config,
    _override_deepseek_ocr_v_head_dim,
    _override_v_head_dim_if_zero,
    check_gguf_file,
    get_hf_text_config,
)


@lru_cache_frozenset(maxsize=32)
def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_override_args: Optional[dict] = None,
    **kwargs,
):
    is_gguf = check_gguf_file(model)
    if is_gguf:
        _ensure_gguf_version()
        kwargs["gguf_file"] = model
        model = Path(model).parent

    if is_remote_url(model):
        # BaseConnector implements __del__() to clean up the local dir.
        # Since config files need to exist all the time, so we DO NOT use
        # with statement to avoid closing the client.
        client = create_remote_connector(model)
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        model = client.get_local_dir()

    if _is_mistral_model(model):
        config = _load_mistral_config(
            model, trust_remote_code=trust_remote_code, revision=revision
        )
    else:
        try:
            config = AutoConfig.from_pretrained(
                model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
            )
        except ValueError as e:
            if not "deepseek_v32" in str(e):
                raise e
            config = _load_deepseek_v32_model(
                model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
            )
        except KeyError as e:
            # Transformers v5 may register a built-in config class that
            # conflicts with sglang's custom one (e.g. NemotronHConfig
            # doesn't handle '-' in hybrid_override_pattern). Fall back
            # to loading the raw config dict and using sglang's class.
            # Also handle deepseek_v32 which v5 doesn't recognize.
            if "deepseek_v32" in str(e):
                config = _load_deepseek_v32_model(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    **kwargs,
                )
            else:
                config_dict, _ = PretrainedConfig.get_config_dict(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    **kwargs,
                )
                model_type = config_dict.get("model_type")
                if model_type in _CONFIG_REGISTRY:
                    config = _CONFIG_REGISTRY[model_type].from_dict(config_dict)
                    config._name_or_path = model
                else:
                    raise

    if (
        config.architectures is not None
        and config.architectures[0] == "Phi4MMForCausalLM"
    ):
        # Phi4MMForCausalLM uses a hard-coded vision_config. See:
        # https://github.com/vllm-project/vllm/blob/6071e989df1531b59ef35568f83f7351afb0b51e/vllm/model_executor/models/phi4mm.py#L71
        # We set it here to support cases where num_attention_heads is not divisible by the TP size.
        from transformers import SiglipVisionConfig

        vision_config = {
            "hidden_size": 1152,
            "image_size": 448,
            "intermediate_size": 4304,
            "model_type": "siglip_vision_model",
            "num_attention_heads": 16,
            "num_hidden_layers": 26,
            # Model is originally 27-layer, we only need the first 26 layers for feature extraction.
            "patch_size": 14,
        }
        config.vision_config = SiglipVisionConfig(**vision_config)

    if config.architectures in [
        ["LongcatCausalLM"],
        ["LongcatFlashForCausalLM"],
        ["LongcatFlashNgramForCausalLM"],
    ]:
        config.model_type = "longcat_flash"

    text_config = get_hf_text_config(config=config)

    if isinstance(model, str) and text_config is not None:
        items = (
            text_config.items()
            if hasattr(text_config, "items")
            else vars(text_config).items()
        )
        for key, val in items:
            if not hasattr(config, key) and val is not None:
                setattr(config, key, val)

    if _is_deepseek_ocr2_model(config):
        _override_v_head_dim_if_zero(config)
        # Temporary hack for load deepseek-ocr2
        config.model_type = "deepseek-ocr"
        config.update({"architectures": ["DeepseekOCRForCausalLM"]})
        config = DeepseekVLV2Config.from_pretrained(model, revision=revision)
        _override_v_head_dim_if_zero(config)
        config.update({"architectures": ["DeepseekOCRForCausalLM"]})
        setattr(config, "_name_or_path", model)
    elif config.model_type in _CONFIG_REGISTRY:
        model_type = config.model_type
        if model_type == "deepseek_vl_v2":
            if _is_deepseek_ocr_model(config) or _is_deepseek_ocr2_model(config):
                model_type = "deepseek-ocr"
        config_class = _CONFIG_REGISTRY[model_type]
        config = config_class.from_pretrained(model, revision=revision)

        if _is_deepseek_ocr_model(config):
            _override_deepseek_ocr_v_head_dim(config)
            config.update({"architectures": ["DeepseekOCRForCausalLM"]})
        elif _is_deepseek_ocr2_model(config):
            _override_v_head_dim_if_zero(config)
            config.update({"architectures": ["DeepseekOCRForCausalLM"]})

        # NOTE(HandH1998): Qwen2VL requires `_name_or_path` attribute in `config`.
        setattr(config, "_name_or_path", model)

    if isinstance(model, str) and config.model_type == "internvl_chat":
        for key, val in config.llm_config.__dict__.items():
            if not hasattr(config, key):
                setattr(config, key, val)

    if config.model_type == "multi_modality":
        config.update({"architectures": ["MultiModalityCausalLM"]})

    if config.model_type == "longcat_flash":
        config.update({"architectures": ["LongcatFlashForCausalLM"]})

    if model_override_args:
        config.update(model_override_args)

    # Special architecture mapping check for GGUF models
    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})

    return config
