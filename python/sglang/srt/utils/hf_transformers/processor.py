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
"""Processor loading utilities."""

import json
from pathlib import Path
from typing import Optional

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from sglang.srt.multimodal.customized_mm_processor_utils import _CUSTOMIZED_MM_PROCESSOR
from sglang.srt.utils import logger

from .common import (
    AutoConfig,
    _is_deepseek_ocr2_model,
    _is_deepseek_ocr_model,
    _is_mistral_model,
    _load_mistral_config,
    _override_v_head_dim_if_zero,
    _resolve_local_or_cached_file,
    attach_additional_stop_token_ids,
    download_from_hf,
    get_tokenizer_from_processor,
)
from .tokenizer import _fix_added_tokens_encoding, _fix_special_tokens_pattern


def _build_processor_manually(
    model_path, config, trust_remote_code, revision, **kwargs
):
    """Build processor when AutoProcessor fails to resolve feature_extractor_type.

    In transformers v5, AutoProcessor.from_pretrained calls
    AutoFeatureExtractor.from_pretrained which fails if
    preprocessor_config.json lacks 'feature_extractor_type'. This resolves
    the processor class via dynamic module resolution and constructs it with
    individually-loaded components.
    """
    import transformers
    from transformers import AutoImageProcessor, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    # Resolve processor class from auto_map -- check both the model config
    # and the preprocessor_config.json (some models like MiniCPM-o only
    # declare AutoProcessor in the latter).
    auto_map = getattr(config, "auto_map", None) or {}
    proc_ref = auto_map.get("AutoProcessor")
    if not proc_ref:
        try:
            pp_file = _resolve_local_or_cached_file(
                model_path, "preprocessor_config.json", revision
            )
            with open(pp_file) as f:
                pp_auto_map = json.load(f).get("auto_map", {})
            proc_ref = pp_auto_map.get("AutoProcessor")
        except Exception as e:
            logger.debug(
                "_build_processor_manually: could not read preprocessor_config.json "
                "for %s: %s",
                model_path,
                e,
            )
    if not proc_ref:
        raise ValueError(f"Cannot determine processor class for {model_path}")

    proc_cls = get_class_from_dynamic_module(
        proc_ref, model_path, code_revision=revision
    )

    # Load sub-components individually (these succeed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, revision=revision
    )
    init_kwargs = {"tokenizer": tokenizer}

    if "image_processor" in getattr(proc_cls, "attributes", []):
        try:
            init_kwargs["image_processor"] = AutoImageProcessor.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, revision=revision
            )
        except (ImportError, OSError, ValueError) as e:
            logger.error(
                "Failed to load image_processor for %s: %s. "
                "Multimodal features may not work correctly.",
                model_path,
                e,
            )

    # Instantiate feature extractor from its declared class
    fe_class_name = getattr(proc_cls, "feature_extractor_class", None)
    if fe_class_name:
        fe_class = getattr(transformers, fe_class_name, None)
        if fe_class is not None:
            init_kwargs["feature_extractor"] = fe_class()

    return proc_cls(**init_kwargs)


def _wrap_as_pixtral(processor, config):
    from transformers.models.pixtral.image_processing_pixtral import (
        PixtralImageProcessor,
    )
    from transformers.models.pixtral.processing_pixtral import (
        PixtralProcessor as HFPixtralProcessor,
    )

    vision_config = config.vision_config
    patch_size = vision_config.patch_size
    image_size = vision_config.image_size
    spatial_merge_size = getattr(vision_config, "spatial_merge_size", 1)

    effective_patch = patch_size * spatial_merge_size
    image_processor = PixtralImageProcessor(
        do_resize=True,
        size={"longest_edge": image_size},
        patch_size={"height": effective_patch, "width": effective_patch},
    )
    return HFPixtralProcessor(
        image_processor=image_processor,
        tokenizer=processor,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
    )


def get_processor(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    use_fast: Optional[bool] = True,
    **kwargs,
):
    revision = kwargs.pop("revision", tokenizer_revision)
    if _is_mistral_model(tokenizer_name):
        config = _load_mistral_config(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    else:
        config = AutoConfig.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )
    if _is_deepseek_ocr_model(config) or _is_deepseek_ocr2_model(config):
        config.model_type = "deepseek-ocr"
        config.update({"architectures": ["DeepseekOCRForCausalLM"]})
        if _is_deepseek_ocr2_model(config):
            _override_v_head_dim_if_zero(config)

    if config.model_type in {"qwen2_vl", "sarashina2_vision"}:
        if "size" not in kwargs:
            kwargs["size"] = {"shortest_edge": 3136, "longest_edge": 1003520}

    if config.model_type not in {"llava", "clip"}:
        kwargs["use_fast"] = use_fast
    try:
        if "InternVL3_5" in tokenizer_name:
            processor = AutoTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
        else:
            if config.model_type in _CUSTOMIZED_MM_PROCESSOR:
                processor = _CUSTOMIZED_MM_PROCESSOR[config.model_type].from_pretrained(
                    tokenizer_name,
                    *args,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    **kwargs,
                )
            else:
                processor = AutoProcessor.from_pretrained(
                    tokenizer_name,
                    *args,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    **kwargs,
                )

    except ValueError as e:
        error_message = str(e)
        if "does not have a slow version" in error_message:
            logger.info(
                "Processor %s does not have a slow version. Automatically use fast version",
                tokenizer_name,
            )
            kwargs["use_fast"] = True
            processor = AutoProcessor.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
        elif "Unrecognized feature extractor" in error_message:
            logger.info(
                "AutoProcessor failed on feature extractor for %s, "
                "constructing processor manually",
                tokenizer_name,
            )
            processor = _build_processor_manually(
                tokenizer_name,
                config,
                trust_remote_code,
                revision,
                **kwargs,
            )
        else:
            raise
    if (
        isinstance(processor, PreTrainedTokenizerBase)
        and getattr(config, "model_type", None) == "pixtral"
    ):
        processor = _wrap_as_pixtral(processor, config)

    tokenizer = get_tokenizer_from_processor(processor)

    if tokenizer.chat_template is None:
        local_path = download_from_hf(
            tokenizer_name, allow_patterns=["*.json", "*.jinja", "*.model"]
        )
        jinja_path = Path(local_path) / "chat_template.jinja"
        if jinja_path.is_file():
            tokenizer.chat_template = jinja_path.read_text()
            logger.info("Loaded chat_template from %s", jinja_path)

    _fix_special_tokens_pattern(tokenizer)
    _fix_added_tokens_encoding(tokenizer)
    attach_additional_stop_token_ids(tokenizer)
    return processor
