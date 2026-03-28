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
"""Tokenizer loading utilities."""

import json
import logging
import warnings
from pathlib import Path
from typing import Optional, Union

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from sglang.srt.connector import create_remote_connector
from sglang.srt.utils import is_remote_url, logger
from sglang.srt.utils.patch_tokenizer import patch_tokenizer

from .compat import _ensure_gguf_version
from .common import (
    _resolve_local_or_cached_file,
    attach_additional_stop_token_ids,
    check_gguf_file,
)

# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


# Filter warnings like: https://github.com/sgl-project/sglang/issues/8082
class TokenizerWarningsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Calling super().encode with" not in record.getMessage()


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_name.endswith(".json"):
        from sglang.srt.tokenizer.tiktoken_tokenizer import TiktokenTokenizer

        return TiktokenTokenizer(tokenizer_name)

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
    elif tokenizer_mode == "auto":
        # In Transformers v5, the default for use_fast changed from True to False.
        # Explicitly set use_fast=True for "auto" mode to maintain previous behavior
        # and avoid issues with models that have incorrect tokenizer_class values.
        if "use_fast" not in kwargs:
            kwargs["use_fast"] = True

    # TODO(Xinyuan): Remove this once we have a proper tokenizer for Devstral
    if tokenizer_name == "mistralai/Devstral-Small-2505":
        tokenizer_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    is_gguf = check_gguf_file(tokenizer_name)
    if is_gguf:
        _ensure_gguf_version()
        kwargs["gguf_file"] = tokenizer_name
        tokenizer_name = Path(tokenizer_name).parent

    if is_remote_url(tokenizer_name):
        # BaseConnector implements __del__() to clean up the local dir.
        # Since config files need to exist all the time, so we DO NOT use
        # with statement to avoid closing the client.
        client = create_remote_connector(tokenizer_name)
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        tokenizer_name = client.get_local_dir()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            clean_up_tokenization_spaces=False,
            **kwargs,
        )
        # Filter tokenizer warnings
        logging.getLogger(tokenizer.__class__.__module__).addFilter(
            TokenizerWarningsFilter()
        )
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA V1 model "
            f"consider using '{_FAST_LLAMA_TOKENIZER}' instead of the "
            "original tokenizer."
        )
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if not trust_remote_code and (
            "does not exist or is not currently imported." in str(e)
            or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    # Transformers v5 may silently fall back to a generic TokenizersBackend
    # when the model requires a custom tokenizer. Retry with trust_remote_code
    # and/or use_fast=False to load the correct tokenizer class.
    if type(tokenizer).__name__ == "TokenizersBackend":
        retry_kwargs = {**kwargs, "trust_remote_code": True}
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            tokenizer_revision=tokenizer_revision,
            clean_up_tokenization_spaces=False,
            **retry_kwargs,
        )
    if type(tokenizer).__name__ == "TokenizersBackend":
        retry_kwargs["use_fast"] = False
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            tokenizer_revision=tokenizer_revision,
            clean_up_tokenization_spaces=False,
            **retry_kwargs,
        )

    _fix_v5_tokenizer_components(tokenizer, tokenizer_name, tokenizer_revision)
    _fix_v5_add_bos_eos_token(tokenizer, tokenizer_name, tokenizer_revision)

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        warnings.warn(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    _fix_special_tokens_pattern(tokenizer)
    attach_additional_stop_token_ids(tokenizer)
    tokenizer = patch_tokenizer(tokenizer)
    return tokenizer


def _fix_v5_tokenizer_components(tokenizer, model_name_or_path, revision=None):
    """Fix pre_tokenizer/decoder when a v5 tokenizer class overwrites them.

    In transformers v5, some tokenizer classes (e.g. LlamaTokenizer) have a
    custom __init__ that rebuilds the pre_tokenizer and decoder from scratch
    with class-specific components, discarding the originals from tokenizer.json.
    This breaks models that specify LlamaTokenizerFast but actually use a
    different tokenizer architecture (e.g. DeepSeek-V3.2 uses ByteLevel).

    Detects the mismatch by comparing against the raw tokenizer.json and
    restores the original components when they differ.
    """
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None:
        return

    try:
        from tokenizers import Tokenizer as RawTokenizer

        tok_file = _resolve_local_or_cached_file(
            model_name_or_path, "tokenizer.json", revision
        )
        raw = RawTokenizer.from_file(tok_file)
    except Exception as e:
        logger.debug(
            "_fix_v5_tokenizer_components: could not load tokenizer.json for %s: %s",
            model_name_or_path,
            e,
        )
        return

    raw_pre = type(raw.pre_tokenizer).__name__ if raw.pre_tokenizer else None
    loaded_pre = type(backend.pre_tokenizer).__name__ if backend.pre_tokenizer else None

    if raw_pre and loaded_pre and raw_pre != loaded_pre:
        logger.info(
            "Fixing v5 tokenizer component mismatch for %s: "
            "pre_tokenizer %s -> %s, decoder %s -> %s",
            model_name_or_path,
            loaded_pre,
            raw_pre,
            type(backend.decoder).__name__ if backend.decoder else None,
            type(raw.decoder).__name__ if raw.decoder else None,
        )
        backend.pre_tokenizer = raw.pre_tokenizer
        backend.decoder = raw.decoder


def _fix_v5_add_bos_eos_token(tokenizer, model_name_or_path, revision=None):
    """Restore add_bos_token/add_eos_token stripped by transformers v5.

    In transformers v5, _from_pretrained() strips add_bos_token and
    add_eos_token from init kwargs when a tokenizer.json file is present,
    assuming the tokenizer.json post-processor handles BOS/EOS addition.
    However, many models (e.g. DeepSeek-V3) have a tokenizer.json whose
    post-processor does NOT add BOS/EOS, and rely on the add_bos_token flag
    from tokenizer_config.json instead. This causes silent accuracy regressions.

    This function reads the tokenizer_config.json and restores the values,
    but only for tokenizer classes that actually supported these flags in v4.
    Classes like Qwen2Tokenizer did not support add_bos_token/add_eos_token
    in v4, so restoring them would change behavior.
    """
    # In transformers v4, only certain tokenizer classes supported
    # add_bos_token / add_eos_token as init parameters.  Restoring these
    # flags for classes that never supported them (e.g. Qwen2Tokenizer)
    # would incorrectly change tokenization behavior.
    _V4_CLASSES_WITH_BOS_EOS_FLAGS = frozenset(
        {
            "LlamaTokenizer",
            "LlamaTokenizerFast",
            "CodeLlamaTokenizer",
            "CodeLlamaTokenizerFast",
            "GemmaTokenizer",
            "GemmaTokenizerFast",
            "CohereTokenizerFast",
        }
    )

    try:
        config_file = _resolve_local_or_cached_file(
            model_name_or_path, "tokenizer_config.json", revision
        )
        with open(config_file) as f:
            config = json.load(f)
    except Exception as e:
        logger.debug(
            "_fix_v5_add_bos_eos_token: could not read tokenizer_config.json "
            "for %s: %s",
            model_name_or_path,
            e,
        )
        return

    tokenizer_class = config.get("tokenizer_class", "")
    if tokenizer_class not in _V4_CLASSES_WITH_BOS_EOS_FLAGS:
        logger.debug(
            "_fix_v5_add_bos_eos_token: skipping %s (tokenizer_class=%s "
            "did not support add_bos/eos_token in v4)",
            model_name_or_path,
            tokenizer_class,
        )
        return

    # In v4, Llama/Gemma tokenizers defaulted add_bos_token=True.
    # When the config omits the key or has null, use the v4 default so that
    # update_post_processor() doesn't drop BOS/EOS that was there before.
    _V4_DEFAULTS = {"add_bos_token": True, "add_eos_token": False}

    changed = False
    for attr in ("add_bos_token", "add_eos_token"):
        config_val = config.get(attr)
        if config_val is None:
            # Key missing or null -> use v4 default for this tokenizer class
            config_val = _V4_DEFAULTS.get(attr, False)
        current_val = getattr(tokenizer, attr, None)
        if current_val != config_val:
            logger.info(
                "Restoring %s=%s for %s (was %s after v5 loading)",
                attr,
                config_val,
                model_name_or_path,
                current_val,
            )
            setattr(tokenizer, f"_{attr}", config_val)
            changed = True

    # Rebuild the post-processor so it respects the restored flags
    if changed and hasattr(tokenizer, "update_post_processor"):
        tokenizer.update_post_processor()


def _fix_special_tokens_pattern(tokenizer):
    """Fix https://github.com/huggingface/transformers/pull/42563 which defaults
    special_tokens_pattern to "cls_sep", inserting None into token IDs when
    cls_token/sep_token are undefined (e.g. Kimi-VL's TikTokenTokenizer).
    """
    pattern = getattr(tokenizer, "special_tokens_pattern", None)
    if pattern == "cls_sep" and (
        tokenizer.cls_token_id is None or tokenizer.sep_token_id is None
    ):
        tokenizer.special_tokens_pattern = "none"


def _fix_added_tokens_encoding(tokenizer):
    """Ensure special tokens encode as single tokens in transformers v5.

    Some model tokenizers (e.g. MiniCPM-V-4) define special tokens like <image>,
    <slice> as attributes on the tokenizer class with corresponding IDs in the
    vocabulary (via tokenizer.json's added_tokens). In transformers v5, these
    tokens may not appear in get_added_vocab() and encode() splits them into
    subwords, breaking multimodal pipelines that rely on finding them in input_ids.

    This function discovers such tokens by scanning tokenizer attributes, checks
    if they encode correctly, and re-registers any that don't.
    """
    # Discover special token strings from tokenizer attributes.
    # Model tokenizers (e.g. MiniCPMVTokenizerFast) store them as attributes
    # like im_start="<image>", slice_start="<slice>", etc.
    candidates = {}
    for attr in dir(tokenizer):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(tokenizer, attr)
        except Exception:
            continue
        if (
            not isinstance(val, str)
            or not val.startswith("<")
            or not val.endswith(">")
            or len(val) > 20
        ):
            continue
        token_id = tokenizer.convert_tokens_to_ids(val)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            candidates[val] = token_id

    if not candidates:
        return

    # Check which tokens fail to encode as single tokens.
    broken = []
    for token_str, expected_id in candidates.items():
        try:
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(ids) != 1 or ids[0] != expected_id:
                broken.append(token_str)
        except Exception:
            broken.append(token_str)

    if not broken:
        return

    from transformers import AddedToken

    tokens_to_add = [AddedToken(tok, special=True, normalized=False) for tok in broken]
    tokenizer.add_tokens(tokens_to_add, special_tokens=True)
    logger.info(
        "Re-registered %d special tokens for correct v5 encoding: %s",
        len(broken),
        broken[:10],
    )
