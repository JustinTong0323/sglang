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
"""Utilities for Huggingface Transformers.

This package re-exports all public symbols so that existing imports like
``from sglang.srt.utils.hf_transformers_utils import X`` continue to work
after the module was split into a subpackage.
"""

# Apply compatibility patches first (before any from_pretrained call).
from .compat import apply_all as _apply_compat

_apply_compat()

# Re-export public API from submodules.
from .compat import normalize_rope_scaling_compat  # noqa: E402
from .common import (  # noqa: E402
    CONTEXT_LENGTH_KEYS,
    attach_additional_stop_token_ids,
    check_gguf_file,
    download_from_hf,
    get_context_length,
    get_generation_config,
    get_hf_text_config,
    get_rope_config,
    get_sparse_attention_config,
    get_tokenizer_from_processor,
)
from .config import get_config  # noqa: E402
from .processor import get_processor  # noqa: E402
from .tokenizer import get_tokenizer  # noqa: E402

__all__ = [
    "CONTEXT_LENGTH_KEYS",
    "attach_additional_stop_token_ids",
    "check_gguf_file",
    "download_from_hf",
    "get_config",
    "get_context_length",
    "get_generation_config",
    "get_hf_text_config",
    "get_processor",
    "get_rope_config",
    "get_sparse_attention_config",
    "get_tokenizer",
    "get_tokenizer_from_processor",
    "normalize_rope_scaling_compat",
]
