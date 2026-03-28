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
"""Compatibility patches for transformers v5.x.

This module applies monkey-patches to work around breaking changes in
transformers v5.  Each patch is tagged with the upstream issue it works
around so it can be removed once the upstream fix lands.

Import this module early (before any ``from_pretrained`` call) to activate
all patches.  It is safe to import multiple times -- patches are idempotent.

Patches fall into two categories:

1. **Transformers bugs / regressions** -- issues in transformers itself.
2. **Remote-model-code compat** -- remote model code (trust_remote_code)
   that hasn't been updated for v5 yet.  These should be removed once
   the model authors publish fixes.
"""

import inspect
import logging

logger = logging.getLogger(__name__)

_applied = False


# ---------------------------------------------------------------------------
# Public API: apply_all() -- import-time patches (idempotent)
# ---------------------------------------------------------------------------


def apply_all():
    """Apply all transformers compatibility patches (idempotent).

    Call this once at import time.  It is safe to call multiple times.
    """
    global _applied
    if _applied:
        return
    _applied = True

    # v5.4 patches (from transformers_v54_compat.py)
    _patch_flash_attn_availability()
    _patch_rope_parameters_validation()
    _patch_removed_symbols()
    _patch_image_processor_kwargs()

    # v5 general patches
    _ensure_clean_up_tokenization_compat()
    _ensure_is_torch_fx_available_compat()

    logger.debug("transformers compatibility patches applied")


# ---------------------------------------------------------------------------
# Public API: on-demand helpers (called explicitly by other modules)
# ---------------------------------------------------------------------------


def normalize_rope_scaling_compat(config) -> None:
    """Ensure rope_scaling dicts have ``"type"`` alongside ``"rope_type"``.

    Transformers v5 standardises rope_scaling to use ``"rope_type"`` and may
    omit the legacy ``"type"`` key.  Remote-code models (e.g. Kimi-VL) still
    read ``rope_scaling["type"]``, causing a ``KeyError``.  This helper adds
    ``"type"`` from ``"rope_type"`` whenever it is missing, recursively across
    the config and all its sub-configs.
    """

    def _patch(cfg):
        try:
            rs = getattr(cfg, "rope_scaling", None)
        except AttributeError:
            rs = None
        if isinstance(rs, dict) and "rope_type" in rs and "type" not in rs:
            rs["type"] = rs["rope_type"]
        # Recurse into sub-configs
        for attr in (
            "text_config",
            "llm_config",
            "language_config",
            "vision_config",
            "thinker_config",
        ):
            sub = getattr(cfg, attr, None)
            if sub is not None:
                _patch(sub)

    _patch(config)


def _ensure_gguf_version():
    """Workaround for transformers v5 bug where is_gguf_available() fails
    when the gguf package lacks __version__ and metadata lookup also fails,
    resulting in packaging.version.InvalidVersion: Invalid version: 'N/A'."""
    try:
        import gguf

        if not hasattr(gguf, "__version__"):
            import importlib.metadata

            try:
                gguf.__version__ = importlib.metadata.version("gguf")
            except Exception:
                gguf.__version__ = "0.0.0"
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# v5.4 patches (merged from transformers_v54_compat.py)
# ---------------------------------------------------------------------------


def _patch_rope_parameters_validation():
    """Fix rope_parameters validation for unregistered model types.

    Transformers v5.4+ validates that ``rope_parameters`` contains
    ``rope_theta`` for yarn/llama3/longrope types.  For unregistered model
    types (e.g. ``deepseek_v32``), the generic ``PretrainedConfig`` lacks a
    ``rope_parameters`` field so the conversion that injects ``rope_theta``
    from the top-level config is skipped, causing a ``KeyError``.

    Fix: patch ``PretrainedConfig.from_dict`` to inject ``rope_theta`` into
    ``rope_scaling`` before ``__init__`` validates.

    TODO(upstream): https://github.com/huggingface/transformers/issues/XXXXX
    """
    from transformers import PretrainedConfig

    original = PretrainedConfig.from_dict.__func__

    @classmethod  # type: ignore[misc]
    def patched(cls, config_dict, **kwargs):
        rope_scaling = config_dict.get("rope_scaling")
        rope_theta = config_dict.get("rope_theta")
        if (
            isinstance(rope_scaling, dict)
            and rope_theta is not None
            and "rope_theta" not in rope_scaling
        ):
            config_dict = config_dict.copy()
            config_dict["rope_scaling"] = {**rope_scaling, "rope_theta": rope_theta}
        return original(cls, config_dict, **kwargs)

    PretrainedConfig.from_dict = patched


def _patch_flash_attn_availability():
    """Prevent flash-attn-4 from masquerading as flash-attn-2.

    flash-attn-4 registers a bare ``flash_attn`` namespace that makes
    ``is_flash_attn_2_available()`` return True, but lacks the v2 API.
    Remote model code (e.g. Kimi-VL) guarded by that check will crash.

    TODO(upstream): model authors should check for specific API symbols.
    """
    try:
        import flash_attn as _fa

        if not hasattr(_fa, "flash_attn_func"):
            import transformers.utils as _u
            import transformers.utils.import_utils as _ui

            _ui.is_flash_attn_2_available = lambda: False
            _u.is_flash_attn_2_available = lambda: False
    except ImportError:
        pass


def _patch_removed_symbols():
    """Re-export symbols removed in transformers v5.4.0.

    Remote model code (e.g. DeepSeek-OCR) still imports these.
    ``check_imports`` in ``dynamic_module_utils.py`` validates imports at
    config-load time, so these must exist before any ``from_pretrained``.

    Removed symbols:
    - ``LlamaFlashAttention2`` -- replaced by unified ``LlamaAttention``
    - ``is_flash_attn_greater_or_equal_2_10`` -- replaced by
      ``is_flash_attn_greater_or_equal("2.1.0")``

    TODO(upstream): DeepSeek-OCR / deepseek_vl_v2 remote code needs update.
    """
    # LlamaFlashAttention2
    try:
        from transformers.models.llama import modeling_llama

        if not hasattr(modeling_llama, "LlamaFlashAttention2"):
            if hasattr(modeling_llama, "LlamaAttention"):
                modeling_llama.LlamaFlashAttention2 = modeling_llama.LlamaAttention
    except (ImportError, ModuleNotFoundError):
        pass

    # is_flash_attn_greater_or_equal_2_10
    try:
        import transformers.utils as _u

        if not hasattr(_u, "is_flash_attn_greater_or_equal_2_10"):
            if hasattr(_u, "is_flash_attn_greater_or_equal"):
                _u.is_flash_attn_greater_or_equal_2_10 = (
                    lambda: _u.is_flash_attn_greater_or_equal("2.1.0")
                )
            else:
                _u.is_flash_attn_greater_or_equal_2_10 = lambda: False
    except (ImportError, ModuleNotFoundError):
        pass


def _patch_image_processor_kwargs():
    """Allow remote image processors that lack ``**kwargs`` in preprocess().

    Transformers v5.4 passes new kwargs (e.g. ``device``) through
    ``BaseImageProcessor.__call__`` -> ``preprocess()``.  Remote model code
    (e.g. KimiVL) that defines ``preprocess()`` without ``**kwargs`` will
    crash with ``TypeError``.

    Fix: wrap ``__call__`` to catch ``TypeError`` and retry with only the
    kwargs that ``preprocess()`` actually accepts.

    TODO(upstream): KimiVL image_processing_kimi_vl.py needs ``**kwargs``.
    """
    try:
        from transformers.image_processing_utils import BaseImageProcessor

        original = BaseImageProcessor.__call__

        def safe_call(self, images, *args, **kwargs):
            try:
                return original(self, images, *args, **kwargs)
            except TypeError as e:
                if "unexpected keyword argument" not in str(e):
                    raise
                sig = inspect.signature(self.preprocess)
                params = sig.parameters
                if any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                ):
                    raise
                valid = {k: v for k, v in kwargs.items() if k in params}
                return original(self, images, *args, **valid)

        BaseImageProcessor.__call__ = safe_call
    except (ImportError, ModuleNotFoundError):
        pass


# ---------------------------------------------------------------------------
# v5 general patches
# ---------------------------------------------------------------------------


def _ensure_clean_up_tokenization_compat() -> None:
    """Re-add ``clean_up_tokenization`` removed in transformers v5.

    Remote-code tokenizers (e.g. InternLM2Tokenizer) call
    ``self.clean_up_tokenization()`` which was a static method on
    ``PreTrainedTokenizerBase`` in v4 but removed in v5. Patch it back
    so existing HuggingFace Hub tokenizer code keeps working.
    """
    from transformers import PreTrainedTokenizerBase

    if hasattr(PreTrainedTokenizerBase, "clean_up_tokenization"):
        return

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    PreTrainedTokenizerBase.clean_up_tokenization = clean_up_tokenization


def _ensure_is_torch_fx_available_compat() -> None:
    """Re-add ``is_torch_fx_available`` removed in transformers v5.

    Remote-code models (e.g. MiniCPM-V) import ``is_torch_fx_available``
    from ``transformers.utils.import_utils``.  The function was removed
    in v5.  Patch it back so existing HuggingFace Hub model code keeps
    working.  torch.fx is always available in PyTorch >= 2.0.
    """
    import transformers.utils.import_utils as _import_utils

    if hasattr(_import_utils, "is_torch_fx_available"):
        return

    _import_utils.is_torch_fx_available = lambda: True
