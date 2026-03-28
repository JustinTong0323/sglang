"""Compatibility patches for transformers v5.4.0.

This module applies monkey-patches to work around breaking changes in
transformers 5.4.0.  Each patch is tagged with the upstream issue it works
around so it can be removed once the upstream fix lands.

Import this module early (before any ``from_pretrained`` call) to activate
all patches.  It is safe to import multiple times — patches are idempotent.

Patches fall into two categories:

1. **Transformers bugs / regressions** — issues in transformers itself.
2. **Remote-model-code compat** — remote model code (trust_remote_code)
   that hasn't been updated for 5.4.0 yet.  These should be removed once
   the model authors publish fixes.
"""

import inspect
import logging

logger = logging.getLogger(__name__)

_applied = False


def apply_all():
    """Apply all transformers v5.4 compatibility patches (idempotent)."""
    global _applied
    if _applied:
        return
    _applied = True

    _patch_flash_attn_availability()
    _patch_rope_parameters_validation()
    _patch_removed_symbols()
    _patch_image_processor_kwargs()

    logger.debug("transformers v5.4 compatibility patches applied")


# ---------------------------------------------------------------------------
# 1. Transformers bugs / regressions
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


# ---------------------------------------------------------------------------
# 2. Remote-model-code compat
# ---------------------------------------------------------------------------


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
    - ``LlamaFlashAttention2`` — replaced by unified ``LlamaAttention``
    - ``is_flash_attn_greater_or_equal_2_10`` — replaced by
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
                _u.is_flash_attn_greater_or_equal_2_10 = lambda: _u.is_flash_attn_greater_or_equal("2.1.0")
            else:
                _u.is_flash_attn_greater_or_equal_2_10 = lambda: False
    except (ImportError, ModuleNotFoundError):
        pass


def _patch_image_processor_kwargs():
    """Allow remote image processors that lack ``**kwargs`` in preprocess().

    Transformers v5.4 passes new kwargs (e.g. ``device``) through
    ``BaseImageProcessor.__call__`` → ``preprocess()``.  Remote model code
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
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                    raise
                valid = {k: v for k, v in kwargs.items() if k in params}
                return original(self, images, *args, **valid)

        BaseImageProcessor.__call__ = safe_call
    except (ImportError, ModuleNotFoundError):
        pass
