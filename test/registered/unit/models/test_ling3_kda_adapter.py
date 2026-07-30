import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.models.bailing_moe_v3 import DsV3MLA
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.models.kimi_linear import KimiDeltaAttention
from sglang.srt.utils.hf_transformers_utils import get_rope_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _config():
    return SimpleNamespace(
        dtype=torch.bfloat16,
        linear_attn_config={
            "head_dim": 128,
            "num_heads": 16,
            "short_conv_kernel_size": 4,
        },
    )


def _parallel():
    return SimpleNamespace(
        tp_size=4,
        tp_rank=3,
        attn_tp_size=2,
        attn_tp_rank=1,
    )


def test_kimi_defaults_keep_global_tp_and_symmetric_shapes():
    with patch("sglang.srt.models.kimi_linear.get_parallel", return_value=_parallel()):
        attention = KimiDeltaAttention(0, 2048, _config())

    assert attention.shard_tp_size == 4
    assert attention.shard_tp_rank == 3
    assert attention.local_num_heads == 4
    assert attention.qkvb_sizes == [2048, 2048, 2048, 16]
    assert attention.split_sizes == [1536, 4, 256]
    assert attention.qkv_conv1d.output_sizes == [2048, 2048, 2048]
    assert attention.o_norm.hidden_size == 128
    assert attention.o_proj.input_size == 2048
    assert attention.attn.lower_bound is None


def test_ling_adapter_uses_attention_tp_and_asymmetric_value_shapes():
    with patch("sglang.srt.models.kimi_linear.get_parallel", return_value=_parallel()):
        attention = KimiDeltaAttention(
            0,
            2048,
            _config(),
            no_kda_lora=True,
            safe_gate=True,
            lower_bound=-5.0,
            shard_on_attn_tp=True,
            v_head_dim=64,
        )

    assert attention.shard_tp_size == 2
    assert attention.shard_tp_rank == 1
    assert attention.local_num_heads == 8
    assert attention.qkvbfg_sizes == [2048, 2048, 1024, 16, 2048, 1024]
    assert attention.split_sizes == [2560, 8, 1024, 512]
    assert attention.qkv_conv1d.output_sizes == [2048, 2048, 1024]
    assert attention.o_norm.hidden_size == 64
    assert attention.o_proj.input_size == 1024
    assert attention.attn.head_v_dim == 64
    assert attention.attn.lower_bound == -5.0


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (torch.empty(0), torch.empty(0)),
        ((torch.empty(0), None, object(), None), None),
    ],
)
def test_empty_dp_attention_shards_do_not_attach_gate(state, expected):
    gate = torch.ones(1)
    result = DsV3MLA._attach_gate(state, gate)

    if isinstance(state, tuple):
        assert result is state
        assert result[3] is expected
    else:
        assert result is state


def test_gate_attaches_only_to_supported_attention_cores():
    inner_state = (object(),)
    gate = torch.ones(1)
    state = (None, AttnForwardMethod.MLA, object(), inner_state)

    result = DsV3MLA._attach_gate(state, gate)

    assert result[:3] == state[:3]
    assert result[3] == inner_state + (gate,)
    unsupported = (None, AttnForwardMethod.MLA_NPU, object(), inner_state)
    assert DsV3MLA._attach_gate(unsupported, gate) is unsupported


def test_rope_config_supports_transformers_v4_and_v5_contracts():
    legacy = SimpleNamespace(rope_theta=600000.0, rope_scaling={"type": "linear"})
    modern = SimpleNamespace(
        rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"}
    )

    assert get_rope_config(legacy) == (600000.0, {"type": "linear"})
    assert get_rope_config(modern) == (
        1000000.0,
        {"rope_theta": 1000000.0, "rope_type": "default"},
    )


def _function_calls(path, function_name):
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    return [node for node in ast.walk(function) if isinstance(node, ast.Call)]


def _call_name(call):
    return ast.unparse(call.func)


@pytest.mark.parametrize(
    ("function_name", "call_names"),
    [
        (
            "forward_decode",
            {
                "kda_fused_decode.kda_fused_decode",
                "self.kernel_dispatcher.packed_decode",
                "self.kernel_dispatcher.decode",
            },
        ),
        ("forward_extend", {"self.kernel_dispatcher.extend"}),
        (
            "_forward_target_verify",
            {
                "self._fused_chain_verify_fn",
                "self.kernel_dispatcher.target_verify",
            },
        ),
        ("_run_dspark_cutedsl_mtp", {"fused_kda_decode_mtp_dspark"}),
    ],
)
def test_safe_gate_lower_bound_reaches_every_kda_lane(function_name, call_names):
    path = _REPO_ROOT / "python/sglang/srt/layers/attention/linear/kda_backend.py"
    calls = _function_calls(path, function_name)

    for call_name in call_names:
        call = next(call for call in calls if _call_name(call) == call_name)
        lower_bound = next(
            keyword.value for keyword in call.keywords if keyword.arg == "lower_bound"
        )
        assert ast.unparse(lower_bound) in {
            "layer.lower_bound",
            "float(layer.lower_bound)",
        }


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
