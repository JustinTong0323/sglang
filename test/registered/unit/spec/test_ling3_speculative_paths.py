import ast
import inspect
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.arg_groups.speculative_hook import (
    _auto_choose_speculative_params,
    _handle_eagle_family,
)
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_ling_nextn_uses_self_draft_and_default_speculative_shape():
    source = ast.parse(inspect.getsource(_handle_eagle_family))
    architectures = [
        node.value
        for node in ast.walk(source)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    args = SimpleNamespace(speculative_algorithm="EAGLE")

    assert "BailingMoeV3ForCausalLM" in architectures
    assert _auto_choose_speculative_params(args, "BailingMoeV3ForCausalLM") == (3, 1, 4)


def test_dspark_draft_sampler_resolves_fused_greedy_toggle():
    model = SimpleNamespace(markov_head=VanillaMarkov(vocab_size=16, markov_rank=4))
    sampler = DsparkDraftSampler(
        model=model,
        gamma=2,
        max_bs=1,
        device=torch.device("cpu"),
        folded_sampling=False,
    )

    assert sampler._fused_greedy is False


def test_dspark_commits_the_last_accepted_mamba_state():
    worker = object.__new__(DSparkWorkerV2)
    backend = MagicMock()
    model = object()
    worker._need_mamba_verify_commit = True
    worker._target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=backend, model=model)
    )
    batch = SimpleNamespace(mamba_track_indices=None)

    with get_context().override_server_args(
        speculative_eagle_topk=1, mamba_track_interval=4
    ):
        worker._commit_target_mamba_states_after_verify(
            batch=batch,
            seq_lens_pre_verify=torch.tensor([8, 12]),
            seq_lens_post_verify=torch.tensor([9, 15]),
            commit_lens=torch.tensor([1, 3], dtype=torch.int32),
        )

    kwargs = backend.update_mamba_state_after_mtp_verify.call_args.kwargs
    torch.testing.assert_close(
        kwargs["last_correct_step_indices"], torch.tensor([0, 2])
    )
    assert kwargs["mamba_track_indices"] is None
    assert kwargs["mamba_steps_to_track"] is None
    assert kwargs["model"] is model


def test_dspark_commits_tracking_boundary_state():
    worker = object.__new__(DSparkWorkerV2)
    backend = MagicMock()
    worker._need_mamba_verify_commit = True
    worker._target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=backend, model=object())
    )
    batch = SimpleNamespace(mamba_track_indices=torch.tensor([20, 21]))

    with get_context().override_server_args(
        speculative_eagle_topk=1, mamba_track_interval=4
    ):
        worker._commit_target_mamba_states_after_verify(
            batch=batch,
            seq_lens_pre_verify=torch.tensor([3, 4]),
            seq_lens_post_verify=torch.tensor([5, 7]),
            commit_lens=torch.tensor([2, 3], dtype=torch.int32),
        )

    kwargs = backend.update_mamba_state_after_mtp_verify.call_args.kwargs
    torch.testing.assert_close(kwargs["mamba_steps_to_track"], torch.tensor([0, -1]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
