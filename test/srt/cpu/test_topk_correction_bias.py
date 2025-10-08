import unittest

import torch

from sglang.srt.layers.moe.topk import fused_topk_torch_native


class TestTopKCorrectionBias(unittest.TestCase):
    def _ref_with_bias(self, hidden_states: torch.Tensor, gating_output: torch.Tensor, correction_bias: torch.Tensor, topk: int, renormalize: bool):
        # Reference: apply softmax, add correction bias for selection only, gather original softmax weights
        scores = torch.softmax(gating_output, dim=-1)
        scores_for_choice = scores + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights, topk_ids

    def _run_case(self, M: int, E: int, topk: int, renormalize: bool, dtype: torch.dtype):
        torch.manual_seed(42)
        hidden_states = torch.randn(M, 16, dtype=dtype)
        gating_output = torch.randn(M, E, dtype=dtype) * 1.5
        correction_bias = torch.randn(E, dtype=torch.float32)

        # Function under test
        w, ids = fused_topk_torch_native(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            correction_bias=correction_bias,
        )

        # Reference
        ref_w, ref_ids = self._ref_with_bias(hidden_states, gating_output, correction_bias, topk, renormalize)

        # Compare (ids are not necessarily sorted)
        self.assertTrue(torch.all(ref_ids.sort(dim=-1).values == ids.sort(dim=-1).values))
        # For weights, sort by ids to make deterministic comparison
        sorted_idx_ref = ref_ids.argsort(dim=-1)
        sorted_idx = ids.argsort(dim=-1)
        self.assertTrue(
            torch.allclose(
                ref_w.gather(1, sorted_idx_ref),
                w.gather(1, sorted_idx),
                atol=1e-6,
                rtol=0,
            )
        )

    def test_small_configs(self):
        for renormalize in [True, False]:
            self._run_case(M=7, E=9, topk=3, renormalize=renormalize, dtype=torch.bfloat16)
            self._run_case(M=11, E=16, topk=4, renormalize=renormalize, dtype=torch.bfloat16)


if __name__ == "__main__":
    unittest.main()

