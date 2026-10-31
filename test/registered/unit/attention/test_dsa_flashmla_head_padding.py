"""CPU contract tests for sparse FlashMLA query-head padding."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashMLASparseHeadPadding(CustomTestCase):
    def test_query_and_sink_padding_match_kernel_shape(self):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        # The 512-wide Blackwell case retains head128 until separately validated.
        cases = [
            (10, 576, 8, 64),
            (10, 576, 64, 64),
            (10, 576, 128, 128),
            (10, 512, 8, 128),
            (9, 576, 8, 64),
        ]
        for sm_major, head_dim, heads, expected_heads in cases:
            with self.subTest(sm_major=sm_major, head_dim=head_dim, heads=heads):
                required_padding = (
                    DeepseekSparseAttnBackend._flashmla_sparse_head_padding(
                        device_sm_major=sm_major, head_dim=head_dim, num_heads=heads
                    )
                )
                self.assertEqual(required_padding, expected_heads)
                q = torch.arange(2 * heads * head_dim, dtype=torch.float32).reshape(
                    2, heads, head_dim
                )
                kv = torch.empty(1)
                indices = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
                topk_length = torch.tensor([2, 1], dtype=torch.int32)
                sinks = torch.arange(heads, dtype=torch.float32)
                calls = []

                def capture(**kwargs):
                    calls.append(kwargs)
                    output = kwargs["q"].new_full((2, expected_heads, 16), 7)
                    return output, None, None

                backend = SimpleNamespace(
                    flashmla_sparse_head_padding=required_padding,
                    _sink_pad_cache={},
                    _call_flash_mla_sparse_fwd=capture,
                )
                output = DeepseekSparseAttnBackend._forward_flashmla_sparse(
                    backend, q, kv, 16, indices, 0.5, topk_length, sinks
                )

                self.assertEqual(len(calls), 1)
                call = calls[0]
                self.assertEqual(tuple(call["q"].shape), (2, expected_heads, head_dim))
                torch.testing.assert_close(call["q"][:, :heads], q)
                self.assertEqual(torch.count_nonzero(call["q"][:, heads:]).item(), 0)
                self.assertEqual(tuple(call["attn_sink"].shape), (expected_heads,))
                torch.testing.assert_close(call["attn_sink"][:heads], sinks)
                self.assertEqual(
                    torch.count_nonzero(call["attn_sink"][heads:]).item(), 0
                )
                torch.testing.assert_close(call["indices"], indices.unsqueeze(1))
                self.assertIs(call["topk_length"], topk_length)
                self.assertIs(call["kv"], kv)
                self.assertEqual(tuple(output.shape), (2, heads, 16))
                self.assertTrue(torch.all(output == 7).item())


if __name__ == "__main__":
    unittest.main()
