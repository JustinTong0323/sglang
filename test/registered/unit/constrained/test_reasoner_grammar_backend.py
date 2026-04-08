import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarBackend
from sglang.srt.constrained.reasoner_grammar_backend import (
    ReasonerGrammarBackend,
    ReasonerGrammarObject,
)
from sglang.srt.constrained.torch_ops.token_filter_torch_ops import (
    set_token_filter_torch,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(2.0, "stage-a-test-cpu")


class _DummyTokenizer:
    def __init__(self, token_map):
        self._token_map = token_map

    def encode(self, text, add_special_tokens=False):
        return list(self._token_map.get(text, []))


class _DummyGrammarBackend(BaseGrammarBackend):
    def __init__(self, support_token_filter=True):
        super().__init__()
        self._support_token_filter = support_token_filter
        self._dispatch_result = None

    @property
    def is_support_token_filter(self):
        return self._support_token_filter

    @staticmethod
    def allocate_vocab_mask(vocab_size, batch_size, device):
        return torch.zeros((batch_size, (vocab_size + 31) // 32), dtype=torch.int32)

    @staticmethod
    def move_vocab_mask(vocab_mask, device):
        return vocab_mask

    @staticmethod
    def apply_vocab_mask(logits, vocab_mask):
        return None

    @staticmethod
    def set_token_filter(vocab_mask, token_ids, batch_idx, is_allowed=True):
        set_token_filter_torch(vocab_mask, token_ids, batch_idx, is_allowed)

    def _init_value_dispatch(self, key, reasoning):
        return self._dispatch_result


def _allowed_token_ids(vocab_mask, token_ids):
    allowed = []
    for token_id in token_ids:
        elem = token_id // 32
        bit = token_id % 32
        if int(vocab_mask[0, elem].item()) & (1 << bit):
            allowed.append(token_id)
    return allowed


class TestReasonerGrammarObject(unittest.TestCase):
    def _make_strict_object(self):
        return ReasonerGrammarObject(
            grammar=None,
            think_end_id=7,
            think_excluded_token_ids=[3, 5],
            max_think_tokens=2,
            enable_token_filter=True,
            token_filter_fn=set_token_filter_torch,
            allocate_vocab_mask_fn=lambda vocab_size, batch_size, device: torch.zeros(
                (batch_size, (vocab_size + 31) // 32), dtype=torch.int32
            ),
            move_vocab_mask_fn=lambda vocab_mask, device: vocab_mask,
            apply_vocab_mask_fn=lambda logits, vocab_mask: None,
        )

    def test_strict_thinking_phase_excludes_configured_tokens(self):
        obj = self._make_strict_object()
        obj.maybe_init_reasoning(True)
        mask = obj.allocate_vocab_mask(64, 1, "cpu")

        obj.fill_vocab_mask(mask, 0)

        allowed = _allowed_token_ids(mask, [0, 1, 3, 5, 7, 8])
        self.assertEqual(allowed, [0, 1, 7, 8])

    def test_budget_exhaustion_allows_only_think_end(self):
        obj = self._make_strict_object()
        obj.maybe_init_reasoning(True)
        obj.accept_token(10)
        obj.accept_token(11)
        mask = obj.allocate_vocab_mask(64, 1, "cpu")

        obj.fill_vocab_mask(mask, 0)

        allowed = _allowed_token_ids(mask, [0, 1, 3, 5, 7, 8, 10, 11])
        self.assertEqual(allowed, [7])

    def test_strict_only_wrapper_exposes_backend_mask_hooks(self):
        obj = self._make_strict_object()
        mask = obj.allocate_vocab_mask(64, 2, "cpu")

        self.assertEqual(mask.shape, (2, 2))
        self.assertIs(obj.move_vocab_mask(mask, "cpu"), mask)
        self.assertIsNotNone(obj.apply_vocab_mask)


class TestReasonerGrammarBackend(unittest.TestCase):
    def setUp(self):
        self._prev_budget = os.environ.get("SGLANG_MAX_THINK_TOKENS")

    def tearDown(self):
        if self._prev_budget is None:
            os.environ.pop("SGLANG_MAX_THINK_TOKENS", None)
        else:
            os.environ["SGLANG_MAX_THINK_TOKENS"] = self._prev_budget

    def _make_parser(self, strict=True):
        detector = SimpleNamespace(
            think_start_token="<think>",
            think_end_token="</think>",
            strict_reasoning_format=strict,
            think_excluded_tokens=["<tool_call>", "</tool_call>"],
        )
        return SimpleNamespace(detector=detector)

    def _make_tokenizer(self, start_ids=None, end_ids=None):
        return _DummyTokenizer(
            {
                "<think>": [1] if start_ids is None else start_ids,
                "</think>": [2] if end_ids is None else end_ids,
                "<tool_call>": [3],
                "</tool_call>": [4],
            }
        )

    def test_init_strict_reasoning_grammar_uses_token_filter_and_budget(self):
        os.environ["SGLANG_MAX_THINK_TOKENS"] = "2"
        backend = _DummyGrammarBackend(support_token_filter=True)
        reasoner = ReasonerGrammarBackend(
            backend, self._make_parser(strict=True), self._make_tokenizer()
        )

        obj = reasoner.init_strict_reasoning_grammar(reasoning=True)

        self.assertIsInstance(obj, ReasonerGrammarObject)
        self.assertTrue(obj.enable_token_filter)
        self.assertEqual(obj.max_think_tokens, 2)
        self.assertEqual(obj.think_excluded_token_ids, [3, 4])

    def test_init_strict_reasoning_grammar_none_for_non_strict_parser(self):
        backend = _DummyGrammarBackend(support_token_filter=True)
        reasoner = ReasonerGrammarBackend(
            backend, self._make_parser(strict=False), self._make_tokenizer()
        )

        self.assertIsNone(reasoner.init_strict_reasoning_grammar(reasoning=True))

    def test_wraps_inner_grammar_with_reasoning_state_machine(self):
        os.environ["SGLANG_MAX_THINK_TOKENS"] = "1"
        backend = _DummyGrammarBackend(support_token_filter=True)
        inner_grammar = MagicMock()
        backend._dispatch_result = inner_grammar
        reasoner = ReasonerGrammarBackend(
            backend, self._make_parser(strict=True), self._make_tokenizer()
        )

        wrapped = reasoner._init_value_dispatch(("json", "{}"), reasoning=True)
        self.assertIsInstance(wrapped, ReasonerGrammarObject)
        wrapped.accept_token(10)
        inner_grammar.accept_token.assert_not_called()
        wrapped.accept_token(2)
        wrapped.accept_token(42)
        inner_grammar.accept_token.assert_called_once_with(42)

    def test_rejects_multi_token_think_start_marker(self):
        backend = _DummyGrammarBackend(support_token_filter=True)

        with self.assertRaisesRegex(ValueError, "must encode to exactly one token"):
            ReasonerGrammarBackend(
                backend,
                self._make_parser(strict=True),
                self._make_tokenizer(start_ids=[1, 2]),
            )

    def test_rejects_multi_token_think_end_marker(self):
        backend = _DummyGrammarBackend(support_token_filter=True)

        with self.assertRaisesRegex(ValueError, "must encode to exactly one token"):
            ReasonerGrammarBackend(
                backend,
                self._make_parser(strict=True),
                self._make_tokenizer(end_ids=[2, 3]),
            )


if __name__ == "__main__":
    unittest.main()
