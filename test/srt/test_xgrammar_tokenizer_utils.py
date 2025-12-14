import pytest

pytest.importorskip("transformers")
pytest.importorskip("xgrammar")

from transformers import PreTrainedTokenizerBase  # noqa: E402

from sglang.srt.constrained.xgrammar_backend import _unwrap_hf_tokenizer  # noqa: E402


class DummyTokenizer(PreTrainedTokenizerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def test_unwrap_returns_direct_hf_tokenizer():
    dummy = DummyTokenizer()
    assert _unwrap_hf_tokenizer(dummy) is dummy


def test_unwrap_resolves_wrapped_tokenizer():
    dummy = DummyTokenizer()

    class Wrapper:
        def __init__(self, tok):
            self.tokenizer = tok

    wrapper = Wrapper(dummy)
    assert _unwrap_hf_tokenizer(wrapper) is dummy


def test_unwrap_returns_none_when_missing_tokenizer():
    class NoTokenizer:
        pass

    assert _unwrap_hf_tokenizer(NoTokenizer()) is None
