import unittest

from sglang.srt.managers.template_manager import (
    ReasoningToggleConfig,
    TemplateManager,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(2.0, "stage-a-test-cpu")


class _DummyTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return {token: i for i, token in enumerate(self._vocab)}


class TestTemplateManagerReasoningDetection(unittest.TestCase):
    def setUp(self):
        self.manager = TemplateManager()

    def _detect(self, template, vocab):
        force, config = self.manager._detect_reasoning_pattern(template)
        self.manager._force_reasoning = force
        self.manager._reasoning_config = config
        parser = self.manager._detect_reasoning_parser(template, _DummyTokenizer(vocab))
        return force, config, parser

    def test_qwen3_template_not_misclassified_as_glm45(self):
        template = """
        {% set enable_thinking = enable_thinking if enable_thinking is defined else true %}
        {% if '</think>' in content %}
        <tool_call>
        """
        _, config, parser = self._detect(
            template, ["<tool_call>", "<|endoftext|>", "</think>"]
        )

        self.assertEqual(
            config,
            ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True),
        )
        self.assertEqual(parser, "qwen3")

    def test_glm45_requires_glm_specific_template_markers(self):
        template = """
        [gMASK]<sop>
        {% set enable_thinking = enable_thinking if enable_thinking is defined else true %}
        /nothink
        <tool_call>
        """
        _, config, parser = self._detect(
            template, ["<tool_call>", "<|endoftext|>", "<|user|>"]
        )

        self.assertEqual(
            config,
            ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True),
        )
        self.assertEqual(parser, "glm45")

    def test_interns1_detects_enable_thinking_default_true(self):
        template = """
        {% set default_thinking_sys %}...<think>...</think>{% endset %}
        {% if enable_thinking is not defined or enable_thinking %}
        """
        _, config, parser = self._detect(template, ["<|endoftext|>"])

        self.assertEqual(
            config,
            ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True),
        )
        self.assertEqual(parser, "interns1")

    def test_nemotron_detects_uppercase_true_assignment(self):
        template = """
        {% set enable_thinking = enable_thinking if enable_thinking is defined else True %}
        {% set truncate_history_thinking = truncate_history_thinking if truncate_history_thinking is defined else True %}
        """
        _, config, parser = self._detect(template, ["<|endoftext|>"])

        self.assertEqual(
            config,
            ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True),
        )
        self.assertEqual(parser, "nemotron_3")

    def test_minimax_uses_template_signature_without_toggle_config(self):
        template = """
        {%- set toolcall_begin_token = '<minimax:tool_call>' -%}
        """
        _, config, parser = self._detect(template, ["<minimax:tool_call>"])

        self.assertIsNone(config)
        self.assertEqual(parser, "minimax")


if __name__ == "__main__":
    unittest.main()
