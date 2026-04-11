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


class TestTemplateDetectionRuleMatrix(unittest.TestCase):
    """Table-driven tests for REASONING_PARSER_RULES and REASONING_MODE_RULES."""

    def setUp(self):
        self.manager = TemplateManager()

    def _detect(self, template, vocab=None):
        if vocab is None:
            vocab = []
        force, config = self.manager._detect_reasoning_pattern(template)
        self.manager._force_reasoning = force
        self.manager._reasoning_config = config
        parser = self.manager._detect_reasoning_parser(template, _DummyTokenizer(vocab))
        return force, config, parser

    PARSER_RULES_MATRIX = [
        # (name, template_snippet, vocab, expected_parser, expected_toggle_param)
        (
            "deepseek_r1_force",
            "{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}\n"
            '{% if "<think>" in content %}<think>',
            [],
            "deepseek-r1",
            None,  # force_reasoning pattern has special_case="always"
        ),
        (
            "deepseek_v3",
            "{% set thinking = thinking if thinking is defined else false %}\n"
            "<think>",
            [],
            "deepseek-v3",
            "thinking",
        ),
        (
            "qwen3_enable_thinking_true",
            "{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}\n",
            [],
            "qwen3",
            "enable_thinking",
        ),
        (
            "kimi_unicode_markers",
            "\u25c1think\u25b7some text\u25c1/think\u25b7",
            [],
            "kimi",
            None,
        ),
        (
            "mistral_reasoning_effort",
            "{% if reasoning_effort %}[THINK]{% endif %}",
            [],
            "mistral",
            None,  # special_case="mistral"
        ),
        (
            "gpt_oss_channel",
            "<|channel|>analysis<|message|>",
            [],
            "gpt-oss",
            None,  # special_case="always"
        ),
        (
            "kimi_k2_with_tool_vocab",
            "{% set thinking = thinking if thinking is defined else true %}\n<think>",
            ["<|tool_calls_section_begin|>", "<|tool_calls_section_end|>"],
            "kimi_k2",
            "thinking",
        ),
        (
            "mimo_enable_thinking_false",
            "{% if enable_thinking is defined and enable_thinking == true %}\n"
            "{% else %}{% set enable_thinking = false %}{% endif %}",
            [],
            "mimo",
            "enable_thinking",
        ),
    ]

    def test_parser_rules_matrix(self):
        for (
            name,
            template,
            vocab,
            expected_parser,
            expected_toggle,
        ) in self.PARSER_RULES_MATRIX:
            with self.subTest(name=name):
                _, config, parser = self._detect(template, vocab)
                self.assertEqual(
                    parser,
                    expected_parser,
                    f"Rule '{name}': expected parser '{expected_parser}', got '{parser}'",
                )
                if expected_toggle is not None:
                    self.assertIsNotNone(
                        config, f"Rule '{name}': expected config, got None"
                    )
                    self.assertEqual(
                        config.toggle_param,
                        expected_toggle,
                        f"Rule '{name}': expected toggle '{expected_toggle}', "
                        f"got '{config.toggle_param}'",
                    )

    def test_unrecognized_template_returns_none(self):
        template = "Hello {{ user_message }}, how can I help you?"
        _, config, parser = self._detect(template)

        self.assertIsNone(config)
        self.assertIsNone(parser)

    def test_empty_template_returns_none(self):
        _, config, parser = self._detect("")

        self.assertIsNone(config)
        self.assertIsNone(parser)

    def test_qwen3_precedence_over_deepseek_r1(self):
        """Template with enable_thinking=true but no <think> tag should be qwen3, not deepseek_r1."""
        template = "{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}"
        _, config, parser = self._detect(template)

        self.assertEqual(parser, "qwen3")
        self.assertEqual(config.toggle_param, "enable_thinking")
        self.assertTrue(config.default_enabled)


if __name__ == "__main__":
    unittest.main()
