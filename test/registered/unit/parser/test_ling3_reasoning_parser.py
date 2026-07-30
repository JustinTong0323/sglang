import sys

import pytest

from sglang.srt.parser.reasoning_parser import Ling3Detector, ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_ling3_reasoning_defaults_on_and_forces_content():
    detector = Ling3Detector()
    result = detector.detect_and_parse("<think>answer</think>")

    assert detector.reasoning_default == "enable_thinking"
    assert detector._force_nonempty_content
    assert result.reasoning_text == ""
    assert result.normal_text == "answer"


def test_ling3_preserves_reasoning_when_content_exists():
    result = Ling3Detector().detect_and_parse("<think>why</think>answer")

    assert result.reasoning_text == "why"
    assert result.normal_text == "answer"


def test_ling3_tool_call_ends_reasoning():
    result = Ling3Detector().detect_and_parse(
        "<think>need tool<tool_call>date</tool_call>"
    )

    assert result.reasoning_text == "need tool"
    assert result.normal_text == "<tool_call>date</tool_call>"


def test_ling3_force_nonempty_can_be_disabled():
    result = Ling3Detector(force_nonempty_content=False).detect_and_parse(
        "<think>answer</think>"
    )

    assert result.reasoning_text == "answer"
    assert result.normal_text == ""


def test_reasoning_parser_registration():
    assert isinstance(ReasoningParser("ling3").detector, Ling3Detector)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
