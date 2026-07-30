import json
import sys

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.ling3_detector import Ling3Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="weather",
                description="Get weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="date",
                description="Get date",
                parameters={"type": "object", "properties": {}},
            ),
        ),
    ]


def _collect(chunks):
    detector = Ling3Detector()
    calls = []
    for chunk in chunks:
        calls.extend(detector.parse_streaming_increment(chunk, _tools()).calls)
    return detector, calls


@pytest.mark.parametrize(
    ("text", "name", "parameters"),
    [
        (
            "<tool_call>weather\n"
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
            "</tool_call>",
            "weather",
            {"city": "Beijing"},
        ),
        (
            "<tool_call>weather"
            "<arg_key>city</arg_key><arg_value>Shanghai</arg_value>"
            "</tool_call>",
            "weather",
            {"city": "Shanghai"},
        ),
        ("<tool_call>date</tool_call>", "date", {}),
    ],
)
def test_non_streaming_ling3_tool_calls(text, name, parameters):
    result = Ling3Detector().detect_and_parse(text, _tools())

    assert len(result.calls) == 1
    assert result.calls[0].name == name
    assert json.loads(result.calls[0].parameters) == parameters


@pytest.mark.parametrize(
    "chunks",
    [
        ["<tool_call>date", "</tool_call>"],
        ["<tool_call>date</tool_call>"],
    ],
)
def test_streaming_empty_arguments_emit_one_object(chunks):
    detector, calls = _collect(chunks)

    assert [(call.name, call.parameters) for call in calls] == [
        ("date", ""),
        (None, "{}"),
    ]
    assert detector.streamed_args_for_tool == ["{}"]


def test_streaming_compact_arguments():
    detector, calls = _collect(
        [
            "<tool_call>weather",
            "<arg_key>city</arg_key><arg_value>Shanghai</arg_value>",
            "</tool_call>",
        ]
    )

    assert calls[0].name == "weather"
    assert json.loads("".join(call.parameters for call in calls)) == {
        "city": "Shanghai"
    }
    assert json.loads(detector.streamed_args_for_tool[0]) == {"city": "Shanghai"}


def test_parser_registration():
    parser = FunctionCallParser(_tools(), "ling3")

    assert isinstance(parser.detector, Ling3Detector)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
