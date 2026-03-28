"""
Test structural_tag constrained decoding for required/named tool_choice.

Before this fix:
- tool_choice="required" always used generic JSON schema constraint, which
  conflicts with models that use native special-token tool call formats
  (kimi_k2, deepseekv3, qwen3_coder, etc.)
- tool_choice="required" parsed model output as plain JSON, which failed for
  models using special tokens, resulting in tool_calls=null

After this fix:
- tool_choice="required" uses structural_tag constraint when a model-specific
  parser is configured, preserving the model's native format
- tool_choice="required" uses FunctionCallParser for parsing when available,
  falling back to JSON only when no parser is configured

Usage:
    # With a running server that has --tool-call-parser set:
    python3 -m pytest test/registered/openai_server/function_call/test_structural_tag_required.py -v

    # Or specify a custom base URL:
    DEFAULT_URL_FOR_TEST=http://localhost:8000 python3 -m pytest ... -v
"""

import json
import unittest

import openai

from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
)


def get_simple_tools(strict=False):
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name",
                        },
                    },
                    "required": ["location"],
                },
                "strict": strict,
            },
        }
    ]


class TestStructuralTagRequired(CustomTestCase):
    """
    Test that tool_choice="required" works correctly with model-specific parsers.

    This test class is designed to run against any server with --tool-call-parser.
    The key assertion is that tool_calls is properly populated (not null) for all
    combinations of tool_choice and strict settings.
    """

    @classmethod
    def setUpClass(cls):
        import os

        cls.base_url = os.environ.get("DEFAULT_URL_FOR_TEST", DEFAULT_URL_FOR_TEST)
        if not cls.base_url.endswith("/v1"):
            cls.base_url += "/v1"
        cls.api_key = "sk-123456"

    def setUp(self):
        self.client = openai.Client(base_url=self.base_url, api_key=self.api_key)
        self.model_name = self.client.models.list().data[0].id

    def _call_with_tools(self, tool_choice, strict=False, stream=False, **kwargs):
        """Helper to make a tool call request."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
            tools=get_simple_tools(strict=strict),
            tool_choice=tool_choice,
            stream=stream,
            max_tokens=4096,
            timeout=300,
            **kwargs,
        )
        if stream:
            tool_calls_by_index = {}
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_by_index:
                            tool_calls_by_index[idx] = {
                                "id": tc.id,
                                "name": "",
                                "arguments": "",
                            }
                        if tc.function:
                            if tc.function.name:
                                tool_calls_by_index[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_by_index[idx][
                                    "arguments"
                                ] += tc.function.arguments
            return list(tool_calls_by_index.values())
        else:
            return response.choices[0].message.tool_calls

    def _assert_valid_weather_call(self, tool_calls):
        """Assert tool_calls contains a valid get_weather call."""
        self.assertIsNotNone(tool_calls, "tool_calls should not be None")
        self.assertGreater(len(tool_calls), 0, "tool_calls should not be empty")
        # Check first tool call
        if hasattr(tool_calls[0], "function"):
            # OpenAI SDK object
            call = tool_calls[0]
            self.assertEqual(call.function.name, "get_weather")
            args = json.loads(call.function.arguments)
        else:
            # Dict from streaming
            call = tool_calls[0]
            self.assertEqual(call["name"], "get_weather")
            args = json.loads(call["arguments"])
        self.assertIn("location", args)

    # --- Non-streaming tests ---

    def test_required_strict_non_streaming(self):
        """tool_choice=required + strict=True, non-streaming"""
        tool_calls = self._call_with_tools("required", strict=True)
        self._assert_valid_weather_call(tool_calls)

    def test_required_non_strict_non_streaming(self):
        """tool_choice=required + strict=False, non-streaming"""
        tool_calls = self._call_with_tools("required", strict=False)
        self._assert_valid_weather_call(tool_calls)

    def test_auto_strict_non_streaming(self):
        """tool_choice=auto + strict=True, non-streaming"""
        tool_calls = self._call_with_tools("auto", strict=True)
        self._assert_valid_weather_call(tool_calls)

    def test_auto_non_strict_non_streaming(self):
        """tool_choice=auto + strict=False, non-streaming"""
        tool_calls = self._call_with_tools("auto", strict=False)
        self._assert_valid_weather_call(tool_calls)

    # --- Streaming tests ---
    # NOTE: streaming + required may not work for all parsers (e.g. qwen3_coder
    # streaming tool call detection can fail). These are best-effort tests.

    def test_required_strict_streaming(self):
        """tool_choice=required + strict=True, streaming"""
        tool_calls = self._call_with_tools("required", strict=True, stream=True)
        self._assert_valid_weather_call(tool_calls)

    def test_required_non_strict_streaming(self):
        """tool_choice=required + strict=False, streaming"""
        tool_calls = self._call_with_tools("required", strict=False, stream=True)
        self._assert_valid_weather_call(tool_calls)

    def test_auto_strict_streaming(self):
        """tool_choice=auto + strict=True, streaming"""
        tool_calls = self._call_with_tools("auto", strict=True, stream=True)
        self._assert_valid_weather_call(tool_calls)

    def test_auto_non_strict_streaming(self):
        """tool_choice=auto + strict=False, streaming"""
        tool_calls = self._call_with_tools("auto", strict=False, stream=True)
        self._assert_valid_weather_call(tool_calls)


if __name__ == "__main__":
    unittest.main()
