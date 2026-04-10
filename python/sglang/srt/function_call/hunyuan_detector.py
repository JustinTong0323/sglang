import json
import logging
import re
from typing import Any, Dict, List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


class HunyuanDetector(BaseFormatDetector):
    """
    Detector for Hunyuan tool call format.

    Format:
        <tool_calls>
        <tool_call>function_name<tool_sep>
        <arg_key>key1</arg_key>
        <arg_value>value1</arg_value>
        </tool_call>
        </tool_calls>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_calls>"
        self.eot_token = "</tool_calls>"

        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_sep_token = "<tool_sep>"

        self.tool_call_regex = re.compile(
            rf"<tool_call>(.*?)<tool_sep>(.*?)</tool_call>",
            re.DOTALL,
        )

        self.func_args_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )

        self._in_tool_calls = False
        self._normal_text_emitted = False

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    @staticmethod
    def _build_param_types(tools: List[Tool]) -> Dict[str, Dict[str, str]]:
        """Build a map of function_name -> {param_name -> type} for all tools."""
        result = {}
        for tool in tools:
            if tool.function.name and tool.function.parameters:
                props = tool.function.parameters.get("properties", {})
                result[tool.function.name] = {
                    k: v.get("type") for k, v in props.items()
                }
        return result

    def _parse_tool_call_block(
        self,
        name: str,
        args_text: str,
        tool_indices: Dict[str, int],
        param_types_map: Dict[str, Dict[str, str]],
    ) -> List[ToolCallItem]:
        """Parse a single tool call block into ToolCallItems."""
        function_name = name.strip()
        arg_pairs = self.func_args_regex.findall(args_text)

        param_types = param_types_map.get(function_name, {})
        arg_dict = {}
        for key, value in arg_pairs:
            key = key.strip()
            if param_types.get(key) != "string":
                arg_dict[key] = self._try_deserialize(value)
            else:
                arg_dict[key] = value

        if function_name not in tool_indices:
            if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                logger.warning(
                    f"Model attempted to call undefined function: {function_name}"
                )
                return []

        return [
            ToolCallItem(
                tool_index=tool_indices.get(function_name, -1),
                name=function_name,
                parameters=json.dumps(arg_dict, ensure_ascii=False),
            )
        ]

    @staticmethod
    def _try_deserialize(value: str) -> Any:
        """Try to deserialize a value as JSON, fall back to string."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value

    def _ensure_tool_cache(self, tools: List[Tool]) -> None:
        """Build and cache tool indices and param types on first use."""
        if not hasattr(self, "_cached_tool_indices"):
            self._cached_tool_indices = self._get_tool_indices(tools)
            self._cached_param_types = self._build_param_types(tools)

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx > 0 else ""

        self._ensure_tool_cache(tools)
        calls = []
        try:
            for match in self.tool_call_regex.findall(text):
                function_name, function_args = match
                calls.extend(
                    self._parse_tool_call_block(
                        function_name,
                        function_args,
                        self._cached_tool_indices,
                        self._cached_param_types,
                    )
                )
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}", exc_info=True)
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Buffers content until complete <tool_call>...</tool_call> blocks are found,
        then extracts and emits tool calls.
        """
        try:
            self._buffer += new_text
            current_text = self._buffer

            if not self._in_tool_calls and self.bot_token not in current_text:
                partial_len = self._ends_with_partial_token(
                    current_text, self.bot_token
                )
                if partial_len:
                    safe_text = current_text[:-partial_len]
                    self._buffer = current_text[-partial_len:]
                    return StreamingParseResult(normal_text=safe_text)
                else:
                    self._buffer = ""
                    return StreamingParseResult(normal_text=current_text)

            if not self._in_tool_calls:
                bot_pos = current_text.find(self.bot_token)
                if bot_pos > 0 and not self._normal_text_emitted:
                    normal_text = current_text[:bot_pos]
                    self._buffer = current_text[bot_pos:]
                    self._normal_text_emitted = True
                    return StreamingParseResult(normal_text=normal_text)
                self._in_tool_calls = True

            self._ensure_tool_cache(tools)
            calls = []
            search_start = 0
            last_end = 0

            while True:
                tc_start = current_text.find(self.tool_call_start_token, search_start)
                if tc_start == -1:
                    break
                tc_end = current_text.find(self.tool_call_end_token, tc_start)
                if tc_end == -1:
                    self._buffer = current_text[tc_start:]
                    if calls:
                        return StreamingParseResult(calls=calls)
                    return StreamingParseResult()

                block_end = tc_end + len(self.tool_call_end_token)
                block = current_text[tc_start:block_end]

                match = self.tool_call_regex.search(block)
                if match:
                    function_name, function_args = match.groups()
                    self.current_tool_id += 1
                    parsed_calls = self._parse_tool_call_block(
                        function_name,
                        function_args,
                        self._cached_tool_indices,
                        self._cached_param_types,
                    )
                    for call in parsed_calls:
                        # Override with sequential ID for the streaming layer
                        call.tool_index = self.current_tool_id
                        calls.append(call)
                else:
                    logger.warning(
                        "Tool call block did not match expected format: %s",
                        block[:200],
                    )

                last_end = block_end
                search_start = block_end

            if calls:
                self._buffer = current_text[last_end:]
                return StreamingParseResult(calls=calls)

            return StreamingParseResult()
        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}", exc_info=True)
            return StreamingParseResult()

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f"<tool_calls>\n<tool_call>{name}<tool_sep>",
            end="</tool_call>\n</tool_calls>",
            trigger="<tool_calls>",
        )

    def supports_structural_tag(self) -> bool:
        return False
