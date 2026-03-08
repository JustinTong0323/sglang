import json
import logging
import re
from typing import Any, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

# Gemma4 special tokens for tool calls
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
STRING_DELIM = '<|"|>'


def _parse_gemma4_value(value_str: str) -> object:
    """Parse a single Gemma4 value (after key:) into a Python object."""
    value_str = value_str.strip()
    if not value_str:
        return value_str

    # Boolean
    if value_str == "true":
        return True
    if value_str == "false":
        return False

    # Number (int or float)
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # Bare string (no <|"|> delimiters)
    return value_str


def _parse_gemma4_array(arr_str: str) -> list:
    """Parse a Gemma4 array content string into a Python list."""
    items: list = []
    i = 0
    n = len(arr_str)

    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # String element
        if arr_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + len(STRING_DELIM)

        # Nested object
        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + len(STRING_DELIM) if nd != -1 else n
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            items.append(_parse_gemma4_args(arr_str[obj_start : i - 1]))

        # Nested array
        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            items.append(_parse_gemma4_array(arr_str[sub_start : i - 1]))

        # Bare value
        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            items.append(_parse_gemma4_value(arr_str[val_start:i]))

    return items


def _parse_gemma4_args(args_str: str) -> dict:
    """Parse Gemma4's custom key:value format into a Python dict."""
    if not args_str or not args_str.strip():
        return {}

    result: dict = {}
    i = 0
    n = len(args_str)

    while i < n:
        # Skip whitespace and commas
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # Parse key (unquoted, ends at ':')
        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        i += 1  # skip ':'

        # Parse value
        if i >= n:
            result[key] = ""
            break

        # Skip whitespace after ':'
        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            result[key] = ""
            break

        # String value: <|"|>...<|"|>
        if args_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                # Unterminated string — take rest
                result[key] = args_str[val_start:]
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + len(STRING_DELIM)

        # Nested object: {...}
        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    # Skip over string contents
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    if next_delim == -1:
                        i = n
                    else:
                        i = next_delim + len(STRING_DELIM)
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            result[key] = _parse_gemma4_args(args_str[obj_start : i - 1])

        # Array: [...]
        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    if next_delim == -1:
                        i = n
                    else:
                        i = next_delim + len(STRING_DELIM)
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            arr_content = args_str[arr_start : i - 1]
            result[key] = _parse_gemma4_array(arr_content)

        # Bare value (number, boolean, etc.)
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result


class Gemma4Detector(BaseFormatDetector):
    def __init__(self):
        super().__init__()
        self.tool_call_start_token = TOOL_CALL_START
        self.tool_call_end_token = TOOL_CALL_END
        self.tool_call_regex = re.compile(
            r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>",
            re.DOTALL,
        )

        # Streaming state
        self.parsed_pos: int = 0
        self.is_inside_tool_call: bool = False
        self.current_func_name: Optional[str] = None
        self.json_started: bool = False

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if self.tool_call_start_token not in text:
            return StreamingParseResult(normal_text=text)

        calls = []
        try:
            matches = self.tool_call_regex.findall(text)
            if not matches:
                return StreamingParseResult(normal_text=text)

            tool_indices = self._get_tool_indices(tools)
            for func_name, args_str in matches:
                arguments = _parse_gemma4_args(args_str)
                calls.append(
                    ToolCallItem(
                        tool_index=tool_indices.get(func_name, -1),
                        name=func_name,
                        parameters=json.dumps(arguments, ensure_ascii=False),
                    )
                )

            # Content = text before first tool call
            content_end = text.find(self.tool_call_start_token)
            normal_text = text[:content_end] if content_end > 0 else ""

            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text

        if not self._buffer:
            return StreamingParseResult()

        calls = []
        normal_text_chunks = []

        while True:
            current_slice = self._buffer[self.parsed_pos :]
            if not current_slice:
                break

            if not self.is_inside_tool_call:
                # Step 4: Outside tool call block
                next_start = current_slice.find(self.tool_call_start_token)
                if next_start == -1:
                    # Check for partial match at the end
                    partial_len = self._ends_with_partial_token(current_slice, self.tool_call_start_token)
                    if partial_len > 0:
                        text_to_append = current_slice[:-partial_len]
                        if text_to_append:
                            normal_text_chunks.append(text_to_append)
                        self.parsed_pos += len(text_to_append)
                        break
                    else:
                        normal_text_chunks.append(current_slice)
                        self.parsed_pos += len(current_slice)
                        continue
                elif next_start == 0:
                    self.parsed_pos += len(self.tool_call_start_token)
                    self.is_inside_tool_call = True
                    continue
                else:
                    normal_text_chunks.append(current_slice[:next_start])
                    self.parsed_pos += next_start
                    continue
            else:
                # Inside tool call block
                
                # Check for TOOL_CALL_END first
                if current_slice.startswith(self.tool_call_end_token):
                    self.parsed_pos += len(self.tool_call_end_token)
                    self.is_inside_tool_call = False
                    self.current_func_name = None
                    continue
                
                if not self.current_func_name:
                    # Skip leading whitespace
                    if current_slice[0] in (" ", "\n", "\t"):
                        self.parsed_pos += 1
                        continue

                    if current_slice.startswith("call:"):
                        brace_pos = current_slice.find("{")
                        if brace_pos != -1:
                            func_name = current_slice[5:brace_pos]
                            self.current_tool_id += 1
                            self.current_func_name = func_name
                            self.current_tool_name_sent = True

                            tool_indices = self._get_tool_indices(tools)
                            calls.append(
                                ToolCallItem(
                                    tool_index=tool_indices.get(func_name, -1),
                                    name=func_name,
                                    parameters="",
                                )
                            )
                            self.parsed_pos += brace_pos + 1
                            continue
                        else:
                            # Incomplete call:name{
                            break
                    else:
                        # Check for partial matches
                        if "call:".startswith(current_slice) or self.tool_call_end_token.startswith(current_slice):
                            break
                        
                        # Unexpected content, skip
                        self.parsed_pos += 1
                        continue
                else:
                    # Parsing arguments (looking for balancing })
                    depth = 1
                    i = 0
                    n = len(current_slice)
                    found = False
                    while i < n:
                        if current_slice[i : i + len(STRING_DELIM)] == STRING_DELIM:
                            i += len(STRING_DELIM)
                            next_delim = current_slice.find(STRING_DELIM, i)
                            if next_delim == -1:
                                i = n # Force wait
                                break
                            i = next_delim + len(STRING_DELIM)
                            continue

                        if current_slice[i] == "{":
                            depth += 1
                        elif current_slice[i] == "}":
                            depth -= 1
                            if depth == 0:
                                args_str = current_slice[:i]
                                arguments = _parse_gemma4_args(args_str)

                                tool_indices = self._get_tool_indices(tools)
                                calls.append(
                                    ToolCallItem(
                                        tool_index=tool_indices.get(
                                            self.current_func_name, -1
                                        ),
                                        parameters=json.dumps(
                                            arguments, ensure_ascii=False
                                        ),
                                    )
                                )
                                self.parsed_pos += i + 1
                                self.current_func_name = None  # Reset for next call:
                                found = True
                                break
                        i += 1

                    if found:
                        continue
                    else:
                        # Incomplete arguments block
                        break

        if self.parsed_pos > 0:
            self._buffer = self._buffer[self.parsed_pos :]
            self.parsed_pos = 0

        normal_text = "".join(normal_text_chunks) if normal_text_chunks else ""
        return StreamingParseResult(calls=calls, normal_text=normal_text)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError
