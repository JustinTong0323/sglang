import re

from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector


class Ling3Detector(Glm4MoeDetector):
    _STREAMING_PARTIAL_PATTERN = re.compile(
        r"<tool_call>\s*(.*?)(?:(?:\\n|\n)\s*|(?=<arg_key>)|(?=</tool_call>))"
        r"(.*?)(</tool_call>|$)",
        re.DOTALL,
    )

    def __init__(self):
        super().__init__()
        self.func_detail_regex = re.compile(
            r"<tool_call>\s*(.*?)(?:(?:\\n|\n)\s*|(?=<arg_key>)|(?=</tool_call>))"
            r"(<arg_key>.*?)?</tool_call>",
            re.DOTALL,
        )
