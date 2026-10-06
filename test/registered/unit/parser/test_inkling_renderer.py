import json
import sys
import unittest
from pathlib import Path
from unittest import mock

from sglang.srt.entrypoints.openai.chat_encoding import encode_simple_chat
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.parser.inkling_output import InklingOutputParser
from sglang.srt.parser.inkling_renderer import (
    TML_RENDERERS_INSTALL_HINT,
    load_tml_renderers,
    render_inkling_assistant_prefix,
    render_inkling_messages,
)
from sglang.srt.parser.inkling_tokenizer import (
    AUDIO_END,
    AUDIO_TOKEN_ID,
    CONTENT_AUDIO_INPUT,
    END_MESSAGE,
    INKLING_SPECIAL_TOKEN_IDS,
    MESSAGE_MODEL,
    MESSAGE_USER,
)
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# tml-renderers ships cp311-abi3 wheels only, so this needs the Python 3.12 lane.
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large-py312")


_GOLDEN = json.loads(
    (Path(__file__).with_name("inkling_tmlv0_golden.json")).read_text()
)


class TestTmlRenderersLoader(unittest.TestCase):
    def test_missing_tml_renderers_names_the_package(self):
        load_tml_renderers.cache_clear()
        self.addCleanup(load_tml_renderers.cache_clear)
        with mock.patch.dict(sys.modules, {"tml_renderers": None}):
            with self.assertRaises(ImportError) as ctx:
                load_tml_renderers()
        self.assertEqual(str(ctx.exception), TML_RENDERERS_INSTALL_HINT)


class TestInklingRenderer(unittest.TestCase):
    def test_prompts_match_tmlv0_reference(self):
        """Golden input_ids were produced by tml-renderers itself (see the
        fixture's ``source``); sglang's message adaptation must not change a
        single token."""
        for case in _GOLDEN["prompts"]:
            with self.subTest(case=case["name"]):
                actual = render_inkling_messages(
                    case["messages"],
                    tools=case["tools"],
                    reasoning_effort=case["reasoning_effort"],
                )
                self.assertEqual(actual, case["input_ids"])

    def test_audio_part_keeps_mm_processor_framing(self):
        """The MM processor expands one AUDIO_TOKEN_ID inside
        <|content_audio_input|> ... <|audio_end|>; tml-renderers cannot render
        audio without DMel-encoding the bytes, so this framing is sglang's."""
        actual = render_inkling_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "", "format": "wav"},
                        }
                    ],
                }
            ]
        )
        self.assertEqual(
            actual[-5:],
            [
                INKLING_SPECIAL_TOKEN_IDS[MESSAGE_USER],
                INKLING_SPECIAL_TOKEN_IDS[CONTENT_AUDIO_INPUT],
                AUDIO_TOKEN_ID,
                INKLING_SPECIAL_TOKEN_IDS[AUDIO_END],
                INKLING_SPECIAL_TOKEN_IDS[END_MESSAGE],
            ],
        )

    def test_audio_outside_user_messages_is_rejected(self):
        """tml-renderers renders AudioPointer only as user input audio; a tool
        or assistant audio part must fail instead of rendering a framing the
        model never saw."""
        audio = {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}}
        for message in (
            {"role": "tool", "tool_call_id": "c", "name": "rec", "content": [audio]},
            {"role": "assistant", "content": [audio]},
        ):
            with self.subTest(role=message["role"]):
                with self.assertRaises(ValueError):
                    render_inkling_messages([{"role": "user", "content": "x"}, message])

    def test_assistant_prefix_text_is_ordinary_tokens(self):
        tokenizer = load_tml_renderers().tokenizer
        prefix = "The answer <|end_message|>"
        self.assertEqual(
            render_inkling_assistant_prefix(prefix),
            [
                INKLING_SPECIAL_TOKEN_IDS[MESSAGE_MODEL],
                INKLING_SPECIAL_TOKEN_IDS["<|content_text|>"],
                *tokenizer.encode_ordinary(prefix),
            ],
        )

    def test_offline_encoder_uses_the_same_inkling_format(self):
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual(
            encode_simple_chat(tokenizer=None, spec="inkling", messages=messages),
            render_inkling_messages(messages),
        )


def _ids(*parts: str | list[int]) -> list[int]:
    tokenizer = load_tml_renderers().tokenizer
    ids: list[int] = []
    for part in parts:
        if isinstance(part, list):
            ids.extend(part)
        elif part.startswith("<|") and part.endswith("|>"):
            ids.append(tokenizer.encode_special(part[2:-2]))
        else:
            ids.extend(tokenizer.encode_ordinary(part))
    return ids


def _batch(token_ids: list[int], **kwargs):
    parser = InklingOutputParser(**kwargs)
    return parser.feed(token_ids).merge(parser.finish())


def _stream(token_ids: list[int], **kwargs):
    parser = InklingOutputParser(**kwargs)
    deltas = [parser.feed([token_id]) for token_id in token_ids]
    deltas.append(parser.finish())
    merged = deltas[0]
    for delta in deltas[1:]:
        merged = merged.merge(delta)
    return merged


class TestInklingOutputParser(unittest.TestCase):
    def test_outputs_match_tmlv0_reference(self):
        """Expected reasoning/content/tool_calls come from tml-renderers'
        own parser; one-shot and per-token parsing must both reproduce them."""
        for case in _GOLDEN["outputs"]:
            for mode, parse in (
                ("batch", _batch),
                ("stream", _stream),
            ):
                with self.subTest(case=case["name"], mode=mode):
                    parsed = parse(
                        case["output_ids"],
                        separate_reasoning=True,
                        parse_tool_calls=True,
                    )
                    self.assertEqual(parsed.reasoning, case["reasoning"])
                    self.assertEqual(parsed.content, case["content"])
                    self.assertEqual(
                        [
                            {"name": call.name, "arguments": call.arguments}
                            for call in parsed.tool_calls
                        ],
                        case["tool_calls"],
                    )
                    self.assertEqual(
                        [call.index for call in parsed.tool_calls],
                        list(range(len(case["tool_calls"]))),
                    )

    def test_unparseable_call_becomes_payload_text(self):
        """A call payload the reference parser rejects (NaN, missing args) must
        not surface as a tool call, and the header tool name must not leak
        into content; parsing resumes at the next message."""
        for payload in ('{"name":"f","args":{"a":NaN}}', '{"name":"f"}'):
            output_ids = _ids(
                "<|message_model|>",
                "f",
                "<|content_invoke_tool_json|>",
                payload,
                "<|end_message|>",
                "<|message_model|>",
                "<|content_text|>",
                "after",
                "<|end_message|>",
                "<|content_model_end_sampling|>",
            )
            for mode, parse in (
                ("batch", _batch),
                ("stream", _stream),
            ):
                with self.subTest(payload=payload, mode=mode):
                    parsed = parse(
                        output_ids, separate_reasoning=True, parse_tool_calls=True
                    )
                    self.assertEqual(parsed.tool_calls, ())
                    self.assertEqual(parsed.content, payload + "after")

    def test_buffered_reasoning_survives_truncation(self):
        """With stream_reasoning=False a thinking block is held until it
        closes; a max_tokens cut inside it must still flush the held text."""
        parser = InklingOutputParser(
            separate_reasoning=True, parse_tool_calls=True, stream_reasoning=False
        )
        held = parser.feed(
            _ids("<|message_model|>", "<|content_thinking|>", "long plan here")
        )
        self.assertEqual(held.reasoning, "")
        self.assertEqual(parser.finish().reasoning, "long plan here")

    def test_disabled_parsers_route_into_content(self):
        output_ids = _ids(
            "<|message_model|>",
            "<|content_thinking|>",
            "plan",
            "<|end_message|>",
            "<|message_model|>",
            "f",
            "<|content_invoke_tool_json|>",
            '{"name":"f","args":{}}',
            "<|end_message|>",
            "<|content_model_end_sampling|>",
        )
        parsed = _batch(output_ids, separate_reasoning=False, parse_tool_calls=False)
        self.assertEqual(parsed.reasoning, "")
        self.assertEqual(parsed.tool_calls, ())
        self.assertEqual(parsed.content, 'plan{"name":"f","args":{}}')


class TestInklingServingPrompt(unittest.TestCase):
    def test_serving_does_not_prefill_model_message(self):
        from sglang.srt.parser.inkling_tokenizer import INKLING_SPECIAL_TOKEN_IDS

        serving = object.__new__(OpenAIServingChat)
        serving.chat_encoding_spec = "inkling"
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            reasoning_effort=0.5,
        )
        prompt_ids = serving._encode_messages(
            [message.model_dump() for message in request.messages],
            request,
            thinking_mode=None,
        )
        self.assertEqual(prompt_ids[-1], INKLING_SPECIAL_TOKEN_IDS["<|end_message|>"])

    def test_continue_final_message_resumes_open_model_text_block(self):
        """Bug regression: continue_final_message was silently ignored on the
        inkling path — the trailing assistant message rendered as a CLOSED
        historical turn (<|end_message|> + <|content_model_end_sampling|>), so
        the model started a fresh turn instead of continuing. The prefix must
        render as an OPEN model text block."""
        from sglang.srt.parser.inkling_tokenizer import INKLING_SPECIAL_TOKEN_IDS

        serving = object.__new__(OpenAIServingChat)
        serving.chat_encoding_spec = "inkling"
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "The answer"},
            ],
            reasoning_effort=0.5,
            continue_final_message=True,
        )
        prompt_ids = serving._encode_messages(
            [message.model_dump() for message in request.messages],
            request,
            thinking_mode=None,
        )
        from sglang.srt.parser.inkling_renderer import load_tml_renderers

        open_block = [
            INKLING_SPECIAL_TOKEN_IDS["<|message_model|>"],
            INKLING_SPECIAL_TOKEN_IDS["<|content_text|>"],
            *load_tml_renderers().tokenizer.encode_ordinary("The answer"),
        ]
        self.assertEqual(prompt_ids[-len(open_block) :], open_block)
        self.assertNotIn(
            INKLING_SPECIAL_TOKEN_IDS["<|content_model_end_sampling|>"], prompt_ids
        )

    def test_continue_final_message_leaves_tool_call_turns_closed(self):
        """A trailing assistant message with tool_calls cannot be continued —
        it must keep rendering as a closed historical turn."""
        from sglang.srt.parser.inkling_tokenizer import INKLING_SPECIAL_TOKEN_IDS

        serving = object.__new__(OpenAIServingChat)
        serving.chat_encoding_spec = "inkling"
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "calling",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{}"},
                        }
                    ],
                },
            ],
            reasoning_effort=0.5,
            continue_final_message=True,
        )
        prompt_ids = serving._encode_messages(
            [message.model_dump() for message in request.messages],
            request,
            thinking_mode=None,
        )
        self.assertEqual(
            prompt_ids[-1],
            INKLING_SPECIAL_TOKEN_IDS["<|content_model_end_sampling|>"],
        )


class InklingTokenOutputTest(CustomTestCase):
    """Inkling chat responses are parsed from output token IDs, not text."""

    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.serving = object.__new__(OpenAIServingChat)
        self.serving.chat_encoding_spec = "inkling"
        self.serving.reasoning_parser = "inkling"
        self.serving.tool_call_parser = "inkling"
        self.serving._inkling_token_output = True
        self.request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}},
                }
            ],
        )

    @staticmethod
    def _output_ids() -> list[int]:
        from sglang.srt.parser.inkling_renderer import load_tml_renderers

        tokenizer = load_tml_renderers().tokenizer
        special = tokenizer.encode_special
        return [
            special("message_model"),
            special("content_thinking"),
            *tokenizer.encode_ordinary("plan"),
            special("end_message"),
            special("message_model"),
            *tokenizer.encode_ordinary("weather"),
            special("content_invoke_tool_json"),
            *tokenizer.encode_ordinary('{"name":"weather","args":{"city":"SF"}}'),
            special("end_message"),
            special("content_model_end_sampling"),
        ]

    def test_non_stream_maps_calls_and_finish_reason(self):
        reasoning, content, tool_calls, finish_reason = (
            self.serving._parse_inkling_response(
                self.request, self._output_ids(), {"type": "stop", "matched": 200006}
            )
        )
        self.assertEqual(reasoning, "plan")
        self.assertEqual(content, "")
        self.assertEqual(finish_reason, {"type": "tool_calls", "matched": None})
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].index, 0)
        self.assertEqual(tool_calls[0].function.name, "weather")
        self.assertEqual(json.loads(tool_calls[0].function.arguments), {"city": "SF"})
        self.assertTrue(tool_calls[0].id.startswith("call_"))

    def test_continued_final_message_output_is_content(self):
        """Bug regression: with continue_final_message the prompt ends inside
        an open model text block, so the sampled tokens carry no header; they
        were silently dropped and the response content came back empty."""
        from sglang.srt.parser.inkling_renderer import load_tml_renderers

        tokenizer = load_tml_renderers().tokenizer
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {"role": "user", "content": "First five primes?"},
                {"role": "assistant", "content": "2, 3,"},
            ],
            continue_final_message=True,
        )
        output_ids = [
            *tokenizer.encode_ordinary(" 5, 7, 11"),
            tokenizer.encode_special("end_message"),
            tokenizer.encode_special("content_model_end_sampling"),
        ]
        _, content, tool_calls, _ = self.serving._parse_inkling_response(
            request, output_ids, {"type": "stop", "matched": 200006}
        )
        self.assertEqual(content, " 5, 7, 11")
        self.assertIsNone(tool_calls)

    def _stream_deltas(
        self,
        request,
        output_ids,
        finish_reason,
        *,
        incremental=False,
        split=4,
        has_tool_calls=None,
    ) -> list[dict]:
        parser_dict, chunks = {}, []
        steps = (
            (output_ids[:split], None),
            (output_ids[split:] if incremental else output_ids, finish_reason),
        )
        with get_context().override_server_args(
            incremental_streaming_output=incremental
        ):
            for ids, finish in steps:
                chunks += self.serving._inkling_stream_chunks(
                    content={
                        "output_ids": ids,
                        "meta_info": {
                            "id": "chatcmpl-1",
                            "completion_tokens": len(output_ids),
                            "finish_reason": finish,
                        },
                    },
                    index=0,
                    request=request,
                    parser_dict=parser_dict,
                    has_tool_calls={} if has_tool_calls is None else has_tool_calls,
                    choice_logprobs=None,
                    finish_reason_type=finish and finish["type"],
                    continuous_usage_stats=False,
                )
        return [
            json.loads(chunk[len("data: ") :])["choices"][0]["delta"]
            for chunk in chunks
        ]

    def _reasoning_and_content(
        self, request, output_ids, finish_reason, *, split=2
    ) -> dict:
        reasoning, content, _, _ = self.serving._parse_inkling_response(
            request, output_ids, finish_reason
        )
        results = {"non-stream": (reasoning or "", content)}
        for incremental in (False, True):
            deltas = self._stream_deltas(
                request,
                output_ids,
                finish_reason,
                incremental=incremental,
                split=split,
            )
            results[f"stream incremental={incremental}"] = (
                "".join(d.get("reasoning_content") or "" for d in deltas),
                "".join(d.get("content") or "" for d in deltas),
            )
        return results

    def test_stream_emits_reasoning_then_one_complete_tool_call(self):
        has_tool_calls = {}
        deltas = self._stream_deltas(
            self.request,
            self._output_ids(),
            {"type": "stop", "matched": 200006},
            has_tool_calls=has_tool_calls,
        )
        self.assertEqual(
            "".join(d.get("reasoning_content") or "" for d in deltas), "plan"
        )
        calls = [call for d in deltas for call in d.get("tool_calls") or []]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "SF"})
        self.assertEqual(has_tool_calls, {0: True})

    def test_matched_stop_is_trimmed_from_visible_text(self):
        """Bug regression: output ids include the matched stop, which only the
        detokenized text had trimmed; the token-ID parse returned it verbatim."""
        from sglang.srt.parser.inkling_renderer import load_tml_renderers

        tokenizer = load_tml_renderers().tokenizer
        encode = tokenizer.encode_ordinary
        special = tokenizer.encode_special
        text_block = [special("message_model"), special("content_text")]
        thinking_block = [special("message_model"), special("content_thinking")]
        bang = encode("!")
        self.assertEqual(len(encode(" hello")), 1)
        cases = [
            (
                "string spanning tokens",
                text_block,
                encode("hello") + encode("EN") + encode("D"),
                "END",
                False,
                ("", "hello"),
            ),
            (
                "string inside a token",
                text_block,
                encode("say") + encode(" hello"),
                "hel",
                False,
                ("", "say "),
            ),
            (
                "kept string inside a token",
                text_block,
                encode("say") + encode(" hello"),
                "hel",
                True,
                ("", "say hel"),
            ),
            (
                "string in reasoning",
                thinking_block,
                encode("plan") + encode("END"),
                "END",
                False,
                ("plan", ""),
            ),
            (
                "ordinary stop token",
                text_block,
                encode("hello") + bang,
                bang[0],
                False,
                ("", "hello"),
            ),
            (
                "kept ordinary stop token",
                text_block,
                encode("hello") + bang,
                bang[0],
                True,
                ("", "hello!"),
            ),
        ]
        for name, header, payload, matched, no_stop_trim, expected in cases:
            request = ChatCompletionRequest(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                no_stop_trim=no_stop_trim,
            )
            results = self._reasoning_and_content(
                request, header + payload, {"type": "stop", "matched": matched}
            )
            for mode, result in results.items():
                with self.subTest(case=name, mode=mode):
                    self.assertEqual(result, expected)

    def test_stop_string_is_cut_where_the_sampler_matched(self):
        """Bug regression: the stop string was searched in the merged content,
        where two text blocks can spell it although framing separated them in
        the sampled stream; the answer was truncated at that false match."""
        from sglang.srt.parser.inkling_renderer import load_tml_renderers

        tokenizer = load_tml_renderers().tokenizer
        encode = tokenizer.encode_ordinary
        special = tokenizer.encode_special
        first_block = [
            special("message_model"),
            special("content_text"),
            *encode("EN"),
            special("end_message"),
        ]
        output_ids = [
            *first_block,
            special("message_model"),
            special("content_text"),
            *encode("D and actual END"),
        ]
        for no_stop_trim, expected in (
            (False, ("", "END and actual ")),
            (True, ("", "END and actual END")),
        ):
            request = ChatCompletionRequest(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                no_stop_trim=no_stop_trim,
            )
            results = self._reasoning_and_content(
                request,
                output_ids,
                {"type": "stop", "matched": "END"},
                split=len(first_block),
            )
            for mode, result in results.items():
                with self.subTest(no_stop_trim=no_stop_trim, mode=mode):
                    self.assertEqual(result, expected)

    def test_overlapping_stop_is_cut_at_its_earliest_start(self):
        """Bug regression: the backward search stopped at the first suffix
        holding any match, so a stop that began in the previous token and
        overlapped a later match in the final token was cut too late."""
        from sglang.srt.parser.inkling_renderer import load_tml_renderers

        tokenizer = load_tml_renderers().tokenizer
        header = [
            tokenizer.encode_special("message_model"),
            tokenizer.encode_special("content_text"),
        ]
        cases = [("\n", "\n\n", "\n\n"), ("ab", "aba", "aba")]
        for previous, final, stop in cases:
            sampled = [
                *tokenizer.encode_ordinary(previous),
                *tokenizer.encode_ordinary(final),
            ]
            self.assertEqual(len(sampled), 2)
            for no_stop_trim, expected in ((False, ("", "")), (True, ("", stop))):
                request = ChatCompletionRequest(
                    model="test-model",
                    messages=[{"role": "user", "content": "hi"}],
                    no_stop_trim=no_stop_trim,
                )
                results = self._reasoning_and_content(
                    request, header + sampled, {"type": "stop", "matched": stop}
                )
                for mode, result in results.items():
                    with self.subTest(stop=stop, no_stop_trim=no_stop_trim, mode=mode):
                        self.assertEqual(result, expected)

    def test_special_stop_inside_open_text_block_is_not_visible(self):
        """Bug regression: inside an open text block TML renders a special token
        as text, so a custom special stop id leaked unless trimmed by id."""
        from sglang.srt.parser.inkling_renderer import load_tml_renderers

        tokenizer = load_tml_renderers().tokenizer
        encode = tokenizer.encode_ordinary
        stop_id = tokenizer.encode_special("content_thinking")
        text_block = [
            tokenizer.encode_special("message_model"),
            tokenizer.encode_special("content_text"),
        ]
        continued = [
            {"role": "user", "content": "First five primes?"},
            {"role": "assistant", "content": "2, 3,"},
        ]
        cases = [
            ("explicit block", [{"role": "user", "content": "hi"}], text_block),
            ("continued block", continued, []),
        ]
        for name, messages, header in cases:
            for no_stop_trim, expected in (
                (False, ("", " 5, 7")),
                (True, ("", " 5, 7<|content_thinking|>")),
            ):
                request = ChatCompletionRequest(
                    model="test-model",
                    messages=messages,
                    continue_final_message=name == "continued block",
                    no_stop_trim=no_stop_trim,
                )
                results = self._reasoning_and_content(
                    request,
                    [*header, *encode(" 5, 7"), stop_id],
                    {"type": "stop", "matched": stop_id},
                    split=len(header) + 1,
                )
                for mode, result in results.items():
                    with self.subTest(case=name, no_stop_trim=no_stop_trim, mode=mode):
                        self.assertEqual(result, expected)

    def test_unframed_prefix_keeps_the_following_marker_kind(self):
        """Bug regression: stray text before a header-less thinking or tool-call
        marker erased the marker, so reasoning became content and the call text."""
        from sglang.srt.parser.inkling_renderer import load_tml_renderers

        tokenizer = load_tml_renderers().tokenizer
        encode = tokenizer.encode_ordinary
        special = tokenizer.encode_special
        eos = special("content_model_end_sampling")
        thinking = [
            *encode("\n\n"),
            special("content_thinking"),
            *encode("plan"),
            special("end_message"),
            special("message_model"),
            special("content_text"),
            *encode("answer"),
            special("end_message"),
            eos,
        ]
        results = self._reasoning_and_content(
            self.request, thinking, {"type": "stop", "matched": eos}
        )
        for mode, result in results.items():
            with self.subTest(case="thinking", mode=mode):
                self.assertEqual(result, ("plan", "\n\nanswer"))

        call = [
            *encode(" "),
            special("content_invoke_tool_json"),
            *encode('{"name":"weather","args":{"city":"SF"}}'),
            special("end_message"),
            eos,
        ]
        _, _, tool_calls, finish_reason = self.serving._parse_inkling_response(
            self.request, call, {"type": "stop", "matched": eos}
        )
        self.assertEqual(finish_reason["type"], "tool_calls")
        self.assertEqual(
            [(c.function.name, json.loads(c.function.arguments)) for c in tool_calls],
            [("weather", {"city": "SF"})],
        )

    def test_constrained_output_without_header_is_content(self):
        """Bug regression: response_format grammars sample bare JSON where a
        message header belongs (after the reasoning terminator, or from the
        first token), and the parser dropped that unframed text."""
        from sglang.srt.parser.inkling_renderer import load_tml_renderers

        tokenizer = load_tml_renderers().tokenizer
        encode = tokenizer.encode_ordinary
        special = tokenizer.encode_special
        payload = '{"marker": "<|end_message|>", "ok": true}'
        thinking = [
            special("message_model"),
            special("content_thinking"),
            *encode("plan"),
            special("end_message"),
        ]
        eos = special("content_model_end_sampling")
        # Grammars can sample a content-kind token inside the payload; the EOS
        # then lands in a reopened text block and must still not render.
        with_kind_token = [
            *encode('{"a": "'),
            special("content_text"),
            *encode('x"}'),
        ]
        cases = [
            ("after thinking", thinking, encode(payload), ("plan", payload)),
            ("from first token", [], encode(payload), ("", payload)),
            ("content kind in payload", [], with_kind_token, ("", '{"a": "x"}')),
        ]
        for name, prefix, sampled, expected in cases:
            results = self._reasoning_and_content(
                self.request,
                [*prefix, *sampled, eos],
                {"type": "stop", "matched": eos},
            )
            for mode, result in results.items():
                with self.subTest(case=name, mode=mode):
                    self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
