# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Centralized template management for chat templates and completion templates.

This module provides a unified interface for managing both chat conversation templates
and code completion templates, eliminating global state and improving modularity.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.parser.code_completion_parser import (
    CompletionTemplate,
    FimPosition,
    completion_template_exists,
    register_completion_template,
    set_completion_template,
)
from sglang.srt.parser.conversation import (
    Conversation,
    SeparatorStyle,
    chat_template_exists,
    get_conv_template_by_model_path,
    register_conv_template,
)
from sglang.srt.parser.jinja_template_utils import detect_jinja_template_content_format

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemplateDetectionContext:
    template: str
    reasoning_config: Optional["ReasoningToggleConfig"]
    force_reasoning: bool
    vocab: set[str]

    def has_text(self, needle: str) -> bool:
        return needle in self.template

    def has_vocab(self, token: str) -> bool:
        return token in self.vocab

    def has_pattern(self, pattern: str, flags: int = 0) -> bool:
        return re.search(pattern, self.template, flags) is not None


@dataclass(frozen=True)
class DetectionRule:
    name: str
    value: object
    predicate: Callable[[TemplateDetectionContext], bool]


@dataclass(frozen=True)
class ReasoningToggleConfig:
    toggle_param: Optional[str] = None
    default_enabled: Optional[bool] = None
    special_case: Optional[str] = None

    @property
    def always_on(self) -> bool:
        return self.special_case == "always"


REASONING_MODE_RULES = (
    DetectionRule(
        name="gpt_oss_channel_markers",
        value=ReasoningToggleConfig(special_case="always"),
        predicate=lambda ctx: ctx.has_text("<|channel|>"),
    ),
    DetectionRule(
        name="force_reasoning_pattern",
        value=ReasoningToggleConfig(special_case="always"),
        predicate=lambda ctx: ctx.has_pattern(
            r"<\|im_start\|>assistant\\n<think>\\n"
        )
        and not ctx.has_text("enable_thinking")
        and not ctx.has_text("thinking"),
    ),
    DetectionRule(
        name="mistral_reasoning_effort",
        value=ReasoningToggleConfig(special_case="mistral"),
        predicate=lambda ctx: ctx.has_text("reasoning_effort")
        and ctx.has_text("[THINK]"),
    ),
    DetectionRule(
        name="explicit_enable_thinking_default_false",
        value=ReasoningToggleConfig(
            toggle_param="enable_thinking", default_enabled=False
        ),
        predicate=lambda ctx: ctx.has_pattern(
            r"{%\s*if\s+not\s+enable_thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+enable_thinking\s*=\s*(?:false|False)\s*%}",
            re.DOTALL,
        ),
    ),
    DetectionRule(
        name="enable_thinking_default_true",
        value=ReasoningToggleConfig(
            toggle_param="enable_thinking", default_enabled=True
        ),
        predicate=lambda ctx: ctx.has_pattern(
            r"{%\s*if\s+not\s+enable_thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+enable_thinking\s*=\s*(?:true|True)\s*%}",
            re.DOTALL,
        )
        or ctx.has_pattern(
            r"set\s+enable_thinking\s*=\s*enable_thinking\s+if\s+enable_thinking\s+is\s+defined\s+else\s+(?:true|True)"
        )
        or ctx.has_pattern(
            r"enable_thinking\s+is\s+defined\s+and\s+(?:enable_thinking\s+is\s+false|not\s+enable_thinking)"
        )
        or ctx.has_pattern(r"enable_thinking\s+is\s+not\s+defined\s+or\s+enable_thinking")
        or ctx.has_pattern(r"namespace\([^)]*enable_thinking\s*=\s*true"),
    ),
    DetectionRule(
        name="explicit_thinking_default_false",
        value=ReasoningToggleConfig(toggle_param="thinking", default_enabled=False),
        predicate=lambda ctx: ctx.has_pattern(
            r"{%\s*if\s+not\s+thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+thinking\s*=\s*(?:false|False)\s*%}",
            re.DOTALL,
        ),
    ),
    DetectionRule(
        name="thinking_default_true",
        value=ReasoningToggleConfig(toggle_param="thinking", default_enabled=True),
        predicate=lambda ctx: ctx.has_pattern(
            r"{%\s*if\s+not\s+thinking\s+is\s+defined\s*%}.*?"
            r"{%\s*set\s+thinking\s*=\s*(?:true|True)\s*%}",
            re.DOTALL,
        )
        or ctx.has_pattern(
            r"set\s+thinking\s*=\s*thinking\s+if\s+thinking\s+is\s+defined\s+else\s+(?:true|True)"
        )
        or ctx.has_pattern(r"thinking\s+is\s+defined\s+and\s+(?:thinking\s+is\s+false|not\s+thinking)")
        or ctx.has_pattern(r"thinking\s+is\s+not\s+defined\s+or\s+thinking")
        or ctx.has_pattern(r"namespace\([^)]*thinking\s*=\s*true"),
    ),
)


REASONING_PARSER_RULES = (
    DetectionRule(
        name="gemma4",
        value="gemma4",
        predicate=lambda ctx: ctx.has_text("<|channel>"),
    ),
    DetectionRule(
        name="kimi",
        value="kimi",
        predicate=lambda ctx: ctx.has_text("\u25c1think\u25b7")
        or ctx.has_text("◁think▷"),
    ),
    DetectionRule(
        name="interns1",
        value="interns1",
        predicate=lambda ctx: ctx.has_text("default_thinking_sys")
        and ctx.reasoning_config
        == ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True),
    ),
    DetectionRule(
        name="mistral",
        value="mistral",
        predicate=lambda ctx: (
            ctx.reasoning_config is not None
            and ctx.reasoning_config.special_case == "mistral"
        ),
    ),
    DetectionRule(
        name="gpt_oss",
        value="gpt-oss",
        predicate=lambda ctx: ctx.has_text("<|channel|>"),
    ),
    DetectionRule(
        name="kimi_k2",
        value="kimi_k2",
        predicate=lambda ctx: ctx.has_vocab("<|tool_calls_section_begin|>"),
    ),
    DetectionRule(
        name="nemotron_3",
        value="nemotron_3",
        predicate=lambda ctx: ctx.has_text("truncate_history_thinking")
        and ctx.reasoning_config
        == ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True),
    ),
    DetectionRule(
        name="glm45",
        value="glm45",
        predicate=lambda ctx: (
            ctx.has_text("[gMASK]<sop>")
            or ctx.has_pattern(r"(?<!<)/nothink")
            or ctx.has_pattern(r"(?<!<)/think")
        )
        and ctx.has_vocab("<tool_call>")
        and ctx.reasoning_config
        == ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True)
        and (ctx.has_vocab("<|user|>") or ctx.has_vocab("<|endoftext|>")),
    ),
    DetectionRule(
        name="mimo",
        value="mimo",
        predicate=lambda ctx: ctx.reasoning_config
        == ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=False),
    ),
    DetectionRule(
        name="minimax",
        value="minimax",
        predicate=lambda ctx: ctx.has_text("<minimax:tool_call>"),
    ),
    DetectionRule(
        name="qwen3",
        value="qwen3",
        predicate=lambda ctx: ctx.reasoning_config
        == ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True),
    ),
    DetectionRule(
        name="deepseek_v3",
        value="deepseek-v3",
        predicate=lambda ctx: ctx.reasoning_config
        == ReasoningToggleConfig(toggle_param="thinking", default_enabled=False),
    ),
    DetectionRule(
        name="deepseek_r1_force",
        value="deepseek-r1",
        predicate=lambda ctx: ctx.force_reasoning,
    ),
    DetectionRule(
        name="deepseek_r1_think_tags",
        value="deepseek-r1",
        predicate=lambda ctx: ctx.has_text("<think>") or ctx.has_text("</think>"),
    ),
)


class TemplateManager:
    """
    Centralized manager for chat and completion templates.

    This class encapsulates all template-related state and operations,
    eliminating the need for global variables and providing a clean
    interface for template management.
    """

    def __init__(self):
        self._chat_template_name: Optional[str] = None
        self._completion_template_name: Optional[str] = None
        self._jinja_template_content_format: Optional[str] = "openai"
        self._force_reasoning: bool = False
        self._reasoning_config: Optional[ReasoningToggleConfig] = None
        self._suggested_reasoning_parser: Optional[str] = None

    @property
    def chat_template_name(self) -> Optional[str]:
        """Get the current chat template name."""
        return self._chat_template_name

    @property
    def completion_template_name(self) -> Optional[str]:
        """Get the current completion template name."""
        return self._completion_template_name

    @property
    def jinja_template_content_format(self) -> Optional[str]:
        """Get the detected template content format ('string' or 'openai' or None)."""
        return self._jinja_template_content_format

    @property
    def force_reasoning(self) -> bool:
        """
        Check if the current chat template enforces reasoning/thinking.

        Returns:
            True if the template contains reasoning patterns like <think> tags
        """
        return self._force_reasoning

    @property
    def reasoning_config(self) -> Optional[ReasoningToggleConfig]:
        """Get the reasoning toggle config inferred from chat template."""
        return self._reasoning_config

    @property
    def suggested_reasoning_parser(self) -> Optional[str]:
        """Get the auto-detected reasoning parser name, or None."""
        return self._suggested_reasoning_parser

    def resolve_auto_reasoning_parser(self, server_args) -> None:
        """Resolve --reasoning-parser=auto using a lightweight tokenizer load."""
        if server_args.reasoning_parser != "auto":
            return
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                server_args.model_path, trust_remote_code=True
            )
            template = getattr(tokenizer, "chat_template", None)
            if template:
                self._force_reasoning, self._reasoning_config = (
                    self._detect_reasoning_pattern(template)
                )
                self._suggested_reasoning_parser = self._detect_reasoning_parser(
                    template, tokenizer
                )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for auto-detection: {e}")

        if self._suggested_reasoning_parser:
            server_args.reasoning_parser = self._suggested_reasoning_parser
            logger.info(
                f"Auto-detected --reasoning-parser as '{server_args.reasoning_parser}' "
                f"from chat template"
            )
        else:
            logger.warning(
                "--reasoning-parser=auto specified but could not detect reasoning "
                "format from chat template. Disabling reasoning parser."
            )
            server_args.reasoning_parser = None

    def _detect_reasoning_pattern(
        self, template: str
    ) -> tuple[bool, Optional[ReasoningToggleConfig]]:
        """
        Detect if the chat template contains reasoning/thinking patterns.
        """
        if template is None:
            return False, None

        ctx = TemplateDetectionContext(
            template=template,
            reasoning_config=None,
            force_reasoning=False,
            vocab=set(),
        )
        for rule in REASONING_MODE_RULES:
            if rule.predicate(ctx):
                logger.info(
                    "Detected reasoning config '%s' from template rule '%s'.",
                    rule.value,
                    rule.name,
                )
                return rule.value.always_on, rule.value

        return False, None

    def _detect_reasoning_parser(
        self, template: Optional[str], tokenizer
    ) -> Optional[str]:
        """
        Auto-detect which reasoning parser to use from the chat template and tokenizer.

        Uses template markers, Jinja variables, and tokenizer vocab to identify the
        model family and return the appropriate parser name.
        """
        if template is None:
            return None

        vocab = set()
        if tokenizer is not None:
            try:
                vocab = set(tokenizer.get_vocab().keys())
            except Exception:
                pass
        ctx = TemplateDetectionContext(
            template=template,
            reasoning_config=self._reasoning_config,
            force_reasoning=self._force_reasoning,
            vocab=vocab,
        )
        for rule in REASONING_PARSER_RULES:
            if rule.predicate(ctx):
                logger.info(
                    "Detected reasoning parser '%s' from template rule '%s'.",
                    rule.value,
                    rule.name,
                )
                return rule.value
        return None

    def load_chat_template(
        self,
        tokenizer_manager: TokenizerManager,
        chat_template_arg: Optional[str],
        model_path: str,
    ) -> None:
        """
        Load a chat template from various sources.

        Args:
            tokenizer_manager: The tokenizer manager instance
            chat_template_arg: Template name, file path, or None to auto-detect
            model_path: Path to the model
        """
        if chat_template_arg:
            self._load_explicit_chat_template(tokenizer_manager, chat_template_arg)
        else:
            # Guess chat template from model path
            self.guess_chat_template_from_model_path(model_path)

            # If no pre-defined template was found, fallback to HuggingFace template
            if self._chat_template_name is None:
                # Try HuggingFace template first
                hf_template = self._resolve_hf_chat_template(tokenizer_manager)
                if hf_template:
                    # override the chat template
                    if tokenizer_manager.tokenizer:
                        tokenizer_manager.tokenizer.chat_template = hf_template
                    self._jinja_template_content_format = (
                        detect_jinja_template_content_format(hf_template)
                    )
                    logger.info(
                        f"Using default HuggingFace chat template with detected content format: {self._jinja_template_content_format}"
                    )
                else:
                    # Default to string content format if no template was found
                    self._jinja_template_content_format = "string"
                    logger.info(
                        "No chat template found, defaulting to 'string' content format"
                    )

        # Detect reasoning pattern and suggest parser from chat template
        if tokenizer_manager.tokenizer:
            template = tokenizer_manager.tokenizer.chat_template
            self._force_reasoning, self._reasoning_config = (
                self._detect_reasoning_pattern(template)
            )
            self._suggested_reasoning_parser = self._detect_reasoning_parser(
                template, tokenizer_manager.tokenizer
            )
            if self._suggested_reasoning_parser:
                logger.info(
                    f"Auto-detected reasoning parser: {self._suggested_reasoning_parser}"
                )

    def _load_explicit_chat_template(
        self, tokenizer_manager: TokenizerManager, chat_template_arg: str
    ) -> None:
        """Load explicitly specified chat template."""
        logger.info(f"Loading chat template from argument: {chat_template_arg}")

        if chat_template_exists(chat_template_arg):
            self._chat_template_name = chat_template_arg
            return

        if not os.path.exists(chat_template_arg):
            raise RuntimeError(
                f"Chat template {chat_template_arg} is not a built-in template name "
                "or a valid chat template file path."
            )

        if chat_template_arg.endswith(".jinja"):
            self._load_jinja_template(tokenizer_manager, chat_template_arg)
        else:
            self._load_json_chat_template(chat_template_arg)

    def guess_chat_template_from_model_path(self, model_path: str) -> None:
        """
        Infer chat template name from model path.

        Args:
            model_path: Path to the model
        """
        template_name = get_conv_template_by_model_path(model_path)
        if template_name is not None:
            logger.info(f"Inferred chat template from model path: {template_name}")
            self._chat_template_name = template_name

    def load_completion_template(self, completion_template_arg: str) -> None:
        """
        Load completion template for code completion.

        Args:
            completion_template_arg: Template name or file path
        """
        logger.info(f"Loading completion template: {completion_template_arg}")

        if not completion_template_exists(completion_template_arg):
            if not os.path.exists(completion_template_arg):
                raise RuntimeError(
                    f"Completion template {completion_template_arg} is not a built-in template name "
                    "or a valid completion template file path."
                )

            self._load_json_completion_template(completion_template_arg)
        else:
            self._completion_template_name = completion_template_arg

        set_completion_template(self._completion_template_name)

    def initialize_templates(
        self,
        tokenizer_manager: TokenizerManager,
        model_path: str,
        chat_template: Optional[str] = None,
        completion_template: Optional[str] = None,
    ) -> None:
        """
        Initialize all templates based on provided configuration.

        Args:
            tokenizer_manager: The tokenizer manager instance
            model_path: Path to the model
            chat_template: Optional chat template name/path
            completion_template: Optional completion template name/path
        """
        # Load chat template
        self.load_chat_template(tokenizer_manager, chat_template, model_path)

        # Load completion template
        if completion_template:
            self.load_completion_template(completion_template)

    def _load_jinja_template(
        self, tokenizer_manager: TokenizerManager, template_path: str
    ) -> None:
        """Load a Jinja template file."""
        with open(template_path, "r") as f:
            chat_template = "".join(f.readlines()).strip("\n")
        tokenizer_manager.tokenizer.chat_template = chat_template.replace("\\n", "\n")
        self._chat_template_name = None
        # Detect content format from the loaded template
        self._jinja_template_content_format = detect_jinja_template_content_format(
            chat_template
        )
        logger.info(
            f"Detected user specified Jinja chat template with content format: {self._jinja_template_content_format}"
        )

    def _load_json_chat_template(self, template_path: str) -> None:
        """Load a JSON chat template file."""
        assert template_path.endswith(
            ".json"
        ), "unrecognized format of chat template file"

        with open(template_path, "r") as filep:
            template = json.load(filep)
            try:
                sep_style = SeparatorStyle[template["sep_style"]]
            except KeyError:
                raise ValueError(
                    f"Unknown separator style: {template['sep_style']}"
                ) from None

            register_conv_template(
                Conversation(
                    name=template["name"],
                    system_template=template["system"] + "\n{system_message}",
                    system_message=template.get("system_message", ""),
                    roles=(template["user"], template["assistant"]),
                    sep_style=sep_style,
                    sep=template.get("sep", "\n"),
                    stop_str=template["stop_str"],
                ),
                override=True,
            )
        self._chat_template_name = template["name"]

    def _load_json_completion_template(self, template_path: str) -> None:
        """Load a JSON completion template file."""
        assert template_path.endswith(
            ".json"
        ), "unrecognized format of completion template file"

        with open(template_path, "r") as filep:
            template = json.load(filep)
            try:
                fim_position = FimPosition[template["fim_position"]]
            except KeyError:
                raise ValueError(
                    f"Unknown fim position: {template['fim_position']}"
                ) from None

            register_completion_template(
                CompletionTemplate(
                    name=template["name"],
                    fim_begin_token=template["fim_begin_token"],
                    fim_middle_token=template["fim_middle_token"],
                    fim_end_token=template["fim_end_token"],
                    fim_position=fim_position,
                ),
                override=True,
            )
        self._completion_template_name = template["name"]

    def _resolve_hf_chat_template(
        self, tokenizer_manager: TokenizerManager
    ) -> Optional[str]:
        try:
            # Try (mm-)processor first, then tokenizer
            template = (
                getattr(tokenizer_manager.processor, "chat_template", None)
                if tokenizer_manager.processor
                else None
            ) or (
                getattr(tokenizer_manager.tokenizer, "chat_template", None)
                if tokenizer_manager.tokenizer
                else None
            )

            if template is None:
                logger.warning("No HuggingFace chat template found")
                return None

            # Handle dict templates (multiple named templates)
            if isinstance(template, dict):
                return self._select_named_template(template, tokenizer_manager)

            # Single string template
            return template

        except Exception as e:
            logger.warning(f"Error getting chat template: {e}")
            return None

    def _select_named_template(
        self, templates: Dict[str, str], tokenizer_manager: TokenizerManager
    ) -> str:
        if not templates:
            raise ValueError("Empty templates dict provided")

        available_names = list(templates.keys())
        logger.info(f"Multiple HuggingFace chat templates available: {available_names}")

        # Use specified template if provided
        if preferred_name := tokenizer_manager.server_args.hf_chat_template_name:
            if preferred_name not in templates:
                raise ValueError(
                    f"Specified template '{preferred_name}' not found. "
                    f"Available templates: {available_names}"
                )
            logger.info(f"Using specified chat template: '{preferred_name}'")
            return templates[preferred_name]

        # Fallback: Use first available template
        first_name = available_names[0]
        logger.info(f"Using first available template: '{first_name}'")
        return templates[first_name]
