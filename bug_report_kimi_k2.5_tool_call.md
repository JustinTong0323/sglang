# Bug Report: Kimi-K2.5 Tool Call 在 SGLang 上概率性解析失败

## 问题概述

Kimi-K2.5 模型通过 SGLang 部署后，工具调用（tool call）存在概率性失败——当模型输出不包含显式的 `</think>` token 而直接输出 `<|tool_calls_section_begin|>` 时，tool call 内容会被错误归类为推理文本（reasoning），导致函数调用解析失败。官方 Kimi API 无此问题。

## 复现环境

| 组件 | 版本 |
|------|------|
| 推理框架 | SGLang 0.5.8 |
| GPU | 8× NVIDIA H100 80GB |
| CUDA | 12.9 |
| Python | 3.12.3 |
| PyTorch | 2.9.1 |

### 启动命令

```bash
sglang serve --model-path /path/to/Kimi-K2_5 \
  --tp 16 --trust-remote-code \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2
```

### 调用客户端

Claude Code / OpenCode / Kimi-Code 等 agentic coding 工具。

## 问题详述

### 正常输出（可正确解析）

模型正常返回 `</think>` 结束推理后再进入工具调用：

```
<think>用户想要读取文件，我需要调用 ReadFile 工具。</think>
<|tool_calls_section_begin|>
<|tool_call_begin|>functions.ReadFile:0<|tool_call_argument_begin|>{"path": "/src/main.py"}<|tool_call_end|>
<|tool_calls_section_end|>
```

### 异常输出（解析失败）

模型**跳过 `</think>`**，直接输出工具调用 token：

```
<think>用户想要读取文件，我需要调用 ReadFile 工具。
<|tool_calls_section_begin|>
<|tool_call_begin|>functions.ReadFile:0<|tool_call_argument_begin|>{"path": "/src/main.py"}<|tool_call_end|>
<|tool_calls_section_end|>
```

此时 SGLang 的推理解析器（reasoning parser）不会终止推理模式，`<|tool_calls_section_begin|>` 及后续所有内容被作为 reasoning_text 吞掉，tool call parser 永远收不到输入，API 返回结果中 `tool_calls` 为空。

## 根因分析

SGLang 对 Kimi-K2.5 的推理解析使用 `Qwen3Detector`，其仅识别 `<think>` / `</think>` 作为推理边界：

```python
# reasoning_parser.py line 434
DetectorMap = {
    "kimi_k2": Qwen3Detector,  # 复用 Qwen3 的 <think>...</think> 解析
}
```

`Qwen3Detector` 初始化时**未传递 `tool_start_token`**：

```python
class Qwen3Detector(BaseReasoningFormatDetector):
    def __init__(self, ...):
        super().__init__(
            "<think>",
            "</think>",
            # 缺少: tool_start_token="<|tool_calls_section_begin|>"
        )
```

而 `BaseReasoningFormatDetector` 本身已支持 `tool_start_token` 参数——当检测到该 token 时，会隐式结束推理模式并将后续内容传递给 tool call parser：

```python
# BaseReasoningFormatDetector.parse_non_stream()
if (
    in_reasoning
    and self.tool_start_token is not None
    and self.tool_start_token in processed_text
):
    tool_idx = processed_text.find(self.tool_start_token)
    reasoning_text = processed_text[:tool_idx].strip()
    normal_text = processed_text[tool_idx:]  # 保留 tool_start_token 给下游解析
```

由于 `tool_start_token` 未设置，当模型跳过 `</think>` 直接生成 `<|tool_calls_section_begin|>` 时，推理解析器无法识别这一隐式边界，导致整个输出被错误归类为推理内容。

## SGLang 侧修复方案

PR [#19696](https://github.com/sgl-project/sglang/pull/19696) 的修复思路：将 `<|tool_calls_section_begin|>` 注册为隐式 think_end token，当推理解析器检测到该 token 时，自动终止推理模式并将后续内容交由 tool call parser 处理。

## 希望 Kimi 团队确认的问题

1. **预期行为规范**：Kimi-K2.5 在 tool call 场景下，`</think>` 是否保证在 `<|tool_calls_section_begin|>` 之前出现？还是说模型设计上允许跳过 `</think>` 直接进入 tool call？如果有明确的 token 生成顺序规范文档，烦请提供。

2. **官方 API 后处理**：官方 Kimi API 是否在推理引擎之上有额外的后处理逻辑（例如全上下文受约束解码 / full-context constrained decoding）来保证 tool call 的正确解析？如果有，这套逻辑的核心思路是什么？

## 参考链接

- GitHub Issue: https://github.com/sgl-project/sglang/issues/18086
- 修复 PR: https://github.com/sgl-project/sglang/pull/19696
- Kimi-K2.5 Tool Call 官方指引: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md
