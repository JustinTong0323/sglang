# Issue #18478 分析：SGLang 工具输出格式缺少 Function 标签

## 问题概述

这个 issue 指出 SGLang 在处理函数调用（function calling）时，工具（tools）的输出格式与 OpenAI 标准格式不一致，缺少了 `type: "function"` 和 `function` 外层包装。

## 问题历史

### 1. PR #6556 (2025.05.23) - 修复工具格式
最初，PR #6556 修复了输入工具格式，使其与 OpenAI 标准保持一致。

**关键改动：**
```python
# 修改前 (只提取 function 字段)
tools = [item.function.model_dump() for item in request.tools]

# 修改后 (保留完整的 Tool 对象，包括 type 和 function)
tools = [item.model_dump() for item in request.tools]
```

这个改动确保工具格式保持为 OpenAI 标准格式：
```json
{
  "type": "function",
  "function": {
    "name": "code_interpreter",
    "description": "...",
    "parameters": {
      "type": "object",
      "properties": { ... }
    }
  }
}
```

### 2. PR #8584 (2025.07.31) - 回滚修复
但是，PR #8584 又把这个修复给回滚了，恢复到只提取 `function` 字段的状态。

### 3. PR #9489 - 部分恢复
根据 issue 描述，PR #9489 部分恢复了 #8584 回滚的代码，但是**没有恢复工具格式的部分**。

## 当前代码状态

### 工具格式处理 (serving_chat.py)

当前代码在处理工具时会尝试两种格式（第464-470行）：

```python
try:
    # 第一次尝试使用原始格式
    prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(...)
except Exception as e:
    # 如果失败，转换工具格式
    tools = (
        [t if "function" in t else {"function": t} for t in tools]
        if tools
        else None
    )
```

这段代码说明：
1. 首先尝试使用工具的原始格式
2. 如果失败，会尝试将工具包装成 `{"function": t}` 格式
3. 但这只是降级处理，并非标准做法

### DeepSeek V3.2 编码格式 (encoding_dsv32.py)

在 DeepSeek V3.2 的编码实现中（第69-70行）：

```python
def tools_from_openai_format(tools):
    return [tool["function"] for tool in tools]
```

这个函数**主动剥离了外层包装**，只保留了 `function` 字段。

但是在输出时（第83-93行）：

```python
def tool_calls_to_openai_format(tool_calls):
    return [
        {
            "type": "function",
            "function": {
                "name": tool_call["name"],
                "arguments": tool_call["arguments"],
            },
        }
        for tool_call in tool_calls
    ]
```

输出时**又添加回了完整的 OpenAI 格式**。

## 问题所在

### 输入不一致
当前的问题是：
- **工具输入格式**：剥离了 `type` 和 `function` 包装，只保留内部字段
- **OpenAI 标准格式**：应该保持完整结构，包括 `type: "function"` 和 `function: {...}`

### 实际格式对比

**当前 SGLang 格式（不符合标准）：**
```json
{
  "description": "Python code sandbox, which can be used to execute Python code.",
  "name": "code_interpreter",
  "parameters": {
    "type": "object",
    "properties": { ... }
  }
}
```

**OpenAI 标准格式（应该使用的）：**
```json
{
  "type": "function",
  "function": {
    "description": "Python code sandbox, which can be used to execute Python code.",
    "name": "code_interpreter",
    "parameters": {
      "type": "object",
      "properties": { ... }
    }
  }
}
```

## 为什么要保持标准格式

1. **OpenAI API 兼容性**：很多客户端和工具期望标准的 OpenAI 格式
2. **一致性**：工具调用的**输出**已经使用标准格式（带 `type` 和 `function`），输入也应该保持一致
3. **互操作性**：标准格式使得不同系统之间的集成更加容易
4. **vLLM 对齐**：issue 中提到 PR #6556 的目的是与 vLLM 对齐，vLLM 使用标准格式

## 相关代码位置

1. **serving_chat.py** (约第113-120行)
   - 当前：`tools = [item.function.model_dump() for item in request.tools]`
   - 应该：`tools = [item.model_dump() for item in request.tools]`

2. **encoding_dsv32.py** (第69-70行)
   - 当前：`return [tool["function"] for tool in tools]`
   - 应该考虑是否需要保留完整格式

3. **protocol.py** (第509-514行)
   - Tool 模型定义已经包含了正确的结构

## 建议修复方案

恢复 PR #6556 中关于工具格式的修改，确保：
1. 输入工具保持 OpenAI 标准格式（包含 `type` 和 `function`）
2. 与工具调用输出格式保持一致
3. 兼容不同模型的聊天模板需求

## 参考

- Issue: https://github.com/sgl-project/sglang/issues/18478
- OpenAI Function Calling 文档：https://platform.openai.com/docs/guides/function-calling
- PR #6556: 修复输入工具格式
- PR #8584: 回滚 #6556
- PR #9489: 部分恢复但未恢复工具格式
