# Breaking Changes Analysis: output_ids Streaming Fix

## 修复前后行为对比

### 修复前的行为

#### BatchStrOutput
```python
if state.obj.stream:
    # 增量输出（只返回新的 token）
    output_token_ids = state.output_ids[state.last_output_offset :]
    state.last_output_offset = len(state.output_ids)
else:
    # 累积输出（返回所有累积的 token）
    output_token_ids = state.output_ids.copy()
```

#### BatchTokenIDOutput
```python
if self.server_args.stream_output and state.obj.stream:
    # 增量输出（只返回新的 token）
    output_token_ids = state.output_ids[state.last_output_offset :]
    state.last_output_offset = len(state.output_ids)
else:
    # 累积输出（返回所有累积的 token）
    output_token_ids = state.output_ids.copy()
```

### 修复后的行为

#### BatchStrOutput & BatchTokenIDOutput
```python
# 统一使用增量输出
state.output_ids.extend(recv_obj.output_ids[i])
output_token_ids = state.output_ids[state.last_output_offset :]
state.last_output_offset = len(state.output_ids)
```

## 可能破坏的行为

### 1. 非 Streaming 请求的中间输出（Force Stream Interval）

**场景**: 非 streaming 请求（`stream=False`）在生成过程中**会**产生中间输出

**触发条件**: `scheduler_output_processor_mixin.py` line 818-822
```python
else:  # stream=False
    should_output = (
        len(req.output_ids) % DEFAULT_FORCE_STREAM_INTERVAL == 0  # 每 50 个 token
        if not self.model_config.is_multimodal_gen
        else False
    )
```

**修复前**:
- BatchStrOutput (stream=False): 每次返回所有累积的 token
- BatchTokenIDOutput (stream=False): 每次返回所有累积的 token
- 这些中间输出每 50 个 token 触发一次

**修复后**:
- 两者都只返回新的 token（增量）
- 中间输出只包含自上次输出以来的新 token

**影响评估**:
- ⚠️ **高风险**: 非 streaming 请求的中间输出行为会改变
- ⚠️ **影响范围**: 所有使用非 streaming 请求且依赖中间输出的代码
- ✅ **最终输出不受影响**: `_handle_finished_req_output` 有自己的逻辑，返回所有累积的 token
- ⚠️ **需要验证**: 检查是否有代码依赖非 streaming 中间输出的累积行为

### 2. BatchTokenIDOutput 在 stream_output=False 时的行为

**场景**: 使用 BatchTokenIDOutput 且 `stream_output=False`（默认值）的 streaming 请求

**修复前**:
- stream=True, stream_output=False: 返回所有累积的 token ❌（这是 bug！）
- 导致 GPT OSS tool calls 等多轮对话出现问题

**修复后**:
- 统一返回新的 token（增量）✅

**影响评估**:
- ✅ **这是修复**: 修复了 GPT OSS tool calls 的 bug
- ✅ **行为更正确**: 多轮对话现在能正确工作

### 3. 多轮对话中的 output_ids

**场景**: 多轮对话（如 GPT OSS tool calls）

**修复前**:
- 非 streaming: 每轮返回所有累积的 token（包含之前轮次）
- streaming + stream_output=False: 每轮返回所有累积的 token（bug）

**修复后**:
- 每轮只返回新的 token（增量）

**影响评估**:
- ✅ **这是修复**: 修复了多轮对话中的 bug
- ✅ **行为更正确**: 每轮只处理新的 token，不会重复处理

### 4. 最终输出（Finished Request）

**场景**: 请求完成时的最终输出

**代码位置**: `_handle_finished_req_output` (line 2046-2049)

```python
output_ids = state.output_ids  # 所有累积的 token
meta_info["completion_tokens"] = len(output_ids)
if is_stream:
    output_ids = [output_ids[-1]] if len(output_ids) > 0 else []  # streaming 只返回最后一个
```

**影响评估**:
- ✅ **不受影响**: 最终输出有自己的逻辑，不受中间输出的影响
- ✅ **行为一致**: streaming 和非 streaming 的最终输出行为不变

## 测试用例影响

### 需要检查的测试

1. **test_skip_tokenizer_init.py**:
   - `test_simple_decode`: 检查 `len(output_ids) == max_new_tokens`
   - `run_decode_stream`: 检查 streaming 和非 streaming 的 output_ids 是否一致
   - ⚠️ **可能受影响**: 如果测试依赖非 streaming 的中间输出行为

2. **test_srt_endpoint.py**:
   - 使用 `output_ids[-1]` 来获取最后一个 token
   - ✅ **不受影响**: 最终输出逻辑不变

3. **其他测试**:
   - 检查是否有测试依赖累积输出的行为

## 风险评估总结

### 高风险场景
- ⚠️ **非 streaming 请求的中间输出（Force Stream Interval）**: 
  - **触发频率**: 每 50 个 token 触发一次（`DEFAULT_FORCE_STREAM_INTERVAL = 50`）
  - **行为变化**: 从累积输出变为增量输出
  - **影响**: 所有依赖非 streaming 中间输出的代码
  - **缓解措施**: 
    - 检查是否有代码依赖中间输出的累积行为
    - 最终输出不受影响（使用 `_handle_finished_req_output`）

### 中等风险场景
- ✅ **无**: 其他场景都是 bug 修复或低风险

### 低风险/修复场景
- ✅ **BatchTokenIDOutput + stream_output=False**: 这是 bug 修复
- ✅ **多轮对话**: 这是 bug 修复
- ✅ **最终输出**: 不受影响

## 建议

1. **保留修复**: 修复是正确的，解决了 GPT OSS tool calls 的 bug
2. **⚠️ 关键验证**: 检查非 streaming 请求的中间输出使用情况
   - 搜索代码库中是否有依赖非 streaming 中间输出的代码
   - 验证非 streaming 请求的中间输出是否被使用
   - 如果被使用，需要评估影响
3. **测试验证**: 运行所有相关测试，特别是：
   - GPT OSS tool calls 测试
   - 多轮对话测试
   - 非 streaming 请求测试（特别是长文本生成，会触发 force stream interval）
4. **文档更新**: 在 release notes 中说明：
   - 修复了 output_ids streaming 的 bug
   - 现在统一使用增量输出（只返回新的 token）
   - ⚠️ **Breaking Change**: 非 streaming 请求的中间输出现在只包含新的 token（之前包含所有累积的 token）
   - 最终输出行为不变

## 与参考 Change 的对比

参考 change 显示：
```python
# 修复后（参考 change）:
if self.server_args.stream_output and state.obj.stream:
    # 增量输出
else:
    # 累积输出
```

我们的修复：
```python
# 统一增量输出（无论 stream_output 的值）
```

**差异**:
- 参考 change: **限制**了增量输出的使用（需要 `stream_output=True`）
- 我们的修复: **统一**使用增量输出

**评估**:
- ✅ 我们的修复更彻底，解决了根本问题
- ✅ 统一行为更可预测
- ⚠️ 但需要确保不会破坏现有行为
