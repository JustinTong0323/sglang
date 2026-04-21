# Hy3-preview Usage

Hy3-preview is a large-scale language model (294B parameters, 20B active parameters) from Tencent Hunyuan team. SGLang supports serving Hy3-preview. This guide describes how to run Hy3-preview with native FP8.

## Installation

### Docker

```bash
docker pull lmsysorg/sglang:hy3-preview
```

### Build from Source

```bash
# Install SGLang
git clone https://github.com/sgl-project/sglang
cd sglang
pip3 install pip --upgrade
pip3 install "transformers>=5.6.0"
pip3 install -e "python"
```

## Launch Hy3-preview with SGLang

To serve the [Hy3-preview-FP8](https://huggingface.co/tencent/Hy3-preview-FP8) model on an 8xH20 GPU machine:

```bash
python3 -m sglang.launch_server \
  --model tencent/Hy3-preview-FP8 \
  --tp 8 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan \
  --served-model-name hy3-preview-fp8
```

### EAGLE Speculative Decoding

**Description**: SGLang supports Hy3-preview models with [EAGLE speculative decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html#eagle-decoding).

**Usage**:
Add `--speculative-algorithm`, `--speculative-num-steps`, `--speculative-eagle-topk`, and `--speculative-num-draft-tokens` to enable this feature. For example:

```bash
python3 -m sglang.launch_server \
  --model tencent/Hy3-preview-FP8 \
  --tp 8 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --speculative-algorithm EAGLE \
  --served-model-name hy3-preview-fp8
```

## OpenAI Client Example

First, install the OpenAI Python client:

```bash
uv pip install -U openai
```

You can use the OpenAI client as follows to verify thinking-mode responses.

```python
from openai import OpenAI

# If running SGLang locally with its default OpenAI-compatible port:
#   http://localhost:30000/v1
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello."},
]

# Thinking mode is disabled by default (no need to pass chat_template_kwargs).
resp = client.chat.completions.create(
    model="hy3-preview-fp8",
    messages=messages,
    temperature=1,
    max_tokens=4096,
)
print(resp.choices[0].message.content)

# Thinking mode is enabled only if 'reasoning_effort' and 'interleaved_thinking' are set in 'chat_template_kwargs'.
# 'reasoning_effort' supports: 'high', 'low', 'no_think'.
resp_think = client.chat.completions.create(
    model="hy3-preview-fp8",
    messages=messages,
    temperature=1,
    max_tokens=4096,
    extra_body={
      "chat_template_kwargs": {
          "reasoning_effort": "high",
          "interleaved_thinking": True
      },
    },
)
output_msg = resp_think.choices[0].message
# thinking content
print(output_msg.reasoning_content)
# response content
print(output_msg.content)
```

### cURL Usage

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hy3-preview-fp8",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello."}
    ],
    "temperature": 1,
    "max_tokens": 4096
  }'
```

## Benchmarking Results

For benchmarking, disable prefix caching by adding `--disable-radix-cache` to the server command.

The following example runs the benchmark on 8 H20 GPUs with 96 GB memory each.

```bash
python3 -m sglang.bench_serving \
    --backend sglang \
    --flush-cache \
    --dataset-name random \
    --random-range-ratio 1.0 \
    --random-input-len 4096 \
    --random-output-len 4096 \
    --num-prompts 160 \
    --max-concurrency 32 \
    --output-file hy3_preview_h20.jsonl \
    --model tencent/Hy3-preview-FP8 \
    --served-model-name hy3-preview-fp8
```

If successful, you will see the following output.

```shell
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 32
Successful requests:                     160
Benchmark duration (s):                  560.37
Total input tokens:                      655360
Total input text tokens:                 655360
Total generated tokens:                  655360
Total generated tokens (retokenized):    654700
Request throughput (req/s):              0.29
Input token throughput (tok/s):          1169.50
Output token throughput (tok/s):         1169.50
Peak output token throughput (tok/s):    1376.00
Peak concurrent requests:                64
Total token throughput (tok/s):          2339.01
Concurrency:                             31.99
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   112055.69
Median E2E Latency (ms):                 111967.15
P90 E2E Latency (ms):                    112712.63
P99 E2E Latency (ms):                    112721.22
---------------Time to First Token----------------
Mean TTFT (ms):                          3828.65
Median TTFT (ms):                        3786.40
P99 TTFT (ms):                           6531.28
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          26.43
Median TPOT (ms):                        26.41
P99 TPOT (ms):                           27.29
---------------Inter-Token Latency----------------
Mean ITL (ms):                           26.43
Median ITL (ms):                         25.78
P95 ITL (ms):                            27.88
P99 ITL (ms):                            28.32
Max ITL (ms):                            6010.40
==================================================
```
