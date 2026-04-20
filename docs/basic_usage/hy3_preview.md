# HY3-Preview Usage

HY3 Preview is a significantly scaled-up language model (294B parameters, 20B active parameters) provided by Tencent Hunyuan Teams. SGLang has supported HY3-Preview. This guide describes how to run HY3 Preview with native FP8.

## Installation

### Docker

```bash

```

### Build From Source

```bash
# Install SGLang
git clone https://github.com/sgl-project/sglang
cd sglang
pip3 install pip --upgrade
pip3 install -e "python"
```

## Launch HY3-Preview with SGLang

To serve [HY3-Preview-FP8](https://huggingface.co/tencent/HY3-FP8) models on 8xH20 GPUs:
```bash
python3 -m sglang.launch_server \
  --model tencent/HY3-FP8 \
  --tp 8 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan \
  --served-model-name hy-3-fp8
```

### EAGLE Speculative Decoding

**Description**: SGLang has supported HY3-Preview models with [EAGLE speculative decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding).

**Usage**:
Add arguments `--speculative-algorithm`, `--speculative-num-steps`, `--speculative-eagle-topk` and `--speculative-num-draft-tokens` to enable this feature. For example:

``` bash
python3 -m sglang.launch_server \
  --model tencent/HY3-FP8 \
  --tp 8 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-algorithm EAGLE \
  --served-model-name hy-3-fp8
```

## OpenAI Client Example

First, install the OpenAI Python client:

```bash
uv pip install -U openai
```

You can use the OpenAI client as follows to  verify the think mode.

```python
from openai import OpenAI

# If running vLLM locally with its default OpenAI-compatible port:
#   http://localhost:8000/v1
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

# Thinking ON (default if you omit chat_template_kwargs)
resp_on = client.chat.completions.create(
    model="hy-3-fp8",
    messages=messages,
    temperature=1,
    max_tokens=4096,
)
print(resp_on.choices[0].message.content)
```

### cURL Usage

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d ' {
    "model": "hy-3-fp8",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello."}
    ],
    "temperature": 1,
    "max_tokens": 4096
  } '
```

## Benchmarking Results

For benchmarking, disable prefix caching by adding `--disable-radix-cache` to the server command.

- The following uses H20(96GB)*8 as an example to demonstrate how to run the benchmark.

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
    --model tencent/HY3-FP8 \
    --served-model-name hy-3-fp8
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
