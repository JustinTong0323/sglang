# VLM Encoder Data Parallelism

Vision-language models such as Qwen2.5-VL and InternVL spend a significant amount of time inside the vision encoder, especially when they need to ingest high-resolution images or multi-frame videos. A single request can easily exceed the compute or memory budget of one GPU, which limits throughput and inflates the tail latency of mixed workloads. **VLM encoder data parallelism (DP)** lets SGLang shard the vision encoder across multiple GPUs while keeping the language model portion of the graph unchanged.

Enabling this feature activates the sharded execution paths implemented in `python/sglang/srt/multimodal/mm_utils.py`. The dispatcher distributes image/video patches across the tensor-parallel group, runs the encoder in parallel, and then reorders the embeddings so that the downstream language model receives the same tensors it would have seen in single-GPU mode. This improves utilization whenever the request mix includes large or heterogeneous visual inputs.

## Supported stacks

| Model family | Vision encoder path | Notes |
| --- | --- | --- |
| Qwen2.5-VL (all sizes) | `Qwen2_5_VLForConditionalGeneration.visual` | Uses `run_dp_sharded_mrope_vision_model` with 3D RoPE metadata, so both images and videos benefit. |
| InternVL / InternVL2.5 | `InternVisionModel` inside `InternVLChatModel` | Uses `run_dp_sharded_vision_model` (no MRoPE). |

New multimodal models must opt in by wiring `get_global_server_args().mm_enable_dp_encoder` into their vision stack.

## Requirements and recommendations

- **Tensor parallel size** – The encoder DP implementation reuses the tensor-parallel communicator (`tensor_model_parallel_all_gather`). Set `--tensor-parallel-size / --tp` to the number of GPUs that should cooperate. A value of 1 is supported but offers no speedup.
- **Single-process launch** – The server assumes that all TP ranks live in the same NCCL communicator. When launching across nodes, follow the standard `--nnodes`, `--node-rank`, and `--dist-init-addr` flow so that tensor parallelism is already configured.
- **Memory planning** – Encoder DP only changes how the vision encoder is executed; the language model still occupies the same memory per TP rank. Continue to size `--mem-fraction-static`, paged attention memory, and KV cache replicas accordingly.
- **Concurrency tuning** – Large batches of multimodal preprocessing can still run on the host. Adjust `--mm-max-concurrent-calls` and `--mm-per-request-timeout` if you see queueing on the CPU side.

## How it works

1. **Input reshaping** – Incoming `pixel_values` and grid descriptors (`grid_thw` for mRoPE encoders) are flattened so that every visual patch becomes a chunk along the first dimension.
2. **Patch-aware load balancing** – `get_dp_encoder_lb_assignment()` sorts the images/videos by total patch count and greedily assigns them to TP ranks. This avoids stragglers when the batch mixes thumbnails and 4K captures.
3. **Sharded execution** – Each rank receives only the pixels assigned to it. For Qwen2.5-VL, both the data tensor and the per-sample `grid_thw` metadata are sliced before calling the underlying ViT.
4. **All-gather and reorder** – Outputs are padded to a uniform length, gathered with `tensor_model_parallel_all_gather`, and then sliced/reordered back into the original request order. The downstream language model therefore remains oblivious to the sharding.

Because the entire shuffle/gather process happens on device, no extra host transfers are introduced beyond the normal multimodal preprocessing.

## Enabling encoder DP

Add `--mm-enable-dp-encoder` to any `sglang.launch_server` invocation. You only need to configure tensor parallelism; traditional LLM data parallelism (`--dp`) is optional and unrelated to this feature.

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-VL-72B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --port 30000 \
  --mm-enable-dp-encoder \
  --mem-fraction-static 0.8
```

To run InternVL on the same 4-GPU node:

```bash
python -m sglang.launch_server \
  --model-path OpenGVLab/InternVL2_5-8B \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --mm-enable-dp-encoder \
  --mm-max-concurrent-calls 32 \
  --port 30001
```

In both cases, the server automatically shreds the vision encoder workload across the four TP ranks. You can expose the instances through SGLang Router in exactly the same way as single-GPU deployments; no router changes are required.

## Validation and troubleshooting

- **Nightly regression test** – `test/nightly/test_encoder_dp.py` launches a 4-GPU server with `--mm-enable-dp-encoder`, runs `lmms-eval` on MMMU, and asserts an accuracy floor. Use `pytest test/nightly/test_encoder_dp.py -k test_vlm_mmmu_benchmark` to reproduce the check locally.
- **Profiling** – Expect the vision encoder stages (`vision_model` or `visual`) to scale close to linearly with the TP size, provided that request batches contain enough patches. If the speedup plateaus, inspect the per-request patch histogram; extremely small images amortize poorly.
- **Mixed workloads** – Encoder DP is most valuable when large video or document images are mixed with text-only requests. Leave the flag unset if your deployment only ever receives small thumbnails—the extra all-gather can add a few milliseconds of overhead.

## Known limitations

- Only the vision encoder is sharded. The language model still follows the configured TP/PP/DP settings, so the total GPU count cannot be reduced.
- All TP ranks must have identical model weights and quantization settings. Mixing BF16 and FP8 ranks, or sharding the vision encoder differently from the LLM, is not supported.
- The current implementation assumes that the entire multimodal request fits into device memory after sharding. Extremely long videos may still need offline feature extraction or frame dropping.

For a full list of server arguments that interact with multimodal serving, see [`advanced_features/server_arguments.md`](server_arguments.md).
