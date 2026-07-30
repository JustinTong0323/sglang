import os

os.environ.setdefault("SGLANG_RAGGED_VERIFY_MODE", "static")

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=7200, stage="base-c", runner_config="4-gpu-h20")

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

TARGET_MODEL = "/root/models/ling-3.0-flash"
DRAFT_MODEL = "/root/models/ling-3.0-flash-dspark-draft"
GSM8K_DATA_PATH = "/root/datasets/gsm8k/test.jsonl"
GSM8K_SCORE_THRESHOLD = 0.90
STOP_RATE_THRESHOLD = 0.95
DSPARK_SCORE_DROP_TOLERANCE = 0.02
MAX_TOKENS = 1024

COMMON_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "4",
    "--mem-fraction-static",
    "0.55",
    "--max-running-requests",
    "4",
]
COMMON_ENV = {"SGLANG_STRICT_CONFIG_MUTATION": "0"}
PARITY_PROMPTS = [
    "Count from 1 to 30, separated by commas.",
    "What is 37 * 19? Answer with only the integer.",
    "Write the first twelve Fibonacci numbers, separated by commas.",
    "Translate 'The quick brown fox jumps over the lazy dog.' into Chinese.",
]


def _gsm8k_args(base_url):
    return SimpleNamespace(
        base_url=base_url,
        model=TARGET_MODEL,
        eval_name="gsm8k",
        api="completion",
        max_tokens=MAX_TOKENS,
        num_examples=None,
        num_threads=128,
        gsm8k_data_path=GSM8K_DATA_PATH,
    )


@contextmanager
def _server(extra_args=None, extra_env=None):
    env = dict(COMMON_ENV)
    env.update(extra_env or {})
    process = popen_launch_server(
        TARGET_MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=COMMON_ARGS + (extra_args or []),
        env=env,
    )
    try:
        yield DEFAULT_URL_FOR_TEST
    finally:
        kill_process_tree(process.pid)


def _run_gsm8k(base_url, label):
    metrics = run_eval(_gsm8k_args(base_url))
    if is_in_ci():
        write_github_step_summary(
            f"### {label}\n"
            f"score={metrics['score']:.4f}\n"
            f"stop_rate={metrics['stop_rate']:.4f}\n"
        )
    assert metrics["score"] >= GSM8K_SCORE_THRESHOLD
    assert metrics["stop_rate"] >= STOP_RATE_THRESHOLD
    return metrics


def _greedy_outputs(base_url):
    outputs = []
    for prompt in PARITY_PROMPTS:
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": TARGET_MODEL,
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": MAX_TOKENS,
            },
            timeout=300,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        assert choice["finish_reason"] == "stop"
        outputs.append(choice["text"])
    return outputs


def test_ling3_pure_target_and_dspark_gsm8k_parity():
    with _server() as base_url:
        baseline_metrics = _run_gsm8k(base_url, "pure target")
        baseline_outputs = _greedy_outputs(base_url)

    with _server(
        extra_args=[
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-draft-model-path",
            DRAFT_MODEL,
        ],
        extra_env={"SGLANG_RAGGED_VERIFY_MODE": "static"},
    ) as base_url:
        dspark_metrics = _run_gsm8k(base_url, "DSpark")
        dspark_outputs = _greedy_outputs(base_url)

    assert (
        dspark_metrics["score"]
        >= baseline_metrics["score"] - DSPARK_SCORE_DROP_TOLERANCE
    )
    assert dspark_outputs == baseline_outputs


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
