1"""Multimodal encoder accuracy tests: compare HF vs SGLang encoder outputs."""

import os
import socket
import tempfile
import unittest
from typing import List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.managers.mm_utils import embed_mm_inputs, init_mm_embedding_cache
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.parser.conversation import generate_chat_conv
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import download_image_with_retry


# Test the logits output between HF and SGLang
class VisionLLMLogitsBase(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.image_url = "https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true"
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model_path = ""
        cls.chat_template = ""
        cls.processor = ""
        cls.main_image = download_image_with_retry(cls.image_url)

    def compare_outputs(self, sglang_output: torch.Tensor, hf_output: torch.Tensor):
        # Convert to float32 for numerical stability if needed
        hf = hf_output.float()
        sg = sglang_output.float()

        # Basic shape and dtype comparison
        print("\n=== Basic Properties ===")
        print(f"Shapes match: {hf.shape == sg.shape}")
        print(f"HF shape: {hf.shape}, SGLang shape: {sg.shape}")
        print(f"HF dtype: {hf.dtype}, SGLang dtype: {sg.dtype}")

        # Move tensors to CPU for numpy operations
        hf_np = hf.cpu().numpy()
        sg_np = sg.cpu().numpy()

        # Statistical metrics
        print("\n=== Statistical Metrics ===")
        print(f"Mean absolute difference: {torch.mean(torch.abs(hf - sg)).item():.6f}")
        print(f"Max absolute difference: {torch.max(torch.abs(hf - sg)).item():.6f}")
        print(f"Mean squared error: {torch.mean((hf - sg) ** 2).item():.6f}")
        print(
            f"Root mean squared error: {torch.sqrt(torch.mean((hf - sg) ** 2)).item():.6f}"
        )

        # Cosine similarity (across feature dimension)
        cos_sim = F.cosine_similarity(hf, sg)
        print(f"Mean cosine similarity: {torch.mean(cos_sim).item():.6f}")
        print(f"Min cosine similarity: {torch.min(cos_sim).item():.6f}")

        # Find largest absolute differences
        print("\n=== Largest Absolute Differences ===")
        diffs = torch.abs(hf - sg)
        flat_diffs = diffs.flatten()

        # Get indices of top 10 differences
        top_k = 10
        top_values, top_flat_indices = torch.topk(flat_diffs, top_k)

        # Convert flat indices to multidimensional indices
        top_indices = np.unravel_index(top_flat_indices.cpu().numpy(), diffs.shape)

        print(f"\nTop {top_k} largest absolute differences:")
        print(
            "Index".ljust(30)
            + "Difference".ljust(15)
            + "HF Value".ljust(15)
            + "SGLang Value"
        )
        print("-" * 75)

        for i in range(top_k):
            # Get the index tuple for this difference
            idx = tuple(dim[i] for dim in top_indices)
        diff_val = top_values[i].item()
        hf_val = hf[idx].item()
        sg_val = sg[idx].item()

        # Format the index tuple and values
        idx_str = str(idx)
        print(f"{idx_str:<30}{diff_val:<15.6f}{hf_val:<15.6f}{sg_val:.6f}")

        np.testing.assert_allclose(hf_np, sg_np)

    def get_completion_request(self) -> ChatCompletionRequest:
        json_str = f"""
        {{
  "model": "{self.model_path}",
  "messages": [
    {{
      "role": "user",
      "content": [
        {{
          "type": "image_url",
          "image_url": {{
            "url": "{self.image_url}"
          }}
        }},
        {{
          "type": "text",
          "text": "What's in this picture?"
        }}
      ]
    }}
  ]
}}
        """

        return ChatCompletionRequest.model_validate_json(json_str)

    def get_processor_output(self, req: Optional[ChatCompletionRequest] = None):
        if req is None:
            req = self.get_completion_request()
        conv = generate_chat_conv(req, template_name=self.chat_template)
        text = conv.get_prompt()

        # Process inputs using processor
        # FIXME: the formal arguments may differ
        inputs = self.processor(
            text=[text],
            images=[self.main_image],
            return_tensors="pt",
        ).to(self.device)

        return inputs

    def get_sglang_model(self):
        self.model_runner = ModelRunner(
            model_config=ModelConfig(self.model_path, model_override_args="{}"),
            mem_fraction_static=0.8,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=12435,
            server_args=ServerArgs(
                model_path=self.model_path,
                disable_cuda_graph=True,
            ),
        )
        return self.model_runner.model


class TestMiniCPMV2_6Logits(VisionLLMLogitsBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_path = "openbmb/MiniCPM-V-2_6"
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.processor = AutoProcessor.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.chat_template = "minicpmv"

        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.hf_model = (
            AutoModel.from_pretrained(
                cls.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .eval()
            .to(cls.device)
        )
        init_mm_embedding_cache()

    async def test_vlm_embedding_output(self):
        """
        Compares the embedding output of vlm
        """
        inputs = self.get_processor_output()

        with torch.no_grad():
            # hf
            model_inputs = {
                "input_ids": inputs.input_ids,
                "image_bound": inputs.image_bound,
                "pixel_values": inputs.pixel_values,
                "tgt_sizes": inputs.tgt_sizes,
            }
            hf_output, _ = self.hf_model.get_vllm_embedding(
                model_inputs,
            )
            hf_output = hf_output.squeeze(0)

            # sglang
            model = self.get_sglang_model()
            input_ids = inputs["input_ids"].to(self.device).flatten()

            pixel_values = inputs["pixel_values"]
            tgt_sizes = inputs["tgt_sizes"]
            pixel_values_flat: List[torch.Tensor] = []
            tgt_sizes_flat: List[torch.Tensor] = []
            for pixel_b, tgt_b in zip(pixel_values, tgt_sizes):
                # per image
                if len(pixel_b) != len(tgt_b):
                    raise ValueError(
                        "Inconsistent N lengths, found: "
                        f"{len(pixel_b)} vs {len(tgt_b)}"
                    )
                for pixel_n, tgt_n in zip(pixel_b, tgt_b):
                    pixel_values_flat += [pixel_n]
                    tgt_sizes_flat += [tgt_n]

            im_start_id, im_end_id = (
                self.tokenizer.im_start_id,
                self.tokenizer.im_end_id,
            )
            slice_start_id, slice_end_id = (
                self.tokenizer.slice_start_id,
                self.tokenizer.slice_end_id,
            )

            image_offsets = BaseMultimodalProcessor.get_mm_items_offset_by_pair(
                input_ids=input_ids, mm_start_id=im_start_id, mm_end_id=im_end_id
            )
            slice_offsets = BaseMultimodalProcessor.get_mm_items_offset_by_pair(
                input_ids=input_ids, mm_start_id=slice_start_id, mm_end_id=slice_end_id
            )
            image_offsets.extend(slice_offsets)
            image_offsets = sorted(image_offsets)

            sglang_output = embed_mm_inputs(
                mm_inputs_list=[
                    MultimodalInputs(
                        mm_items=[
                            MultimodalDataItem(
                                feature=pixel_values_flat,
                                offsets=image_offsets,
                                tgt_size=tgt_sizes_flat,
                                modality=Modality.IMAGE,
                                pad_value=self.processor.tokenizer.unk_token_id,
                            )
                        ]
                    ),
                ],
                extend_prefix_lens=[0],
                extend_seq_lens=[input_ids.shape[0]],
                input_ids=input_ids,
                input_embedding=model.get_input_embeddings(),
                multimodal_model=model,
                placeholder_tokens={
                    Modality.IMAGE: self.processor.tokenizer.unk_token_id,
                },
            )

        self.compare_outputs(sglang_output, hf_output)


class TestMiniCPMV4Logits(VisionLLMLogitsBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_path = "openbmb/MiniCPM-V-4"
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.processor = AutoProcessor.from_pretrained(
            cls.model_path, trust_remote_code=True
        )
        cls.chat_template = "minicpmv"

        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.hf_model = (
            AutoModel.from_pretrained(
                cls.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .eval()
            .to(cls.device)
        )
        init_mm_embedding_cache()

    async def test_vlm_embedding_output(self):
        """
        Compares the embedding output of vlm
        """
        inputs = self.get_processor_output()

        with torch.no_grad():
            # hf
            model_inputs = {
                "input_ids": inputs.input_ids,
                "image_bound": inputs.image_bound,
                "pixel_values": inputs.pixel_values,
                "tgt_sizes": inputs.tgt_sizes,
            }
            hf_output = self.hf_model.get_input_embeddings()(inputs.input_ids)

            # sglang
            model = self.get_model()
            sglang_output = self.vlm_func(
                model,
                input_ids=inputs.input_ids.to(self.device),
                pixel_values=inputs.pixel_values,
                image_bound=inputs.image_bound.to(self.device),
                tgt_sizes=inputs.tgt_sizes.to(self.device),
                input_embedding=model.get_input_embeddings(),
                multimodal_model=model,
                placeholder_tokens={
                    Modality.IMAGE: self.processor.tokenizer.unk_token_id,
                },
            )

        self.compare_outputs(sglang_output, hf_output)


# ---------------------------------------------------------------------------
# Gemma 4 encoder accuracy: vision tower + audio tower vs HF reference
# ---------------------------------------------------------------------------


class TestGemma4EncoderAccuracy(unittest.TestCase):
    """Compare Gemma 4 vision and audio encoder outputs between HF and SGLang.

    For each encoder we compare:
      1. Raw tower output (before the multimodal embedder projection).
      2. Projected output (tower + ``embed_vision`` / ``embed_audio``).

    Inputs are random tensors so that the test is self-contained and does not
    depend on image / audio files.
    """

    MODEL_PATH = "gg-hf-gg/gemma-4-e4b-it"
    COSINE_THRESHOLD = 0.99

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -- HF model: extract encoder components, discard the rest -----------
        from transformers import (
            Gemma4ForConditionalGeneration as HFGemma4ForConditionalGeneration,
        )

        hf_full = HFGemma4ForConditionalGeneration.from_pretrained(
            cls.MODEL_PATH, torch_dtype=torch.bfloat16
        )

        cls.hf_vision_tower = hf_full.model.vision_tower.eval().to(cls.device)
        cls.hf_embed_vision = hf_full.model.embed_vision.eval().to(cls.device)

        cls.hf_audio_tower = None
        cls.hf_embed_audio = None
        cls.mel_bins = None
        if hf_full.model.audio_tower is not None:
            cls.hf_audio_tower = hf_full.model.audio_tower.eval().to(cls.device)
            cls.hf_embed_audio = hf_full.model.embed_audio.eval().to(cls.device)
            config = AutoConfig.from_pretrained(cls.MODEL_PATH)
            cls.mel_bins = config.audio_config.input_feat_size

        del hf_full
        torch.cuda.empty_cache()

        # -- SGLang model via ModelRunner -------------------------------------
        cls.model_runner = ModelRunner(
            model_config=ModelConfig(cls.MODEL_PATH, model_override_args="{}"),
            mem_fraction_static=0.8,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            moe_ep_rank=0,
            moe_ep_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=12435,
            server_args=ServerArgs(
                model_path=cls.MODEL_PATH,
                disable_cuda_graph=True,
                mm_attention_backend="sdpa",
            ),
        )
        cls.sg_model = cls.model_runner.model

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _cosine_stats(a: torch.Tensor, b: torch.Tensor):
        cos = F.cosine_similarity(a.float(), b.float())
        return cos.mean().item(), cos.min().item()

    def _assert_cosine_close(self, hf: torch.Tensor, sg: torch.Tensor, label: str):
        mean_cos, min_cos = self._cosine_stats(hf, sg)
        print(f"  {label}: mean_cos={mean_cos:.6f}  min_cos={min_cos:.6f}")
        self.assertGreater(
            min_cos,
            self.COSINE_THRESHOLD,
            f"{label} min cosine {min_cos:.6f} < {self.COSINE_THRESHOLD}",
        )

    # -- vision ---------------------------------------------------------------

    def test_vision_encoder(self):
        """Vision tower + embed_vision should match HF on random pixels."""
        pixel_values = torch.randn(
            1, 3, 768, 768, device=self.device, dtype=torch.bfloat16
        )

        with torch.no_grad():
            # HF: last_hidden_state is [1, num_real_tokens, hidden] (padding stripped)
            hf_out = self.hf_vision_tower(pixel_values)
            hf_tokens = hf_out.last_hidden_state.squeeze(0)
            hf_projected = self.hf_embed_vision(hf_tokens.unsqueeze(0)).squeeze(0)

            # SGLang: returns (pooled, pooler_mask) with mask True = valid
            sg_pooled, sg_mask = self.sg_model.vision_tower(pixel_values)
            sg_tokens = torch.cat([hs[m] for hs, m in zip(sg_pooled, sg_mask)])
            sg_projected = self.sg_model.embed_vision(sg_tokens.unsqueeze(0)).squeeze(0)

        self.assertEqual(hf_tokens.shape, sg_tokens.shape)
        print()
        self._assert_cosine_close(hf_tokens, sg_tokens, "vision tower")
        self._assert_cosine_close(hf_projected, sg_projected, "vision projected")

    # -- audio ----------------------------------------------------------------

    def test_audio_encoder(self):
        """Audio tower + embed_audio should match HF on random mel input."""
        if self.hf_audio_tower is None:
            self.skipTest("Model does not have an audio tower")

        num_frames = 200
        audio_mel = torch.randn(
            1, num_frames, self.mel_bins, device=self.device, dtype=torch.bfloat16
        )
        audio_mel_mask = torch.zeros(
            1, num_frames, device=self.device, dtype=torch.bool
        )

        with torch.no_grad():
            # HF: returns (encodings, mask) — does NOT zero-fill padding
            hf_enc, hf_mask = self.hf_audio_tower(audio_mel, audio_mel_mask)
            hf_valid_mask = ~hf_mask
            hf_valid = hf_enc[hf_valid_mask.unsqueeze(-1).expand_as(hf_enc)].reshape(
                -1, hf_enc.shape[-1]
            )
            hf_projected = self.hf_embed_audio(hf_valid.unsqueeze(0)).squeeze(0)

            # SGLang: returns (encodings, mask) — zero-fills padding positions
            sg_enc, sg_mask = self.sg_model.audio_tower(audio_mel, audio_mel_mask)
            sg_valid_mask = ~sg_mask
            sg_valid = sg_enc[sg_valid_mask.unsqueeze(-1).expand_as(sg_enc)].reshape(
                -1, sg_enc.shape[-1]
            )
            sg_projected = self.sg_model.embed_audio(sg_valid.unsqueeze(0)).squeeze(0)

        self.assertEqual(hf_valid.shape, sg_valid.shape)
        print()
        self._assert_cosine_close(hf_valid, sg_valid, "audio tower")
        self._assert_cosine_close(hf_projected, sg_projected, "audio projected")


# ---------------------------------------------------------------------------
# Gemma 4 encoder accuracy at TP=2: compare SGLang (TP=2) vs HF reference
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        s.listen(1)
        return s.getsockname()[1]


def _tp2_encoder_worker(
    local_rank: int,
    world_size: int,
    nccl_port: int,
    model_path: str,
    mel_bins: int,
    num_frames: int,
    result_file: str,
):
    """Worker spawned by mp.spawn — loads SGLang model with TP and runs encoders."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(world_size))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    model_runner = ModelRunner(
        model_config=ModelConfig(model_path, model_override_args="{}"),
        mem_fraction_static=0.5,
        gpu_id=local_rank,
        tp_rank=local_rank,
        tp_size=world_size,
        moe_ep_rank=0,
        moe_ep_size=1,
        pp_rank=0,
        pp_size=1,
        nccl_port=nccl_port,
        server_args=ServerArgs(
            model_path=model_path,
            disable_cuda_graph=True,
            mm_attention_backend="sdpa",
            mem_fraction_static=0.5,
        ),
    )
    sg_model = model_runner.model

    # Deterministic input — identical on every rank.
    torch.manual_seed(42)
    audio_mel = torch.randn(
        1, num_frames, mel_bins, device=device, dtype=torch.bfloat16
    )
    audio_mel_mask = torch.zeros(1, num_frames, device=device, dtype=torch.bool)
    pixel_values = torch.randn(1, 3, 768, 768, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        # Audio
        sg_audio_enc, sg_audio_mask = sg_model.audio_tower(audio_mel, audio_mel_mask)
        sg_audio_valid_mask = ~sg_audio_mask
        sg_audio_valid = sg_audio_enc[
            sg_audio_valid_mask.unsqueeze(-1).expand_as(sg_audio_enc)
        ].reshape(-1, sg_audio_enc.shape[-1])
        sg_audio_proj = sg_model.embed_audio(sg_audio_valid.unsqueeze(0)).squeeze(0)

        # Vision
        sg_vis_pooled, sg_vis_mask = sg_model.vision_tower(pixel_values)
        sg_vis_tokens = torch.cat([hs[m] for hs, m in zip(sg_vis_pooled, sg_vis_mask)])
        sg_vis_proj = sg_model.embed_vision(sg_vis_tokens.unsqueeze(0)).squeeze(0)

    if local_rank == 0:
        torch.save(
            {
                "audio_valid": sg_audio_valid.cpu(),
                "audio_projected": sg_audio_proj.cpu(),
                "vision_tokens": sg_vis_tokens.cpu(),
                "vision_projected": sg_vis_proj.cpu(),
            },
            result_file,
        )


class TestGemma4EncoderAccuracyTP2(unittest.TestCase):
    """Compare Gemma 4 vision + audio encoder outputs at TP=2 against HF.

    Uses ``mp.spawn`` to create 2 workers that jointly load the SGLang model
    with tensor parallelism, then compares rank-0 output with the HF reference
    computed in the parent process.
    """

    MODEL_PATH = "gg-hf-gg/gemma-4-e4b-it"
    # TP=2 all-reduce introduces small bf16 rounding that compounds across
    # 12 conformer blocks; 0.98 is the practical floor.
    COSINE_THRESHOLD = 0.98
    NUM_FRAMES = 200

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise unittest.SkipTest("Need >= 2 GPUs for TP=2 test")

        cls.device = torch.device("cuda:0")
        config = AutoConfig.from_pretrained(cls.MODEL_PATH)
        cls.mel_bins = config.audio_config.input_feat_size

        # -- HF reference (run on GPU 0, then free) ----------------------------
        from transformers import (
            Gemma4ForConditionalGeneration as HFGemma4ForConditionalGeneration,
        )

        hf_full = HFGemma4ForConditionalGeneration.from_pretrained(
            cls.MODEL_PATH, torch_dtype=torch.bfloat16
        )
        hf_audio_tower = hf_full.model.audio_tower.eval().to(cls.device)
        hf_embed_audio = hf_full.model.embed_audio.eval().to(cls.device)
        hf_vision_tower = hf_full.model.vision_tower.eval().to(cls.device)
        hf_embed_vision = hf_full.model.embed_vision.eval().to(cls.device)
        del hf_full
        torch.cuda.empty_cache()

        torch.manual_seed(42)
        audio_mel = torch.randn(
            1, cls.NUM_FRAMES, cls.mel_bins, device=cls.device, dtype=torch.bfloat16
        )
        audio_mel_mask = torch.zeros(
            1, cls.NUM_FRAMES, device=cls.device, dtype=torch.bool
        )
        pixel_values = torch.randn(
            1, 3, 768, 768, device=cls.device, dtype=torch.bfloat16
        )

        with torch.no_grad():
            hf_enc, hf_mask = hf_audio_tower(audio_mel, audio_mel_mask)
            hf_valid_mask = ~hf_mask
            cls.hf_audio_valid = (
                hf_enc[hf_valid_mask.unsqueeze(-1).expand_as(hf_enc)]
                .reshape(-1, hf_enc.shape[-1])
                .cpu()
            )
            cls.hf_audio_proj = (
                hf_embed_audio(cls.hf_audio_valid.unsqueeze(0).to(cls.device))
                .squeeze(0)
                .cpu()
            )

            hf_vis_out = hf_vision_tower(pixel_values)
            cls.hf_vis_tokens = hf_vis_out.last_hidden_state.squeeze(0).cpu()
            cls.hf_vis_proj = (
                hf_embed_vision(cls.hf_vis_tokens.unsqueeze(0).to(cls.device))
                .squeeze(0)
                .cpu()
            )

        del hf_audio_tower, hf_embed_audio, hf_vision_tower, hf_embed_vision
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        # -- Run SGLang at TP=2 via mp.spawn -----------------------------------
        nccl_port = _find_free_port()
        cls._result_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        cls._result_file.close()

        mp.spawn(
            _tp2_encoder_worker,
            args=(
                2,
                nccl_port,
                cls.MODEL_PATH,
                cls.mel_bins,
                cls.NUM_FRAMES,
                cls._result_file.name,
            ),
            nprocs=2,
            join=True,
        )

        cls.sg_results = torch.load(cls._result_file.name, weights_only=True)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_result_file"):
            os.unlink(cls._result_file.name)

    @staticmethod
    def _cosine_stats(a: torch.Tensor, b: torch.Tensor):
        cos = F.cosine_similarity(a.float(), b.float())
        return cos.mean().item(), cos.min().item()

    def _assert_cosine_close(self, hf: torch.Tensor, sg: torch.Tensor, label: str):
        mean_cos, min_cos = self._cosine_stats(hf, sg)
        print(f"  {label}: mean_cos={mean_cos:.6f}  min_cos={min_cos:.6f}")
        self.assertGreater(
            min_cos,
            self.COSINE_THRESHOLD,
            f"{label} min cosine {min_cos:.6f} < {self.COSINE_THRESHOLD}",
        )

    def test_audio_encoder_tp2(self):
        """Audio tower + embed_audio at TP=2 should match HF reference."""
        sg_valid = self.sg_results["audio_valid"]
        sg_proj = self.sg_results["audio_projected"]
        self.assertEqual(self.hf_audio_valid.shape, sg_valid.shape)
        print()
        self._assert_cosine_close(self.hf_audio_valid, sg_valid, "audio tower TP=2")
        self._assert_cosine_close(self.hf_audio_proj, sg_proj, "audio projected TP=2")

    def test_vision_encoder_tp2(self):
        """Vision tower + embed_vision at TP=2 should match HF reference."""
        sg_tokens = self.sg_results["vision_tokens"]
        sg_proj = self.sg_results["vision_projected"]
        self.assertEqual(self.hf_vis_tokens.shape, sg_tokens.shape)
        print()
        self._assert_cosine_close(self.hf_vis_tokens, sg_tokens, "vision tower TP=2")
        self._assert_cosine_close(self.hf_vis_proj, sg_proj, "vision projected TP=2")
