import concurrent
import concurrent.futures
import dataclasses
import multiprocessing as mp
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import BaseImageProcessorFast

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.utils import (
    is_npu,
    load_audio,
    load_image,
    load_video,
    logger,
    print_warning_once,
)

_is_npu = is_npu()


@dataclasses.dataclass
class BaseMultiModalProcessorOutput:
    # input_text, with each frame of video/image represented with a image_token
    input_text: str

    # frames loaded from image, in given order
    images: Optional[list[Union[Image.Image, dict]]] = dataclasses.field(
        default_factory=list
    )

    # videos
    videos: Optional[list[Union[torch.Tensor, dict]]] = dataclasses.field(
        default_factory=list
    )

    # audios
    audios: Optional[list[Union[np.ndarray, dict]]] = dataclasses.field(
        default_factory=list
    )

    def organize_results(self) -> List[Tuple[Modality, Any]]:
        """

        :return: a list of results, with their corresponding modalities
        """
        return (
            [(Modality.IMAGE, data) for data in self.images]
            + [(Modality.VIDEO, data) for data in self.videos]
            + [(Modality.AUDIO, data) for data in self.audios]
        )


@dataclasses.dataclass
class MultimodalSpecialTokens:
    image_token: Optional[Union[str, List[str]]] = None
    video_token: Optional[Union[str, List[str]]] = None
    audio_token: Optional[Union[str, List[str]]] = None

    image_token_id: Optional[int] = None
    video_token_id: Optional[int] = None
    audio_token_id: Optional[int] = None

    image_token_regex: Optional[re.Pattern] = None
    video_token_regex: Optional[re.Pattern] = None
    audio_token_regex: Optional[re.Pattern] = None

    combined_regex: Optional[re.Pattern] = None

    def build(self, processor):
        self.convert_to_strs(processor)
        self.parse_regex()
        self.get_combined_regex()
        return self

    def convert_to_str(self, token: Union[str, int], processor) -> str:
        if token is None:
            return token
        if isinstance(token, str):
            return token
        return processor.tokenizer.convert_ids_to_tokens([token])[0]

    def convert_to_strs(self, processor):
        if not self.image_token:
            self.image_token = self.convert_to_str(self.image_token_id, processor)
        if not self.video_token:
            self.video_token = self.convert_to_str(self.video_token_id, processor)
        if not self.audio_token:
            self.audio_token = self.convert_to_str(self.audio_token_id, processor)

    def get_modality_of_token(self, token: str) -> Optional[Modality]:
        """
        :return: the modality associated with the given token, if the token is a special_token or matches with the multimodal token regex
        """
        modality = {
            self.image_token: Modality.IMAGE,
            self.video_token: Modality.VIDEO,
            self.audio_token: Modality.AUDIO,
        }.get(token)
        if modality:
            return modality

        for regex, modality in [
            (self.image_token_regex, Modality.IMAGE),
            (self.video_token_regex, Modality.VIDEO),
            (self.audio_token_regex, Modality.AUDIO),
        ]:
            if regex and regex.match(token):
                return modality

        return None

    def get_token_id_by_modality(self, modality: Modality) -> Optional[int]:
        return {
            Modality.IMAGE: self.image_token_id,
            Modality.MULTI_IMAGES: self.image_token_id,
            Modality.VIDEO: self.video_token_id,
            Modality.AUDIO: self.audio_token_id,
        }.get(modality)

    def parse_regex(self):
        if self.image_token_regex is None and self.image_token is not None:
            self.image_token_regex = re.compile(re.escape(self.image_token))
        if self.video_token_regex is None and self.video_token is not None:
            self.video_token_regex = re.compile(re.escape(self.video_token))
        if self.audio_token_regex is None and self.audio_token is not None:
            self.audio_token_regex = re.compile(re.escape(self.audio_token))

    def get_combined_regex(self) -> re.Pattern:
        """
        Builds and returns a regex, used to split input str into tokens (with mm special tokens)
        """
        if self.combined_regex:
            return self.combined_regex
        tokens = [
            self.image_token_regex,
            self.video_token_regex,
            self.audio_token_regex,
        ]
        patterns = []
        flags = 0
        for t in tokens:
            if t is not None:
                patterns.append(t.pattern)
                flags |= t.flags
        combined = "(" + "|".join(f"(?:{p})" for p in patterns) + ")"
        self.combined_regex = re.compile(combined, flags)
        return self.combined_regex


class BaseMultimodalProcessor(ABC):
    models = []

    _FAST_IMAGE_MEMORY_KEY = "_sglang_fast_image_processor_memory_bytes"
    _FAST_IMAGE_DEVICE_KEY = "_sglang_fast_image_processor_device"
    _FAST_IMAGE_USED_FAST_KEY = "_sglang_fast_image_processor_used_fast"

    def __init__(
        self, hf_config, server_args, _processor, transport_mode, *args, **kwargs
    ):
        self.hf_config = hf_config
        self._processor = _processor
        self.server_args = server_args
        self.transport_mode = transport_mode

        # FIXME: not accurate, model and image specific
        self.NUM_TOKEN_PER_FRAME = 330

        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SGLANG_IO_WORKERS", 4))
        )
        self.cpu_executor = concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context("fork"),
            max_workers=int(os.environ.get("SGLANG_CPU_WORKERS", os.cpu_count())),
        )

        self._fast_image_processor_cap_ratio = self._get_env_float(
            "SGLANG_FAST_IMAGE_PROCESSOR_CAP_RATIO", 0.25
        )
        self._fast_image_processor_cap_ratio = min(
            max(self._fast_image_processor_cap_ratio, 0.05), 1.0
        )
        self._fast_image_processor_min_free_bytes = int(
            float(os.environ.get("SGLANG_FAST_IMAGE_PROCESSOR_MIN_FREE_MB", "512"))
            * (1024**2)
        )
        self._fast_image_processor_overhead_factor = self._get_env_float(
            "SGLANG_FAST_IMAGE_PROCESSOR_OVERHEAD_FACTOR", 4.0
        )
        self._fast_image_processor_constant_overhead_bytes = max(
            int(
                float(os.environ.get("SGLANG_FAST_IMAGE_PROCESSOR_BASE_MB", "128"))
                * (1024**2)
            ),
            0,
        )
        self._fast_image_processor_safety_factor = self._get_env_float(
            "SGLANG_FAST_IMAGE_PROCESSOR_SAFETY_FACTOR", 1.6
        )
        self._fast_image_processor_allow_empty_cache = (
            os.environ.get("SGLANG_FAST_IMAGE_PROCESSOR_EMPTY_CACHE_ON_REJECT", "0")
            .strip()
            .lower()
            in {"1", "true", "yes"}
        )
        self._fast_image_processor_dtype_size = (
            self._infer_fast_image_processor_dtype_size()
        )
        self._fast_image_processor_device_index: Optional[int] = None
        self._fast_image_processor_memory_cap_bytes: Optional[int] = None
        self._fast_image_processor_stats = {
            "last_estimated_bytes": 0,
            "last_free_bytes": None,
            "last_allowed_bytes": None,
            "cap_bytes": None,
            "rejection_count": 0,
            "last_used_fast": None,
            "last_device": None,
        }
        self._init_fast_image_processor_device()

        # Mapping from attribute names to modality types
        self.ATTR_NAME_TO_MODALITY = {
            # Image-related attributes
            "pixel_values": Modality.IMAGE,
            "image_sizes": Modality.IMAGE,
            "image_grid_thw": Modality.IMAGE,
            "image_attention_mask": Modality.IMAGE,
            "image_emb_mask": Modality.IMAGE,
            "images_spatial_crop": Modality.IMAGE,
            "images_crop": Modality.IMAGE,
            "tgt_size": Modality.IMAGE,
            "image_grid_hws": Modality.IMAGE,
            "aspect_ratio_ids": Modality.IMAGE,
            "aspect_ratio_mask": Modality.IMAGE,
            "num_patches": Modality.IMAGE,
            "patch_pixel_values": Modality.IMAGE,
            "block_sizes": Modality.IMAGE,
            # Audio-related attributes
            "audio_features": Modality.AUDIO,
            "audio_feature_lens": Modality.AUDIO,
            "input_features": Modality.AUDIO,
            "input_features_mask": Modality.AUDIO,
            "audio_attention_mask": Modality.AUDIO,
            "feature_attention_mask": Modality.AUDIO,
            # Video-related attributes
            "pixel_values_videos": Modality.VIDEO,
            "second_per_grid_ts": Modality.VIDEO,
            "video_grid_thw": Modality.VIDEO,
            # Generic attributes that could apply to multiple modalities
            # "precomputed_embeddings" - handled specially as it can be any modality
        }

        # name of the feature filed
        # TODO: pass from processors
        self.FEATURE_NAMES = [
            "pixel_values",
            "pixel_values_videos",
            "audio_features",
            "input_features",
        ]

    @staticmethod
    def _get_env_float(env_name: str, default: float) -> float:
        value = os.environ.get(env_name)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            print_warning_once(
                f"Invalid value '{value}' for {env_name}. Falling back to default {default}."
            )
            return default

    def _init_fast_image_processor_device(self) -> None:
        if _is_npu:
            self._fast_image_processor_device_index = None
            self._fast_image_processor_memory_cap_bytes = None
            return

        if not torch.cuda.is_available():
            self._fast_image_processor_device_index = None
            self._fast_image_processor_memory_cap_bytes = None
            return

        try:
            device_index = torch.cuda.current_device()
        except RuntimeError:
            device_index = 0

        self._fast_image_processor_device_index = device_index

        try:
            total_memory = torch.cuda.get_device_properties(device_index).total_memory
        except RuntimeError:
            total_memory = None

        if total_memory is None or total_memory <= 0:
            self._fast_image_processor_memory_cap_bytes = None
            return

        cap_bytes = int(total_memory * self._fast_image_processor_cap_ratio)
        env_cap = os.environ.get("SGLANG_FAST_IMAGE_PROCESSOR_CAP_BYTES")
        if env_cap:
            try:
                cap_override = int(float(env_cap))
                if cap_override > 0:
                    cap_bytes = min(cap_bytes, cap_override)
            except (TypeError, ValueError):
                print_warning_once(
                    f"Invalid value '{env_cap}' for SGLANG_FAST_IMAGE_PROCESSOR_CAP_BYTES."
                )

        self._fast_image_processor_memory_cap_bytes = max(cap_bytes, 0)

    @staticmethod
    def _dtype_size(dtype) -> int:
        if dtype is None:
            return 0
        if isinstance(dtype, torch.dtype):
            return torch.tensor([], dtype=dtype).element_size()
        if isinstance(dtype, np.dtype):
            return int(dtype.itemsize)
        if isinstance(dtype, str):
            try:
                return int(np.dtype(dtype).itemsize)
            except (TypeError, ValueError):
                torch_dtype = getattr(torch, dtype, None)
                if isinstance(torch_dtype, torch.dtype):
                    return torch.tensor([], dtype=torch_dtype).element_size()
        return 0

    def _infer_fast_image_processor_dtype_size(self) -> int:
        processor = getattr(self._processor, "image_processor", None)
        dtype_candidates = []
        if processor is not None:
            for attr in ("torch_dtype", "dtype", "target_dtype", "_torch_dtype"):
                val = getattr(processor, attr, None)
                if val is not None:
                    dtype_candidates.append(val)
        for candidate in dtype_candidates:
            size = self._dtype_size(candidate)
            if size:
                return size
        return self._dtype_size(torch.float32)

    def _estimate_fast_image_processor_memory_bytes(
        self,
        images=None,
        videos=None,
        audios=None,
    ) -> Optional[int]:
        total_bytes = 0

        def accumulate(item):
            nonlocal total_bytes
            if item is None:
                return
            if isinstance(item, (list, tuple)):
                for sub in item:
                    accumulate(sub)
                return
            if isinstance(item, dict):
                return
            if isinstance(item, Image.Image):
                width, height = item.size
                bands = item.getbands() or []
                channels = len(bands) if bands else 3
                total_bytes += (
                    width
                    * height
                    * channels
                    * max(self._fast_image_processor_dtype_size, 1)
                )
                return
            if torch.is_tensor(item):
                total_bytes += item.numel() * self._dtype_size(item.dtype)
                return
            if isinstance(item, np.ndarray):
                total_bytes += int(item.size * item.itemsize)
                return

        accumulate(images)
        accumulate(videos)
        accumulate(audios)

        if total_bytes == 0:
            return 0

        estimated = int(
            total_bytes * max(self._fast_image_processor_overhead_factor, 1.0)
        )
        estimated += self._fast_image_processor_constant_overhead_bytes
        return estimated

    def _should_use_fast_image_processor(self, estimated_bytes: Optional[int]) -> bool:
        if estimated_bytes is None or estimated_bytes <= 0:
            return True

        if _is_npu:
            self._fast_image_processor_stats.update(
                {
                    "last_estimated_bytes": int(estimated_bytes),
                    "last_used_fast": True,
                    "last_device": "npu",
                }
            )
            return True

        if not torch.cuda.is_available():
            self._fast_image_processor_stats["rejection_count"] += 1
            return False

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except RuntimeError as exc:
            logger.debug("Failed to query CUDA memory info: %s", exc)
            return True

        cap_bytes = self._fast_image_processor_memory_cap_bytes
        if cap_bytes is None or cap_bytes <= 0:
            cap_bytes = int(total_bytes * self._fast_image_processor_cap_ratio)
        cap_bytes = max(cap_bytes, int(estimated_bytes))

        allowed_bytes = min(
            cap_bytes,
            max(free_bytes - self._fast_image_processor_min_free_bytes, 0),
        )

        self._fast_image_processor_stats.update(
            {
                "last_estimated_bytes": int(estimated_bytes),
                "last_free_bytes": int(free_bytes),
                "last_allowed_bytes": int(allowed_bytes),
                "cap_bytes": int(cap_bytes),
            }
        )

        if allowed_bytes <= 0:
            if self._fast_image_processor_allow_empty_cache:
                torch.cuda.empty_cache()
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                allowed_bytes = min(
                    cap_bytes,
                    max(free_bytes - self._fast_image_processor_min_free_bytes, 0),
                )
                self._fast_image_processor_stats.update(
                    {
                        "last_free_bytes": int(free_bytes),
                        "last_allowed_bytes": int(allowed_bytes),
                    }
                )
            else:
                self._fast_image_processor_stats["rejection_count"] += 1
                return False

        if estimated_bytes * self._fast_image_processor_safety_factor <= allowed_bytes:
            return True

        self._fast_image_processor_stats["rejection_count"] += 1
        return False

    def _attach_processor_output_metadata(self, processor_output, metadata: dict) -> None:
        if processor_output is None or not metadata:
            return
        for key, value in metadata.items():
            try:
                processor_output[key] = value
            except (TypeError, AttributeError):
                if hasattr(processor_output, "data") and isinstance(
                    processor_output.data, dict
                ):
                    processor_output.data[key] = value
                else:
                    setattr(processor_output, key, value)

    @staticmethod
    def _iter_processor_output_sources(processor_output):
        if isinstance(processor_output, dict):
            yield processor_output
        data = getattr(processor_output, "data", None)
        if isinstance(data, dict):
            yield data

    def _extract_from_processor_output(self, processor_output, key):
        if processor_output is None:
            return None
        for source in self._iter_processor_output_sources(processor_output):
            if key in source:
                return source[key]
        return getattr(processor_output, key, None)

    def _get_fast_image_processor_metadata(self, processor_output) -> dict:
        metadata = {}
        mem_bytes = self._extract_from_processor_output(
            processor_output, self._FAST_IMAGE_MEMORY_KEY
        )
        if mem_bytes is not None:
            try:
                metadata["fast_image_processor_memory_bytes"] = int(mem_bytes)
            except (TypeError, ValueError):
                pass

        device = self._extract_from_processor_output(
            processor_output, self._FAST_IMAGE_DEVICE_KEY
        )
        if device is not None:
            metadata["fast_image_processor_device"] = str(device)

        used_fast = self._extract_from_processor_output(
            processor_output, self._FAST_IMAGE_USED_FAST_KEY
        )
        if used_fast is not None:
            metadata["fast_image_processor_used_fast"] = bool(used_fast)

        return metadata

    def get_fast_image_processor_stats(self) -> dict:
        return dict(self._fast_image_processor_stats)

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ) -> dict:
        """
        process multimodal data with transformers AutoProcessor
        """
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
        if audios:
            if self._processor.__class__.__name__ in {
                "Gemma3nProcessor",
                "Qwen2AudioProcessor",
                "Qwen3OmniMoeProcessor",
            }:
                # Note(Xinyuan): for gemma3n, ref: https://github.com/huggingface/transformers/blob/ccf2ca162e33f381e454cdb74bf4b41a51ab976d/src/transformers/models/gemma3n/processing_gemma3n.py#L107
                kwargs["audio"] = audios
            else:
                kwargs["audios"] = audios

        processor = self._processor

        estimated_memory_bytes: Optional[int] = None
        fast_device: Optional[str] = None
        used_fast = False

        if (
            hasattr(processor, "image_processor")
            and isinstance(processor.image_processor, BaseImageProcessorFast)
            and not self.server_args.disable_fast_image_processor
        ):
            estimated_memory_bytes = self._estimate_fast_image_processor_memory_bytes(
                kwargs.get("images"), kwargs.get("videos"), kwargs.get("audios")
            )

            if _is_npu:
                kwargs["device"] = "npu"
                fast_device = "npu"
                used_fast = True
            else:
                if self._should_use_fast_image_processor(estimated_memory_bytes):
                    device_index = (
                        self._fast_image_processor_device_index
                        if self._fast_image_processor_device_index is not None
                        else torch.cuda.current_device()
                    )
                    fast_device = "cuda" if device_index == 0 else f"cuda:{device_index}"
                    kwargs["device"] = fast_device
                    used_fast = True
                else:
                    kwargs.pop("device", None)
                    fast_device = "cpu"
                    stats = self._fast_image_processor_stats
                    logger.debug(
                        "Fast image processor fallback to CPU (estimate %.2f MiB, allowed %.2f MiB, free %.2f MiB)",
                        (estimated_memory_bytes or 0) / (1024**2),
                        (stats.get("last_allowed_bytes") or 0) / (1024**2),
                        (stats.get("last_free_bytes") or 0) / (1024**2),
                    )
                    print_warning_once(
                        "Fast image processor fallback to CPU due to insufficient GPU memory. "
                        "Consider lowering image resolution, adjusting SGLANG_FAST_IMAGE_PROCESSOR_CAP_RATIO, "
                        "or disable the fast image processor."
                    )

        result = processor.__call__(
            text=[input_text],
            padding=True,
            return_tensors="pt",
            **kwargs,
        )

        if estimated_memory_bytes is not None:
            if fast_device is None:
                fast_device = "npu" if _is_npu else ("cuda" if used_fast else "cpu")
            metadata = {
                self._FAST_IMAGE_MEMORY_KEY: int(estimated_memory_bytes),
                self._FAST_IMAGE_DEVICE_KEY: fast_device,
                self._FAST_IMAGE_USED_FAST_KEY: used_fast,
            }
            self._fast_image_processor_stats.update(
                {"last_device": fast_device, "last_used_fast": used_fast}
            )
            self._attach_processor_output_metadata(result, metadata)

        if not self.server_args.keep_mm_feature_on_device:
            # move feature tensors to cpu
            for feature_name in self.FEATURE_NAMES:
                if feature_name in result and isinstance(
                    result[feature_name], torch.Tensor
                ):
                    result[feature_name] = result[feature_name].to("cpu")

        return result

    @abstractmethod
    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        pass

    def get_estimated_frames_list(self, image_data):
        """
        estimate the total frame count from all visual input
        """
        # Lazy import because decord is not available on some arm platforms.
        from decord import VideoReader, cpu

        # Before processing inputs
        if not image_data or len(image_data) == 0:
            return []
        estimated_frames_list = []
        for image in image_data:
            if isinstance(image, str) and image.startswith("video:"):
                path = image[len("video:") :]
                # Estimate frames for the video
                vr = VideoReader(path, ctx=cpu(0))
                num_frames = len(vr)
            else:
                # For images, each contributes one frame
                num_frames = 1
            estimated_frames_list.append(num_frames)

        return estimated_frames_list

    @staticmethod
    def _load_single_item(
        data,
        modality: Modality,
        frame_count_limit=None,
        audio_sample_rate: Optional[int] = None,
        discard_alpha_channel=True,
    ):
        """
        Load a single multimodal data.

        If data is precomputed, returns directly.

        Static method that can be pickled for multiprocessing"""
        if isinstance(data, dict):
            return data
        try:
            if modality == Modality.IMAGE:
                img, _ = load_image(data)
                if discard_alpha_channel and img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            elif modality == Modality.VIDEO:
                return load_video(data, frame_count_limit)
            elif modality == Modality.AUDIO:
                return load_audio(data, audio_sample_rate)

        except Exception as e:
            raise RuntimeError(f"Error while loading data {data}: {e}")

    def submit_data_loading_tasks(
        self,
        text_parts: List[str],
        multimodal_tokens: MultimodalSpecialTokens,
        data_iterators: dict[Modality, Iterator[Any]],
        discard_alpha_channel: bool = True,
        image_estimated_frames_iter: Optional[iter] = None,
        image_scaling_factor: float = 1.0,
        max_image_frames: int = 30,
        audio_sample_rate: Optional[int] = None,
    ) -> Tuple[List, List]:
        """
        load multimodal data parallelly using iterators.
        """
        futures = []
        task_info = []

        for text_part in text_parts:
            modality = multimodal_tokens.get_modality_of_token(text_part)
            if modality is not None:
                data_iterator = data_iterators.get(modality)
                if data_iterator is None:
                    raise ValueError(f"No data iterator found for token: {text_part}")

                try:
                    data = next(data_iterator)
                except StopIteration:
                    raise ValueError(
                        f"Mismatch: More '{text_part}' tokens found than corresponding data items provided."
                    )

                frame_count_limit = None
                if modality == Modality.IMAGE and image_estimated_frames_iter:
                    try:
                        estimated_frames = next(image_estimated_frames_iter)
                        # Use the pre-calculated scaling factor and max frames
                        frame_count_limit = max(
                            1, int(estimated_frames * image_scaling_factor)
                        )
                        # Ensure we don't exceed the absolute max (redundant if scaling_factor handles it)
                        # frame_count_limit = min(frame_count_limit, max_image_frames)
                    except StopIteration:
                        raise ValueError(
                            "Mismatch between image tokens and estimated frame counts."
                        )

                futures.append(
                    self.io_executor.submit(
                        BaseMultimodalProcessor._load_single_item,
                        data,
                        modality,
                        frame_count_limit,
                        audio_sample_rate,
                        discard_alpha_channel,
                    )
                )
                task_info.append((modality, data, frame_count_limit))

        for modality, iterator in data_iterators.items():
            try:
                next(iterator)
                logger.warning(
                    f"Warning: More {modality.name.lower()} data items provided than corresponding tokens found in the prompt."
                )
            except StopIteration:
                pass
            except Exception:
                pass

        return futures, task_info

    def load_mm_data(
        self,
        prompt: str,
        multimodal_tokens: MultimodalSpecialTokens,
        image_data: Optional[list] = None,
        video_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        return_text: Optional[bool] = True,
        discard_alpha_channel: bool = True,
        audio_sample_rate: Optional[int] = None,
    ) -> BaseMultiModalProcessorOutput:
        """
        Each frame of video/image will be replaced by a single image token

        Args:
            multimodal_tokens (list[str]): list of special token which denoting a single multimodal data
                e.g. image token or audio token
            discard_alpha_channel: if True, discards the alpha channel in the returned images

        """
        multimodal_tokens_pattern = multimodal_tokens.get_combined_regex()

        if isinstance(prompt, list) and return_text:
            assert len(prompt) and isinstance(prompt[0], int)
            prompt = self._processor.tokenizer.decode(prompt)
        else:
            prompt = prompt

        assert isinstance(prompt, str)
        # split text into list of normal text and special tokens
        text_parts = re.split(multimodal_tokens_pattern, prompt)

        # collect all data
        data_iterators = {}
        if multimodal_tokens.image_token and image_data:
            data_iterators[Modality.IMAGE] = iter(image_data)
        if multimodal_tokens.video_token and video_data:
            data_iterators[Modality.VIDEO] = iter(video_data)
        if multimodal_tokens.audio_token and audio_data:
            data_iterators[Modality.AUDIO] = iter(audio_data)

        # futures: the futures of loaded data
        # task_info: modality, raw_data, and other metadata of each data
        futures, task_info = self.submit_data_loading_tasks(
            text_parts=text_parts,
            multimodal_tokens=multimodal_tokens,
            data_iterators=data_iterators,
            discard_alpha_channel=discard_alpha_channel,
            audio_sample_rate=audio_sample_rate,
        )
        task_info_iter = iter(task_info)
        futures_iter = iter(futures)

        # Process results
        images, videos, audios = [], [], []
        new_text_parts = []
        for text_part in text_parts:
            try:
                if multimodal_tokens_pattern.match(text_part):
                    modality, raw_data, frame_limit = next(task_info_iter)
                    is_precomputed = isinstance(raw_data, dict)
                    result = next(futures_iter).result()

                    if modality == Modality.IMAGE:
                        # If data is already processed it will be a
                        # dictionary(precomputed). In this case we want to keep the
                        # expanded tokens in text_part. Otherwise, we will
                        # call the processor code, so keep only a single image
                        # token.
                        mm_tokens = (
                            text_part
                            if is_precomputed
                            else multimodal_tokens.image_token
                        )
                        frames = [result] if not isinstance(result, list) else result
                        if frames:
                            # only for minicpmv
                            images += frames
                            new_text_parts += mm_tokens * len(frames)
                    elif modality == Modality.VIDEO:
                        # load as video
                        mm_tokens = (
                            text_part
                            if is_precomputed
                            else multimodal_tokens.video_token
                        )
                        videos += [result]
                        new_text_parts += mm_tokens
                    elif modality == Modality.AUDIO:
                        # audio
                        mm_tokens = (
                            text_part
                            if is_precomputed
                            else multimodal_tokens.audio_token
                        )
                        audios += [result]
                        new_text_parts += mm_tokens
                else:
                    # normal text
                    new_text_parts += [text_part]

            except Exception as e:
                raise RuntimeError(
                    f"An exception occurred while loading multimodal data: {e}"
                )
        return BaseMultiModalProcessorOutput(
            images=images,
            audios=audios,
            videos=videos,
            input_text="".join(new_text_parts),
        )

    @staticmethod
    def get_mm_items_offset(
        input_ids: torch.Tensor, mm_token_id: int
    ) -> List[Tuple[int, int]]:
        """
        Get a set of range for mm_items from input_ids
        Example:
            input_ids = [1, 2, 3, 3, 3, 4, 3, 3]
            mm_token_id = 3
            return result = [(2,4),(6,7)]
        """
        mask = input_ids == mm_token_id
        start_positions = (mask & ~torch.roll(mask, 1)).nonzero(as_tuple=True)[0]
        end_positions = (mask & ~torch.roll(mask, -1)).nonzero(as_tuple=True)[0]

        return list(zip(start_positions.tolist(), end_positions.tolist()))

    @staticmethod
    def get_mm_items_offset_by_pair(
        input_ids: torch.Tensor, mm_start_id: int, mm_end_id: int
    ) -> List[Tuple[int, int]]:
        indices_start = (input_ids == mm_start_id).nonzero(as_tuple=True)[0] + 1
        indices_end = (input_ids == mm_end_id).nonzero(as_tuple=True)[0] - 1

        return list(zip(indices_start.tolist(), indices_end.tolist()))

    def collect_mm_items_from_processor_output(
        self, data_dict: dict
    ) -> List[MultimodalDataItem]:
        """Create mm_items directly from processor output."""
        items: dict[Modality, MultimodalDataItem] = {}
        for attr_name, value in data_dict.items():
            if attr_name == "input_ids":
                continue

            # Get modality for this attribute
            modality = self.ATTR_NAME_TO_MODALITY.get(attr_name)

            if attr_name == "precomputed_embeddings":
                modality_str = data_dict.get("modality")
                modality = Modality.IMAGE
                if modality_str:
                    try:
                        modality = Modality.from_str(modality_str)
                    except ValueError:
                        pass

            if modality:
                # Create item if needed
                if modality not in items:
                    items[modality] = MultimodalDataItem(modality=modality)

                if attr_name in self.FEATURE_NAMES:
                    attr_name = "feature"

                items[modality].set(attr_name, value)

        return list(items.values())

    def _process_and_collect_mm_items(
        self, input_text: str, images=None, audios=None, videos=None, **kwargs
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """
        Helper method to process multimodal data and create mm_items in one step.

        Returns:
            Tuple of (created mm_items, input_ids)
        """
        ret = self.process_mm_data(
            input_text=input_text, images=images, audios=audios, videos=videos, **kwargs
        )

        input_ids = ret["input_ids"].flatten()
        collected_items = self.collect_mm_items_from_processor_output(ret)

        return collected_items, input_ids, ret

    def process_and_combine_mm_data(
        self,
        base_output: BaseMultiModalProcessorOutput,
        mm_tokens: MultimodalSpecialTokens,
        **kwargs,
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """
        Process multimodal data and return the combined multimodal items and input_ids.
        Supports mixed modalities (images and audio in the same request).

        Returns:
            Tuple of (list of mm_items, input_ids)
        """
        # Collect all items and categorize them
        all_items = base_output.organize_results()
        # Handle text-only case
        if not all_items:
            input_ids = self._processor.tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()
            return [], input_ids, {}

        dict_items, raw_images, raw_audios, raw_videos = [], [], [], []
        for modality, item in all_items:
            if isinstance(item, dict):
                dict_items.append(item)
            elif modality == Modality.IMAGE:
                raw_images.append(item)
            elif modality == Modality.AUDIO:
                raw_audios.append(item)
            elif modality == Modality.VIDEO:
                raw_videos.append(item)
            else:
                raise ValueError(f"Unknown multimodal item type: {type(item)}")
        # Process items and get input_ids
        all_collected_items: list[MultimodalDataItem] = []
        input_ids = None

        # Handle raw items (need processing)
        if raw_images or raw_audios or raw_videos:
            collected_items, input_ids, ret = self._process_and_collect_mm_items(
                input_text=base_output.input_text,
                images=raw_images,
                audios=raw_audios,
                videos=raw_videos,
                **kwargs,
            )
            all_collected_items = collected_items
        else:
            ret = None

        # Handle dict items (already processed)
        for dict_item in dict_items:
            all_collected_items.extend(
                self.collect_mm_items_from_processor_output(dict_item)
            )

        # Fallback tokenization if no raw items were processed
        if input_ids is None:
            input_ids = self._processor.tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()

        # Add offsets to all items
        for mm_item in all_collected_items:
            mm_token_id = mm_tokens.get_token_id_by_modality(mm_item.modality)
            if mm_token_id is None:
                raise ValueError(f"No token id found for modality: {mm_item.modality}")
            mm_item.offsets = self.get_mm_items_offset(
                input_ids=input_ids,
                mm_token_id=mm_token_id,
            )

        return all_collected_items, input_ids, ret
