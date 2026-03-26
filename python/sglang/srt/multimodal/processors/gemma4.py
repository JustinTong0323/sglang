# Copyright 2025 SGLang Team
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

from typing import Dict, List, Optional, Union

import numpy as np

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens


class Gemma4SGLangProcessor(SGLangBaseProcessor):
    """Multimodal processor for Gemma4 supporting image and audio inputs."""

    models = [Gemma4ForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IM_START_TOKEN_ID = hf_config.boi_token_id
        self.IM_END_TOKEN_ID = hf_config.eoi_token_id

        self.AUDIO_START_TOKEN_ID = hf_config.boa_token_id
        self.AUDIO_END_TOKEN_ID = hf_config.eoa_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token_id=hf_config.image_token_id,
            audio_token_id=hf_config.audio_token_id,
        ).build(_processor)

        # Register new image-processor outputs so they are stored on
        # MultimodalDataItem via collect_mm_items_from_processor_output.
        self.ATTR_NAME_TO_MODALITY["pixel_position_ids"] = Modality.IMAGE
        self.ATTR_NAME_TO_MODALITY["vision_output_length"] = Modality.IMAGE

    def _get_audio_pad_multiple(self) -> int:
        """Derive the waveform padding alignment from processor config.

        The HF processor's ceil(duration_ms / audio_ms_per_token) formula can
        overshoot by 1 token relative to what the SSCP convolutions produce.
        Padding waveforms to a multiple of (hop_length * first_conv_stride)
        aligns the two calculations.
        See: gemma-4-eap-extras/examples/gemma-4-audio-examples.ipynb
        """
        fe = getattr(self._processor, "feature_extractor", None)
        hop = getattr(fe, "hop_length", 160)
        ac = getattr(self.hf_config, "audio_config", None)
        first_stride = ac.sscp_conv_stride_size[0][0] if ac is not None else 2
        return hop * first_stride

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ):
        if audios:
            pad_multiple = self._get_audio_pad_multiple()
            padded = []
            for a in audios:
                a = np.asarray(a)
                remainder = len(a) % pad_multiple
                if remainder != 0:
                    a = np.pad(a, (0, pad_multiple - remainder), mode="constant")
                padded.append(a)
            audios = padded
        return super().process_mm_data(
            input_text, images=images, videos=videos, audios=audios, **kwargs
        )

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,
        audio_data: Optional[List[Union[str, bytes, Dict]]] = None,
        input_text: str = "",
        request_obj=None,
        *args,
        **kwargs,
    ):
        """Process multimodal data including images and audio."""
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
            "audio_token_id": self.mm_tokens.audio_token_id,
        }
