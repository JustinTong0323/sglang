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
from typing import Iterable, List, Set, Tuple

import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel

from sglang.srt.configs import Gemma3p5Config
from sglang.srt.configs.gemma3_5_config import Gemma3p5AudioConfig, Gemma3p5VisionConfig
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import maybe_remap_kv_scale_name
from sglang.srt.models.gemma3_5_causal import Gemma3p5ForCausalLM, Gemma3p5RMSNorm
from sglang.srt.utils import flatten_nested_list


class Gemma3p5MultiModalProjector(nn.Module):
    def __init__(self, config: Gemma3p5Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(
                config.vision_config.hidden_size, config.text_config.hidden_size
            )
        )

        self.mm_soft_emb_norm = Gemma3p5RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.tokens_per_side = int(config.vision_soft_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.kernel_size, stride=self.kernel_size
        )

    def forward(self, vision_outputs: torch.Tensor):
        batch_size, _, seq_length = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = torch.matmul(
            normed_vision_outputs, self.mm_input_projection_weight
        )
        return projected_vision_outputs.type_as(vision_outputs)


class Gemma3p5VisionEmbedder(nn.Module):
    def __init__(self, config: Gemma3p5Config, *args, vocab_offset: int = 0, **kwargs):
        super().__init__(*args, **kwargs)

        if (vision_config := config.vision_config) is None:
            raise ValueError(
                "`Gemma3p5Config` passed as `config` cannot have `vision_config=None`"
            )

        self.vision_config: Gemma3p5VisionConfig = vision_config
        self.text_config = config.text_config
        self.vocab_offset = vocab_offset

        self.embedding = nn.Embedding(
            self.vision_config.vocab_size, self.vision_config.hidden_size
        )

        self.embedding_norm = Gemma3p5RMSNorm(
            self.vision_config.hidden_size,
            eps=self.text_config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )

        self.embedding_projection = nn.Linear(
            self.vision_config.hidden_size, self.text_config.hidden_size, bias=False
        )

        self.embedding_post_projection_norm = Gemma3p5RMSNorm(
            dim=self.text_config.hidden_size,
            eps=self.text_config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=False,
        )

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        hard_emb = self.embedding(input_ids - self.vocab_offset)
        hard_emb_norm = self.embedding_norm(hard_emb)
        hard_emb_norm_proj = self.embedding_projection(hard_emb_norm)
        return self.embedding_post_projection_norm(hard_emb_norm_proj)


class Gemma3p5AudioEmbedder(nn.Module):
    def __init__(self, config: Gemma3p5Config, *args, vocab_offset: int = 0, **kwargs):
        super().__init__(*args, **kwargs)

        if (audio_config := config.audio_config) is None:
            raise ValueError(
                "`Gemma3p5Config` passed as `config` cannot have `audio_config=None`"
            )

        self.audio_config: Gemma3p5AudioConfig = audio_config
        self.text_config = config.text_config
        self.vocab_offset = vocab_offset

        self.embedding = nn.Embedding(
            self.audio_config.vocab_size, self.audio_config.hidden_size
        )

        self.embedding_norm = Gemma3p5RMSNorm(
            dim=self.audio_config.hidden_size,
            eps=self.audio_config.embedding_norm_eps,
            scale_shift=0.0,
            with_scale=True,
        )

        self.embedding_projection = nn.Linear(
            self.audio_config.hidden_size, self.text_config.hidden_size, bias=False
        )

        self.embedding_post_projection_norm = Gemma3p5RMSNorm(
            dim=self.text_config.hidden_size,
            eps=self.audio_config.embedding_norm_eps,
            scale_shift=0.0,
            with_scale=False,
        )

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        hard_emb = self.embedding(input_ids - self.vocab_offset)
        hard_emb_norm = self.embedding_norm(hard_emb)
        hard_emb_norm_proj = self.embedding_projection(hard_emb_norm)
        return self.embedding_post_projection_norm(hard_emb_norm_proj)


class Gemma3p5AudioEncoder(nn.Module):
    def __init__(self, config: Gemma3p5AudioConfig, *args, **kwargs):
        super().__init__(config)
        self.config = config
        pass


class Gemma3p5ForConditionalGeneration(PreTrainedModel):
    model_type = "gemma3p5"

    def __init__(self, config: Gemma3p5Config):
        super().__init__(config)
        # TODO
        self.vision_tower = AutoModel.from_pretrained(config.vision_config)
        self.audio_tower = Gemma3p5AudioEncoder(config.audio_config)
        self.multi_modal_projector = Gemma3p5MultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size

        language_model = Gemma3p5ForCausalLM(config=config.text_config)

        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"language_model.{k}" for k in language_model._tied_weights_keys
            ]
        self.language_model = language_model

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )
        # self.audio_tower = AutoModel.from_config(config=config.audio_config)

        vision_vocab_offset = audio_vocab_offset = config.text_config.vocab_size

        if config.vision_config is not None:
            audio_vocab_offset += config.vision_config.vocab_size
            self.embed_vision = Gemma3p5VisionEmbedder(
                config, vocab_offset=vision_vocab_offset
            )

        if config.audio_config is not None:
            self.embed_audio = Gemma3p5AudioEmbedder(
                config, vocab_offset=audio_vocab_offset
            )
        self.post_init()

    def pad_input_ids(
        self, input_ids: List[int], image_inputs: MultimodalInputs
    ) -> List[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens(
            [image_inputs.im_token_id, image_inputs.audio_token_id]
        )
        ids = pattern.pad_input_tokens(input_ids, image_inputs)
        return ids

    def get_audio_feature(self, items: List[MultimodalDataItem]):
        # TODO
        audio_feature = torch.stack(
            flatten_nested_list([item.audio_features for item in items]), dim=0
        )
        audio_feature = audio_feature.to(device=self.audio_tower.device)
        audio_feature = audio_feature.to(dtype=self.language_model.dtype())

        audio_outputs = self.audio_tower(audio_feature).last_hidden_state
        audio_features = self.multi_modal_projector(audio_outputs)
        return audio_features

    def get_image_feature(self, items: List[MultimodalDataItem]):
        """
        Projects the last hidden state from the vision model into language model space.

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        pixel_values = torch.stack(
            flatten_nested_list([item.pixel_values for item in items]), dim=0
        )

        pixel_values = pixel_values.to(device=self.vision_tower.device)
        pixel_values = pixel_values.to(dtype=self.language_model.dtype())

        vision_outputs = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        image_features = self.multi_modal_projector(vision_outputs)
        return image_features

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs: object,
    ) -> torch.Tensor:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/Gemma3-test-224px-hf")
        >>> processor = AutoProcessor.from_pretrained("google/Gemma3-test-224px-hf")

        >>> prompt = "answer en Where is the cow standing?"
        >>> url = "https://huggingface.co/gv-hf/Gemma3-test-224px-hf/resolve/main/cow_beach_1.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "answer en Where is the cow standing?\nbeach"
        ```"""

        # Important: position_ids in Gemma3 are 1-indexed
        # This really does cost me sometime
        # positions += 1

        # Replace image id with PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_index >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        hs = general_mm_embed_routine(
            input_ids=llm_input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            image_data_embedding_func=self.get_image_feature,
            positions=positions,
        )

        return hs

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights for the model."""
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            if "language_model" in name:
                # Gemma3ForCausalLM.load_weights(self, [(name.replace("language_model.", ""), loaded_weight)])
                causal_loaded_params = Gemma3p5ForCausalLM.load_weights(
                    self, [(name, loaded_weight)]
                )
                loaded_params.update(causal_loaded_params)
                continue
            else:
                # Skip lm_head.weight as it's tied with embed_tokens
                if "lm_head.weight" in name:
                    continue

                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue

                # Remapping the name of FP8 kv-scale
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            # pass
            raise RuntimeError(
                f"Some weights are not initialized from checkpoints: {unloaded_params}"
            )
        return loaded_params


EntryClass = Gemma3p5ForConditionalGeneration
AutoModel.register(Gemma3p5Config, Gemma3p5ForConditionalGeneration, exist_ok=True)
