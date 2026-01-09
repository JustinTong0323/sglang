# adapted from
# https://github.com/vllm-project/vllm/blob/82a1b1a82b1fbb454c82a9ef95730b929c9b270c/vllm/model_executor/layers/pooler.py

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.layers.activation import get_cross_encoder_activation_function
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class PoolingType(IntEnum):
    LAST = 0
    CLS = 1


@dataclass
class EmbeddingPoolerOutput:
    # Pooler can return list[tensor] instead of tensor if the dimension of each tensor in the batch is different
    # due to different per-request matryoshka dim truncation
    embeddings: torch.Tensor | list[torch.Tensor]


class Pooler(nn.Module):
    """A layer that pools specific information from hidden states.
    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.
    Attributes:
        pooling_type: The type of pooling to use (LAST, AVERAGE, MAX).
        normalize: Whether to normalize the pooled data.
    """

    def __init__(self, pooling_type: PoolingType, normalize: bool):
        super().__init__()
        self.pooling_type = pooling_type
        self.normalize = normalize

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> EmbeddingPoolerOutput:

        if self.pooling_type == PoolingType.LAST:
            last_token_indices = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            pooled_data = hidden_states[last_token_indices]
        elif self.pooling_type == PoolingType.CLS:
            prompt_lens = forward_batch.extend_seq_lens
            first_token_flat_indices = torch.zeros_like(prompt_lens)
            first_token_flat_indices[1:] += torch.cumsum(prompt_lens, dim=0)[:-1]
            pooled_data = hidden_states[first_token_flat_indices]
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")

        if forward_batch.dimensions is not None:
            all_same_dimensions = len(set(forward_batch.dimensions)) == 1
            if all_same_dimensions:
                pooled_data = pooled_data[..., : forward_batch.dimensions[0]]
            else:
                pooled_data = [
                    tensor[..., :dim]
                    for tensor, dim in zip(pooled_data, forward_batch.dimensions)
                ]

        if self.normalize:
            if isinstance(pooled_data, list):
                pooled_data = [
                    nn.functional.normalize(tensor, p=2, dim=-1)
                    for tensor in pooled_data
                ]
            else:
                pooled_data = nn.functional.normalize(pooled_data, p=2, dim=-1)

        return EmbeddingPoolerOutput(embeddings=pooled_data)


class CrossEncodingPooler(nn.Module):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `EmbeddingPoolerOutput`.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        classifier: nn.Module,
        pooler: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.classifier = classifier
        self.pooler = pooler
        self.default_activation_function = get_cross_encoder_activation_function(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> EmbeddingPoolerOutput:
        """Pools sentence pair scores from the hidden_states."""

        prompt_lens = forward_batch.extend_seq_lens

        offset = 0
        pooled_data_lst = []
        for prompt_len in prompt_lens:
            pooled_data_i = hidden_states[offset : offset + prompt_len]

            if self.pooler is not None:
                final_shape_tensor = self.pooler(pooled_data_i, forward_batch)
            else:
                final_shape_tensor = self.classifier(pooled_data_i)

            pooled_data_lst.append(final_shape_tensor)
            offset += prompt_len

        pooled_output = torch.stack(pooled_data_lst)

        if self.pooler is not None:
            # apply classifier once on the full batch if possible
            pooled_output = self.classifier(pooled_output)

        scores = self.default_activation_function(pooled_output).squeeze(-1)

        return EmbeddingPoolerOutput(embeddings=scores)


class DecoderOnlyRerankerScorer(nn.Module):
    """Scorer for decoder-only reranker models (e.g., Qwen3-VL-Reranker).

    This implements the official scoring method which uses the difference of
    lm_head weights for yes/no tokens applied to the last hidden state:

        score = sigmoid(last_hidden_state @ (weight_yes - weight_no))

    This approach is more robust than extracting logprobs from model outputs,
    especially for larger model sizes.
    """

    def __init__(
        self,
        lm_head: nn.Module,
        yes_token_id: int,
        no_token_id: int,
    ):
        """Initialize the reranker scorer.

        Args:
            lm_head: The language model head (nn.Linear or ParallelLMHead).
            yes_token_id: Token ID for "yes".
            no_token_id: Token ID for "no".
        """
        super().__init__()
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id

        # Extract and cache the weight difference for scoring
        # weight_yes - weight_no gives us a vector that, when dotted with
        # hidden states, produces a score where positive = yes, negative = no
        self._init_score_weights(lm_head)

    def _init_score_weights(self, lm_head: nn.Module):
        """Initialize the scoring weights from lm_head."""
        with torch.no_grad():
            # Handle both regular Linear and ParallelLMHead
            if hasattr(lm_head, "weight"):
                lm_head_weights = lm_head.weight.data
            else:
                raise ValueError(
                    f"Cannot extract weights from lm_head of type {type(lm_head)}"
                )

            # Get weights for yes and no tokens
            weight_yes = lm_head_weights[self.yes_token_id]
            weight_no = lm_head_weights[self.no_token_id]

            # Store the weight difference
            # score = hidden @ (weight_yes - weight_no)
            self.register_buffer(
                "score_weights", weight_yes - weight_no, persistent=False
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> EmbeddingPoolerOutput:
        """Compute reranker scores from hidden states.

        Args:
            hidden_states: Hidden states from the model. Shape: (total_tokens, hidden_size)
            forward_batch: Batch information containing sequence lengths.

        Returns:
            EmbeddingPoolerOutput with scores for each sequence in the embeddings field.
            Scores are returned as shape (batch_size, 1) for compatibility with embedding pipeline.
        """
        # Extract last token hidden state for each sequence
        last_token_indices = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
        last_hidden_states = hidden_states[last_token_indices]

        # Compute scores: hidden @ score_weights
        # score_weights shape: (hidden_size,)
        # last_hidden_states shape: (batch_size, hidden_size)
        logits = torch.matmul(last_hidden_states, self.score_weights)

        # Apply sigmoid to get probabilities in [0, 1]
        scores = torch.sigmoid(logits)

        # Return as (batch_size, 1) for embedding pipeline compatibility
        return EmbeddingPoolerOutput(embeddings=scores.unsqueeze(-1))

    def forward_for_generation(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """Compute reranker scores for use in generation mode.

        This returns a LogitsProcessorOutput-compatible result with scores
        stored in customized_info for processing by the scheduler.

        Args:
            hidden_states: Hidden states from the model. Shape: (total_tokens, hidden_size)
            forward_batch: Batch information containing sequence lengths.

        Returns:
            LogitsProcessorOutput with scores stored in customized_info["reranker_scores"].
        """
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        # Extract last token hidden state for each sequence
        last_token_indices = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
        last_hidden_states = hidden_states[last_token_indices]

        # Compute scores: hidden @ score_weights
        logits = torch.matmul(last_hidden_states, self.score_weights)

        # Apply sigmoid to get probabilities in [0, 1]
        scores = torch.sigmoid(logits)

        # Convert to list for JSON serialization
        scores_list = scores.cpu().tolist()

        # Return LogitsProcessorOutput with scores in customized_info
        return LogitsProcessorOutput(
            next_token_logits=None,  # No logits needed for reranker
            customized_info={"reranker_scores": scores_list},
        )
