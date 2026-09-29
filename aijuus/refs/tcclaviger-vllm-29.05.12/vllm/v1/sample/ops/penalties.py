# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.v1.sample.ops.dry import DryParams, apply_dry  # FORK-LOCAL

from vllm.model_executor.layers.utils import apply_penalties
from vllm.utils.torch_utils import PIN_MEMORY, make_tensor_with_pad


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
    dry_params: dict[int, "DryParams"] | None = None,
    apply_classic: bool = True,
) -> torch.Tensor:
    """Applies classic penalties then DRY; ``dry_params`` is keyed by logits row."""
    _, vocab_size = logits.shape

    # FORK-LOCAL (patches/dry_sampler): classic penalties are skippable so a batch
    # with only DRY requests does not pay for them.
    if apply_classic:
        output_tokens_t = _convert_to_tensors(
            output_token_ids, vocab_size, logits.device
        )

        # In the async scheduling case, rows that won't have penalties applied may
        # contain -1 placeholder token ids. We must replace these with valid token ids
        # so that the scatter done in apply_penalties is valid.
        # NOTE(nick): The penalties implementation is currently quite inefficient and
        # will be reworked anyhow.
        output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)

        apply_penalties(
            logits,
            prompt_token_ids,
            output_tokens_t,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
        )

    if dry_params:
        # DRY must run after repetition penalty, which branches on logit sign.
        apply_dry(
            logits,
            prompt_token_ids,
            output_token_ids,
            dry_params,
            vocab_size,
        )

    return logits


def _convert_to_tensors(
    output_token_ids: list[list[int]], vocab_size: int, device: torch.device
) -> torch.Tensor:
    """Convert the different list data structures to tensors."""
    output_tokens_tensor = make_tensor_with_pad(
        output_token_ids,
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=PIN_MEMORY,
    )
    return output_tokens_tensor.to(device, non_blocking=True)
