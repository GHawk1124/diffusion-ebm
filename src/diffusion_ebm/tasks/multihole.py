"""Masked-input prompts for the M3 multi-hole-agreement experiment.

Each instance has 2-3 [MASK] holes that must take the same vocab id. The joint
sampler at src/diffusion_ebm/sampler/thrml_joint.py consumes
MultiHoleInstance.equality_groups to wire equality factors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from diffusion_ebm.backbones.mdlm import MASK_TOKEN_ID


@dataclass
class MultiHoleInstance:
    text: str                       # display string, with literal [M] markers in place of holes
    masked_input_ids: torch.Tensor  # shape (L,), dtype torch.long, with MASK_TOKEN_ID at every hole position
    mask_positions: list[int]       # indices into masked_input_ids; sorted ascending
    equality_groups: list[list[int]]  # partition of mask_positions; every position in exactly one group


def _build(tokenizer, segments: list[str]) -> tuple[torch.Tensor, list[int]]:
    """Tokenizes the literal segments and inserts ONE mask token between each pair.
    Returns (masked_input_ids, mask_positions).
    Note: tokenizer.encode of each segment WITHOUT bos/eos. Concatenate, tracking the
    index of each inserted MASK_TOKEN_ID. The joining is done by Python list ops, not
    tokenizer.encode of the full sentence.
    """
    input_ids: list[int] = []
    mask_positions: list[int] = []

    for index, segment in enumerate(segments):
        input_ids.extend(tokenizer.encode(segment, add_special_tokens=False))
        if index < len(segments) - 1:
            mask_positions.append(len(input_ids))
            input_ids.append(MASK_TOKEN_ID)

    return torch.tensor(input_ids, dtype=torch.long), mask_positions


def color_template(tokenizer) -> MultiHoleInstance:
    masked_input_ids, mask_positions = _build(
        tokenizer,
        ["Alice's favorite color is", ". Bob's favorite color is also", "."],
    )
    return MultiHoleInstance(
        text="Alice's favorite color is [M]. Bob's favorite color is also [M].",
        masked_input_ids=masked_input_ids,
        mask_positions=mask_positions,
        equality_groups=[mask_positions],
    )


def variable_template(tokenizer) -> MultiHoleInstance:
    masked_input_ids, mask_positions = _build(
        tokenizer,
        [
            "The variable",
            " was assigned the value 7. Later,",
            " was used in a loop.",
        ],
    )
    return MultiHoleInstance(
        text="The variable [M] was assigned the value 7. Later, [M] was used in a loop.",
        masked_input_ids=masked_input_ids,
        mask_positions=mask_positions,
        equality_groups=[mask_positions],
    )


def repeat3_template(tokenizer) -> MultiHoleInstance:
    masked_input_ids, mask_positions = _build(
        tokenizer,
        ["My name is", ". You can call me", ". I said", " three times."],
    )
    return MultiHoleInstance(
        text="My name is [M]. You can call me [M]. I said [M] three times.",
        masked_input_ids=masked_input_ids,
        mask_positions=mask_positions,
        equality_groups=[mask_positions],
    )


def all_templates(tokenizer) -> list[MultiHoleInstance]:
    return [
        color_template(tokenizer),
        variable_template(tokenizer),
        repeat3_template(tokenizer),
    ]
