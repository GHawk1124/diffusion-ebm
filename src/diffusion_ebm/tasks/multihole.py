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


def distance_template(tokenizer) -> MultiHoleInstance:
    """Long-context equality (M5b). Same constraint as `color_template`, but
    with an 80-ish-token distractor paragraph between the two holes — tests
    whether attention preserves the equality signal across distance.
    """
    distractor = (
        " The weather that morning was unusually mild. A light breeze stirred"
        " the curtains, and somewhere in the next room a clock ticked steadily."
        " Neither of them spoke for a long while; the silence felt comfortable"
        " rather than awkward, and the coffee on the table had gone cold."
    )
    masked_input_ids, mask_positions = _build(
        tokenizer,
        [
            "Alice's favorite color is",
            "." + distractor + " Bob's favorite color is also",
            ".",
        ],
    )
    return MultiHoleInstance(
        text="Alice's favorite color is [M]. <80-token distractor>. Bob's favorite color is also [M].",
        masked_input_ids=masked_input_ids,
        mask_positions=mask_positions,
        equality_groups=[mask_positions],
    )


def many_holes_template(tokenizer) -> MultiHoleInstance:
    """Four equality holes in tight quarters (M5b). Stress-tests the
    equality factor when the joint state space is k^4. mask-predict can
    cascade-commit since adjacent context is informative once one hole is
    fixed; THRML must explore the full joint.
    """
    masked_input_ids, mask_positions = _build(
        tokenizer,
        [
            "Whenever",
            " walked into the room,",
            " smiled politely. Then",
            " sat down beside",
            " quietly.",
        ],
    )
    return MultiHoleInstance(
        text="Whenever [M] walked into the room, [M] smiled politely. Then [M] sat down beside [M] quietly.",
        masked_input_ids=masked_input_ids,
        mask_positions=mask_positions,
        equality_groups=[mask_positions],
    )


def multi_group_template(tokenizer) -> MultiHoleInstance:
    """Two distinct equality groups in one prompt (M5b). Group A spans the
    name slots; group B spans the city slots. mask-predict must satisfy both
    constraints jointly without seeing them as separate; the joint sampler
    sees both as factors.
    """
    masked_input_ids, mask_positions = _build(
        tokenizer,
        [
            "The author",
            " was born in",
            ", and years later, when",
            " returned to",
            ", the city had changed.",
        ],
    )
    # Holes (in order): name1, city1, name2, city2.
    return MultiHoleInstance(
        text="The author [M_a] was born in [M_b], and years later, when [M_a] returned to [M_b], the city had changed.",
        masked_input_ids=masked_input_ids,
        mask_positions=mask_positions,
        equality_groups=[
            [mask_positions[0], mask_positions[2]],  # name group
            [mask_positions[1], mask_positions[3]],  # city group
        ],
    )


def distractor_template(tokenizer) -> MultiHoleInstance:
    """Leading prefix biases per-hole conditional toward a specific token
    (M5b). The mask_predict argmax is likely to pick the prefix-matching
    token at both held holes, trivially "agreeing" but on a token chosen
    by the misleading context — not the joint optimum given the *equality*
    constraint between the *latter two* holes.

    Equality group: only the two later holes (Bob's, Carol's). Alice's
    color is fixed surface text in a separate (singleton) group.
    """
    masked_input_ids, mask_positions = _build(
        tokenizer,
        [
            "Alice's favorite color is red. Bob's favorite color is",
            ". Carol's favorite color is also",
            ".",
        ],
    )
    return MultiHoleInstance(
        text="Alice's favorite color is red. Bob's favorite color is [M]. Carol's favorite color is also [M].",
        masked_input_ids=masked_input_ids,
        mask_positions=mask_positions,
        equality_groups=[mask_positions],
    )


def polyseme_template(tokenizer) -> MultiHoleInstance:
    """Polyseme/homonym intersection (M5b, brainstorm-derived). The local
    LM mode at hole 1 ("trunk") differs from the local mode at hole 2
    ("shoe"), but both contexts admit a single shared token ("boot"). A
    joint sampler with an equality factor finds "boot"; mask-predict at
    T=0 commits each hole to its local argmax and produces an
    inconsistent fill.
    """
    masked_input_ids, mask_positions = _build(
        tokenizer,
        [
            "She packed her bag into the car's",
            ", then put a sturdy leather",
            " on her foot.",
        ],
    )
    return MultiHoleInstance(
        text="She packed her bag into the car's [M], then put a sturdy leather [M] on her foot.",
        masked_input_ids=masked_input_ids,
        mask_positions=mask_positions,
        equality_groups=[mask_positions],
    )


_CORE = (color_template, variable_template, repeat3_template)
_M5B = (
    distance_template,
    many_holes_template,
    multi_group_template,
    distractor_template,
    polyseme_template,
)


def all_templates(
    tokenizer, family: str | None = None
) -> list[MultiHoleInstance]:
    """Build templates for the requested family.

    ``family`` ∈ {None, 'core', 'm5b', 'all'}. ``None`` and ``'core'``
    return the original three M3 templates so existing callers keep
    working. ``'m5b'`` returns the five new boundary-probing templates;
    ``'all'`` returns both.
    """
    fam = family or "core"
    if fam == "core":
        builders = _CORE
    elif fam == "m5b":
        builders = _M5B
    elif fam == "all":
        builders = _CORE + _M5B
    else:
        raise ValueError(f"unknown family {family!r}")
    return [b(tokenizer) for b in builders]
