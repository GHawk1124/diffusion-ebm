"""WinoGrande adapter for forced-choice rerank evaluation.

Each WinoGrande item has one blank (`_`) and two candidate fills.
This module builds `WinoGrandeItem` instances that expose:

  - masked_input_ids: sentence with blank replaced by MASK token(s)
  - option{1,2}_token_ids: tokenized fills
  - option_positions: token positions of the blank

Strategy: tokenize the sentence with each option filled in, find the
differing token span, and replace it with MASK tokens.  Items where the
two options tokenize to different lengths are skipped by default — they
require a different scoring formulation (pseudo-likelihood over variable-
length spans) and are rare enough (<5%) not to bias the pilot.

Used by experiments/m5c_eval_winogrande.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

BLANK = "_"


@dataclass
class WinoGrandeItem:
    item_id: str
    sentence: str
    option1: str
    option2: str
    gold: int                          # 1 or 2 (matches dataset "answer")
    masked_input_ids: torch.Tensor     # [L] with MASK at blank positions
    option1_token_ids: list[int]       # tokenized option1
    option2_token_ids: list[int]       # tokenized option2
    option_positions: list[int]        # token positions of the blank


def _diff_span(ids_a: list[int], ids_b: list[int]) -> tuple[int, int, int]:
    """Find contiguous span where two token sequences differ.

    Returns (prefix_len, a_end, b_end) such that:
      ids_a[prefix_len:a_end]  is option_a's tokens
      ids_b[prefix_len:b_end]  is option_b's tokens
      ids_a[a_end:] == ids_b[b_end:]   (shared suffix)
    """
    n = min(len(ids_a), len(ids_b))
    prefix_len = 0
    while prefix_len < n and ids_a[prefix_len] == ids_b[prefix_len]:
        prefix_len += 1

    suffix_len = 0
    max_suf = min(len(ids_a) - prefix_len, len(ids_b) - prefix_len)
    while suffix_len < max_suf and ids_a[-(suffix_len + 1)] == ids_b[-(suffix_len + 1)]:
        suffix_len += 1

    a_end = len(ids_a) - suffix_len if suffix_len else len(ids_a)
    b_end = len(ids_b) - suffix_len if suffix_len else len(ids_b)
    return prefix_len, a_end, b_end


def build_items(
    dataset,
    tokenizer,
    mask_token_id: int,
    max_items: Optional[int] = None,
    skip_length_mismatch: bool = True,
) -> tuple[list[WinoGrandeItem], dict]:
    """Build WinoGrandeItem list from a HF dataset split.

    Returns (items, stats) where stats has counts of accepted/skipped items.
    """
    items: list[WinoGrandeItem] = []
    stats: dict[str, int] = {
        "total": 0,
        "ok": 0,
        "length_mismatch": 0,
        "no_blank": 0,
        "no_diff": 0,
    }

    for row in dataset:
        stats["total"] += 1
        if max_items is not None and stats["ok"] >= max_items:
            break

        sentence = row["sentence"]
        if BLANK not in sentence:
            stats["no_blank"] += 1
            continue

        opt1 = row["option1"]
        opt2 = row["option2"]
        gold = int(row["answer"])
        item_id = str(row.get("qID", stats["total"]))

        ids1 = tokenizer.encode(sentence.replace(BLANK, opt1))
        ids2 = tokenizer.encode(sentence.replace(BLANK, opt2))

        prefix_len, a_end, b_end = _diff_span(ids1, ids2)
        opt1_toks = ids1[prefix_len:a_end]
        opt2_toks = ids2[prefix_len:b_end]

        if not opt1_toks and not opt2_toks:
            stats["no_diff"] += 1
            continue

        if skip_length_mismatch and len(opt1_toks) != len(opt2_toks):
            stats["length_mismatch"] += 1
            continue

        n_mask = len(opt1_toks)
        prefix_ids = ids1[:prefix_len]
        suffix_ids = ids1[a_end:]
        masked_ids = prefix_ids + [mask_token_id] * n_mask + suffix_ids
        option_positions = list(range(prefix_len, prefix_len + n_mask))

        items.append(WinoGrandeItem(
            item_id=item_id,
            sentence=sentence,
            option1=opt1,
            option2=opt2,
            gold=gold,
            masked_input_ids=torch.tensor(masked_ids, dtype=torch.long),
            option1_token_ids=opt1_toks,
            option2_token_ids=opt2_toks,
            option_positions=option_positions,
        ))
        stats["ok"] += 1

    return items, stats
