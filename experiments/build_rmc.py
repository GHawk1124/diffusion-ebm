"""Build frozen RMC (Repeated Mention Cloze) benchmark files.

Deterministic extractor — pure GPT-2 tokenizer + heuristic rules, no ML.
Run once to produce the .jsonl artefacts; commit them to freeze the benchmark
so reviewers can rerun bit-exact.

Corpora
-------
owt_heldout   — last 1000 documents of Skylion007/openwebtext.
                The m5c_train pipeline streams from the start of OWT, so the
                last documents in the fixed HuggingFace ordering are unlikely
                to have been used during scorer training.
wikitext103   — validation split of wikitext/wikitext-103-raw-v1
                (out-of-domain; training used OWT only).

Algorithm (frozen)
------------------
1. Tokenize each document with the GPT-2 tokenizer (same vocab as MDLM).
2. Slide a 64-token window over each document with stride 32.
3. Within each window, find candidate entity tokens:
   - Decoded string has first non-space character uppercase.
   - Decoded string (lowercased) is not in STOPWORDS.
   - Appears ≥ 2 and ≤ 4 times in the window.
   - (Stability filter) re-encoding the decoded string gives back the same token id.
4. single_chain items: one entity per item (all its occurrences masked).
5. multi_chain items: 2–4 entities per item (all occurrences of each masked).
   Total holes capped at 10; if >10 holes, take entities greedily by fewest
   occurrences until the cap is reached, requiring ≥ 2 entities.
6. Each item records item_id, corpus, track, window_text, masked_input_ids,
   mask_positions, gold_token_ids, chain_labels.

Usage
-----
    uv run python experiments/build_rmc.py --out-dir data/rmc
    uv run python experiments/build_rmc.py --out-dir data/rmc --print-stats
    uv run python experiments/build_rmc.py --out-dir data/rmc \\
        --max-docs-owt 100 --max-docs-wt103 100   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

# Published stopword list (~50 strings; lowercase surface forms after lstrip).
# Months, days-of-week, common pronouns, common discourse markers, and
# function words that can appear capitalized at sentence starts.
STOPWORDS: frozenset[str] = frozenset({
    # Pronouns
    "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose", "which", "what",
    # Articles / determiners
    "the", "a", "an", "this", "that", "these", "those", "such",
    "all", "both", "each", "every", "any", "some", "few",
    "more", "most", "other", "another",
    # Discourse markers
    "however", "therefore", "furthermore", "moreover", "nevertheless",
    "meanwhile", "indeed", "also", "then", "now", "here", "there",
    "thus", "hence", "while", "when", "where", "how", "why",
    # Function words that appear capitalized at sentence starts
    "in", "on", "at", "by", "for", "with", "from", "to",
    "of", "about", "as", "into", "through", "during", "before", "after",
    "and", "but", "or", "nor", "yet", "not", "no", "so",
    "than", "if", "is", "are", "was", "were", "be", "been", "have",
    # Days
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    # Months
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
})

WINDOW_SIZE = 64
STRIDE = 32
MAX_HOLES = 10
MIN_OCCURRENCES = 2
MAX_OCCURRENCES = 4


def _make_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.model_max_length = 1_000_000  # suppress truncation warnings
    return tok


def _is_entity_token(tok_id: int, tokenizer) -> bool:
    """Return True if tok_id is a capitalized, non-stopword, stable token."""
    decoded = tokenizer.decode([tok_id])
    stripped = decoded.lstrip()
    if not stripped or not stripped[0].isupper():
        return False
    if stripped.lower() in STOPWORDS:
        return False
    # Stability: re-encoding the decoded string must give back exactly [tok_id].
    re_enc = tokenizer.encode(decoded, add_special_tokens=False)
    return re_enc == [tok_id]


def _find_candidates(
    window_ids: list[int],
    entity_cache: dict[int, bool],
    tokenizer,
) -> list[tuple[int, list[int]]]:
    """Return list of (token_id, [positions]) for repeated entity tokens in window.

    Only includes tokens with MIN_OCCURRENCES ≤ count ≤ MAX_OCCURRENCES.
    entity_cache speeds up repeated is_entity_token lookups across windows.
    """
    from collections import defaultdict
    pos_map: dict[int, list[int]] = defaultdict(list)
    for pos, tok in enumerate(window_ids):
        if tok not in entity_cache:
            entity_cache[tok] = _is_entity_token(tok, tokenizer)
        if entity_cache[tok]:
            pos_map[tok].append(pos)

    return [
        (tok, positions)
        for tok, positions in pos_map.items()
        if MIN_OCCURRENCES <= len(positions) <= MAX_OCCURRENCES
    ]


def _mask_window(window_ids: list[int], positions: list[int], mask_id: int) -> list[int]:
    masked = list(window_ids)
    for p in positions:
        masked[p] = mask_id
    return masked


def _build_single_items(
    window_ids: list[int],
    candidates: list[tuple[int, list[int]]],
    item_id_prefix: str,
    tokenizer,
    mask_id: int,
) -> list[dict]:
    items = []
    window_text = tokenizer.decode(window_ids)
    for cand_idx, (tok_id, positions) in enumerate(candidates):
        masked = _mask_window(window_ids, positions, mask_id)
        items.append({
            "item_id": f"{item_id_prefix}_s{cand_idx}",
            "corpus": item_id_prefix.split("_")[0],
            "track": "single_chain",
            "window_text": window_text,
            "masked_input_ids": masked,
            "mask_positions": sorted(positions),
            "gold_token_ids": [tok_id] * len(positions),
            "chain_labels": [0] * len(positions),
        })
    return items


def _build_multi_item(
    window_ids: list[int],
    candidates: list[tuple[int, list[int]]],
    item_id_prefix: str,
    tokenizer,
    mask_id: int,
) -> dict | None:
    """Build one multi_chain item from 2–4 distinct entity candidates.

    Greedily picks entities (fewest occurrences first) until MAX_HOLES would
    be exceeded or we run out of candidates.  Returns None if <2 entities fit.
    """
    # Sort by number of occurrences ascending (fewest first → easier to pack)
    sorted_cands = sorted(candidates, key=lambda x: len(x[1]))
    chosen: list[tuple[int, list[int]]] = []
    total_holes = 0
    for tok_id, positions in sorted_cands:
        if total_holes + len(positions) > MAX_HOLES:
            continue
        chosen.append((tok_id, positions))
        total_holes += len(positions)
        if len(chosen) == 4:
            break

    if len(chosen) < 2:
        return None

    # Build merged mask positions + gold + chain_labels, sorted by position.
    triples: list[tuple[int, int, int]] = []  # (position, tok_id, chain_idx)
    for chain_idx, (tok_id, positions) in enumerate(chosen):
        for pos in positions:
            triples.append((pos, tok_id, chain_idx))
    triples.sort()

    mask_positions = [t[0] for t in triples]
    gold_token_ids = [t[1] for t in triples]
    chain_labels = [t[2] for t in triples]

    masked = _mask_window(window_ids, mask_positions, mask_id)
    return {
        "item_id": f"{item_id_prefix}_m",
        "corpus": item_id_prefix.split("_")[0],
        "track": "multi_chain",
        "window_text": tokenizer.decode(window_ids),
        "masked_input_ids": masked,
        "mask_positions": mask_positions,
        "gold_token_ids": gold_token_ids,
        "chain_labels": chain_labels,
    }


def process_corpus(
    corpus_tag: str,
    texts: list[str],
    tokenizer,
    mask_id: int,
    max_docs: int | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Extract single_chain and multi_chain items from a list of documents.

    Returns (single_items, multi_items, stats).
    stats keys: docs_seen, windows_considered, items_single, items_multi.
    """
    single_items: list[dict] = []
    multi_items: list[dict] = []
    stats = {"docs_seen": 0, "windows_considered": 0,
             "items_single": 0, "items_multi": 0}

    entity_cache: dict[int, bool] = {}

    for doc_idx, text in enumerate(texts):
        if max_docs is not None and doc_idx >= max_docs:
            break
        stats["docs_seen"] += 1

        token_ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < WINDOW_SIZE:
            continue

        for win_start in range(0, len(token_ids) - WINDOW_SIZE + 1, STRIDE):
            window = token_ids[win_start: win_start + WINDOW_SIZE]
            stats["windows_considered"] += 1
            prefix = f"{corpus_tag}_d{doc_idx}_w{win_start}"

            candidates = _find_candidates(window, entity_cache, tokenizer)
            if not candidates:
                continue

            s_items = _build_single_items(window, candidates, prefix, tokenizer, mask_id)
            single_items.extend(s_items)
            stats["items_single"] += len(s_items)

            if len(candidates) >= 2:
                m_item = _build_multi_item(window, candidates, prefix, tokenizer, mask_id)
                if m_item is not None:
                    multi_items.append(m_item)
                    stats["items_multi"] += 1

    return single_items, multi_items, stats


def _write_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"  wrote {len(items):6d} items → {path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/rmc",
                   help="Output directory for .jsonl files")
    p.add_argument("--print-stats", action="store_true",
                   help="Print per-corpus selectivity table")
    p.add_argument("--max-docs-owt", type=int, default=None,
                   help="Cap OWT documents (for smoke testing)")
    p.add_argument("--max-docs-wt103", type=int, default=None,
                   help="Cap WT-103 documents (for smoke testing)")
    args = p.parse_args()

    from datasets import load_dataset  # type: ignore
    from diffusion_ebm.backbones.mdlm import MASK_TOKEN_ID

    out_dir = Path(args.out_dir)

    print("[build-rmc] loading tokenizer…")
    tokenizer = _make_tokenizer()

    # ── OWT heldout ──────────────────────────────────────────────────────────
    # Use the last 1000 documents by fixed HuggingFace ordering.
    # The m5c_train pipeline streams from the beginning of OWT, so these
    # tail documents are unlikely to have been used in scorer training.
    print("[build-rmc] loading OWT (last 1000 docs)…")
    owt_ds = load_dataset(
        "Skylion007/openwebtext", split="train[-1000:]", trust_remote_code=True
    )
    owt_texts = [ex["text"] for ex in owt_ds]
    print(f"[build-rmc]   {len(owt_texts)} OWT documents loaded")

    print("[build-rmc] extracting OWT heldout items…")
    owt_single, owt_multi, owt_stats = process_corpus(
        "owt_heldout", owt_texts, tokenizer, MASK_TOKEN_ID,
        max_docs=args.max_docs_owt,
    )

    # ── WikiText-103 validation ──────────────────────────────────────────────
    print("[build-rmc] loading WikiText-103 validation…")
    wt_ds = load_dataset(
        "wikitext", "wikitext-103-raw-v1", split="validation", trust_remote_code=True
    )
    wt_texts = [ex["text"] for ex in wt_ds if ex["text"].strip()]
    print(f"[build-rmc]   {len(wt_texts)} WT-103 non-empty paragraphs loaded")

    print("[build-rmc] extracting WikiText-103 items…")
    wt_single, wt_multi, wt_stats = process_corpus(
        "wikitext103", wt_texts, tokenizer, MASK_TOKEN_ID,
        max_docs=args.max_docs_wt103,
    )

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("\n[build-rmc] writing .jsonl files…")
    _write_jsonl(owt_single,  out_dir / "owt_heldout_single.jsonl")
    _write_jsonl(owt_multi,   out_dir / "owt_heldout_multi.jsonl")
    _write_jsonl(wt_single,   out_dir / "wikitext103_single.jsonl")
    _write_jsonl(wt_multi,    out_dir / "wikitext103_multi.jsonl")

    # ── Stats ─────────────────────────────────────────────────────────────────
    if args.print_stats:
        print("\n[build-rmc] === extraction selectivity ===")
        for tag, st, n_s, n_m in [
            ("owt_heldout",  owt_stats,  len(owt_single),  len(owt_multi)),
            ("wikitext103",  wt_stats,   len(wt_single),   len(wt_multi)),
        ]:
            win = st["windows_considered"]
            rate_s = n_s / win if win else 0.0
            rate_m = n_m / win if win else 0.0
            print(
                f"  {tag:<16s}  docs={st['docs_seen']:5d}  "
                f"windows={win:7d}  "
                f"single={n_s:6d} ({rate_s:.3f}/win)  "
                f"multi={n_m:6d} ({rate_m:.3f}/win)"
            )
    else:
        for tag, st in [("owt_heldout", owt_stats), ("wikitext103", wt_stats)]:
            print(
                f"[build-rmc] {tag}: "
                f"{st['items_single']} single, {st['items_multi']} multi "
                f"(from {st['docs_seen']} docs, {st['windows_considered']} windows)"
            )

    print("\n[build-rmc] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
