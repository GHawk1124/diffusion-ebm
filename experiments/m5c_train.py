"""M5c — train the learned pairwise scorer (Tabular NCE, v2).

Self-contained training script.  For each training step:

  1. Sample a batch of (document, position_a, position_b) triples via
     the pair miner (tier-1: repeated tokens; tier-2: random fallback).
  2. Forward frozen MDLM to get ``(logits, hidden)`` at both positions.
  3. Compute the full ``[B, k, k]`` score table via ``PairwiseScorer.forward_table``.
  4. Minimise cross-entropy over the flattened k² logits with the corpus
     pair ``(x_i_pos_rank, x_j_pos_rank)`` as the positive cell.

This gives k²−1 = 4095 implicit negatives per positive at k=64 — ~256×
more gradient signal than v1's sampled-negative InfoNCE with K=16 — and
matches the inference geometry where THRML consumes the full ``[k, k]``
energy table.

Pair mining (tiered):
  * Tier 1 (--mine-repeats, default on): scan each window for positions
    a < b where ids[a] == ids[b] and the token is not a stopword /
    punctuation.  These are guaranteed joint constraints (same token must
    fill both holes).
  * Tier 2 (random fallback): current v1 behaviour — pick two random
    positions with min separation.

Synth-frac schedule:
  Two separate pools (OWT, synthetic). Each row in the batch is drawn
  from the synth pool with probability p_synth(step), which linearly
  anneals from --synth-frac-start to --synth-frac-end over
  --synth-anneal-steps.  High synth early bootstraps the scorer on hard
  equality signal; it then focuses on OWT.

Other v2 improvements: learnable temperature (PairwiseScorer.log_temp),
linear LR warmup, checkpoint resumption.

Run (long training, A100):

    LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH" \\
    TRITON_LIBCUDA_PATH="/run/opengl-driver/lib" \\
        uv run python experiments/m5c_train.py \\
            --corpus owt --steps 200000 --batch 256 \\
            --top-k 64 --ckpt-dir results/m5c

Run (8 GB laptop GPU, overnight):

    LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH" \\
    TRITON_LIBCUDA_PATH="/run/opengl-driver/lib" \\
        uv run python experiments/m5c_train.py \\
            --corpus owt --steps 100000 --batch 32 --grad-accum 4 \\
            --top-k 64 --embed-dim 128 --head-dim 64 --mlp-dim 256 \\
            --ckpt-dir results/m5c_local

``--grad-accum N`` accumulates gradients across N micro-batches before
each optimizer step, so ``effective_batch = batch × grad_accum``.
Peak VRAM is determined by ``--batch``; effective signal by the product.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from diffusion_ebm.backbones.mdlm import MASK_TOKEN_ID, MDLM  # noqa: E402
from diffusion_ebm.factors.learned import PairwiseScorer  # noqa: E402

# Stopwords and punctuation that don't carry useful joint constraint signal.
# Generated at startup from the tokenizer so the IDs are exact.
_STOP_STRINGS = [
    "the", " the", "a", " a", "an", " an", "of", " of", "and", " and",
    "or", " or", "in", " in", "to", " to", "for", " for", "on", " on",
    "with", " with", "at", " at", "by", " by", "from", " from",
    "as", " as", "is", " is", "are", " are", "was", " was", "were", " were",
    "be", " be", "been", " been", "have", " have", "has", " has", "had", " had",
    "it", " it", "its", " its", "he", " he", "she", " she", "they", " they",
    "we", " we", "you", " you", "I", " I",
    "this", " this", "that", " that", "which", " which",
    "not", " not", "no", " no", "but", " but", "if", " if",
    "The", " The", "A", " A", "It", " It", "He", " He", "She", " She",
    ",", " ,", ".", " .", "!", " !", "?", " ?", ";", " ;", ":", " :",
    "'", '"', "-", " -", "--", " --", "(", " (", ")", "[", "]",
    "'s", " 's", "n't", " n't", "'re", " 're", "'ve", " 've", "'ll", " 'll",
    "\n", "\n\n", " ", "  ",
]


def _make_stop_ids(tokenizer) -> set[int]:
    """Return GPT-2 token IDs for stopwords and punctuation."""
    ids: set[int] = {MASK_TOKEN_ID}
    for s in _STOP_STRINGS:
        toks = tokenizer(s, add_special_tokens=False)["input_ids"]
        if len(toks) == 1:
            ids.add(toks[0])
    return ids


def _synthetic_docs() -> list[str]:
    """Expanded synthetic corpus — hybrid programmatic + curated, ~300 seed sentences.

    Four axes targeting the v2 failure modes diagnosed in A3:
      1. Variable names  — programmatic (21 identifiers × 5 phrasings = 105 sentences)
      2. Person names    — programmatic (20 names × 5 phrasings = 100 sentences)
      3. Polyseme pairs  — hand-curated (21 sentences, same token two senses)
      4. Multi-group     — programmatic/curated (64 + 6 = 70 sentences)

    Tokenisation note: all templates place repeat tokens in space-prefixed
    positions so GPT-2 assigns the same token ID to both occurrences, enabling
    tier-1 pair mining (ids[a] == ids[b]).
    """
    # ---- Axis 1: variable names (programmatic) --------------------------------
    _VARS = [
        "x", "y", "z", "n", "m", "i", "j", "k", "s", "t",
        "p", "q", "r", "v", "w", "f", "c", "e", "g", "h", "d",
    ]
    _VAR_PHRASINGS = [
        "Let {v} = 5. The function returns {v} + 1 as the result.",
        "We set {v} to zero. Each iteration increments {v} by one.",
        "Initialize {v} at the start. After the loop, {v} holds the final count.",
        "The counter {v} starts low. When {v} exceeds the threshold, we stop.",
        "Define {v} = 10. Then {v} is passed to the helper function.",
    ]
    var_sentences = [p.format(v=v) for v in _VARS for p in _VAR_PHRASINGS]

    # ---- Axis 2: person names (programmatic) ----------------------------------
    _NAMES = [
        "Maria", "Carlos", "Ana", "Diego", "Sofia", "Luis", "Elena", "Marco",
        "Lucia", "Pablo", "Sarah", "Daniel", "Rachel", "Thomas", "Hannah",
        "Oliver", "Emma", "Henry", "Grace", "James",
    ]
    _NAME_PHRASINGS = [
        "{n} arrived first. The team waited for {n} to begin the meeting.",
        "They called him {n}. Everyone knew {n} from the old neighborhood.",
        "{n} stepped forward calmly. A moment later {n} spoke to the crowd.",
        "The letter was addressed to {n}. Without hesitation, {n} opened it.",
        "{n} had been quiet all evening. Then {n} suddenly stood up.",
    ]
    name_sentences = [p.format(n=name) for name in _NAMES for p in _NAME_PHRASINGS]

    # ---- Axis 3: polyseme pairs (hand-curated) --------------------------------
    polyseme_sentences = [
        # Keep original car-boot (works in v2).
        "She packed her bag into the car's boot. Then she put a sturdy leather boot on her foot.",
        # New polyseme pairs: same surface token, two different senses.
        "He drew a line on the map. The new policy crossed a line that day.",
        "The spring in the clock had broken. She loved the spring season most.",
        "He withdrew cash from the bank. He sat on the river bank to rest.",
        "The astronomer saw a distant star. The actor walked the carpet like a star.",
        "She booked a flight at dawn. The fugitive had taken flight overnight.",
        "The pitcher threw a curve ball. He filled the pitcher with cold water.",
        "The bat flew out of the cave. He swung the bat with both hands.",
        "She locked the trunk of the car. An elephant raised its trunk high.",
        "He needed a new mouse for the computer. The cat chased the mouse across the floor.",
        "She pressed her palm against the wall. The palm tree swayed in the warm breeze.",
        "The seal on the envelope was broken. A seal rested on the rocky shore.",
        "He offered a bow to the audience. She noticed the bow tied in her hair.",
        "The mole in the garden had burrowed deep. The spy was later identified as a mole.",
        "She ordered a mint chocolate dessert. The mint grew wild along the garden path.",
        "He cast a spell on the audience. She checked the spell carefully in the dictionary.",
        "He tightened his tie before the interview. The tie between the two teams was broken.",
        "She found a well behind the old house. He replied that he was doing quite well.",
        "The scale tipped in her favor. She played a musical scale slowly on the piano.",
        "The wave crashed against the shore. She gave a friendly wave as she left the room.",
        "He measured the yard with a tape. The dog ran freely across the yard.",
    ]

    # ---- Axis 4: multi-group templates (programmatic + curated) ---------------
    # Each sentence has two independent equality pairs: name + city.
    _MG_NAMES = ["Ada", "Carlos", "Maria", "Anna", "Leo", "Rosa", "Max", "Elena"]
    _MG_CITIES = ["London", "Madrid", "Paris", "Berlin", "Rome", "Vienna", "Prague", "Lisbon"]
    _MG_ROLES = [
        "author", "painter", "architect", "detective",
        "composer", "scientist", "poet", "philosopher",
    ]
    _MG_TEMPLATES = [
        "The {role} {name} was born in {city}. {name} later wrote about {city} at length.",
        "The {role} {name} lived and worked in {city}. {name} rarely left {city} for long.",
    ]
    multi_group_sentences: list[str] = []
    for name_idx, name in enumerate(_MG_NAMES):
        for city_idx, city in enumerate(_MG_CITIES):
            role = _MG_ROLES[(name_idx + city_idx) % len(_MG_ROLES)]
            tmpl = _MG_TEMPLATES[(name_idx * 3 + city_idx) % len(_MG_TEMPLATES)]
            multi_group_sentences.append(tmpl.format(name=name, city=city, role=role))
    # Original seed sentences (keep for continuity).
    multi_group_sentences += [
        "Alice's favorite color is blue. Bob's favorite color is also blue.",
        "The cat sat on the mat. The cat looked at the dog later that day.",
        "The book was on the shelf. The shelf was made of dark wood and stood by the window.",
        "She wrote her name on the form. The form was sent to the office downtown.",
        "My name is Sam. You can call me Sam. I said Sam three times.",
        "Alice greeted Bob warmly. Bob waved at Alice across the crowded room.",
    ]

    all_sentences = var_sentences + name_sentences + polyseme_sentences + multi_group_sentences

    docs: list[str] = []
    for i in range(5000):
        rng = random.Random(i)
        doc = " ".join(rng.sample(all_sentences, k=6))
        docs.append(doc)
    return docs


def _stream_owt(min_tokens: int = 64):
    """Lazily yields OpenWebText documents (strings). Requires network."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        "Skylion007/openwebtext", split="train", streaming=True
    )
    for ex in ds:
        text = ex.get("text", "")
        if len(text.split()) >= min_tokens:
            yield text


def _pretokenize_docs(
    tokenizer, docs: list[str], seq_len: int
) -> list[list[int]]:
    """Tokenise documents in batches; keep only those with >= seq_len+2 tokens."""
    result: list[list[int]] = []
    chunk = 1024
    for i in range(0, len(docs), chunk):
        ids_batch = tokenizer(
            docs[i : i + chunk], add_special_tokens=False
        )["input_ids"]
        result.extend(ids for ids in ids_batch if len(ids) >= seq_len + 2)
    return result


@dataclass
class PairBatch:
    masked_ids: torch.Tensor  # [B, L]
    pos_a: torch.Tensor       # [B] long — position index in [0, L)
    pos_b: torch.Tensor       # [B]
    x_i_pos: torch.Tensor     # [B] long — true token at pos_a
    x_j_pos: torch.Tensor     # [B] long — true token at pos_b


def _synth_frac(step: int, start: float, end: float, anneal_steps: int) -> float:
    """Linear anneal from start to end over anneal_steps steps."""
    if anneal_steps <= 0 or step >= anneal_steps:
        return end
    t = step / anneal_steps
    return start + t * (end - start)


def _build_batch_pretok(
    tok_docs_main: list[list[int]],
    tok_docs_synth: list[list[int]],
    p_synth: float,
    seq_len: int,
    batch: int,
    min_pair_dist: int,
    rng: random.Random,
    mine_repeats: bool,
    stop_ids: set[int],
) -> tuple[PairBatch, dict[str, int]] | None:
    """Sample one batch of (masked_ids, positions, true tokens).

    Returns (PairBatch, tier_counts) where tier_counts maps "tier1" and
    "tier2" to the number of batch items sourced from each mining tier.
    Returns None if we cannot fill the batch after the attempt budget.
    """
    masked_rows: list[torch.Tensor] = []
    pos_a_l: list[int] = []
    pos_b_l: list[int] = []
    x_i_l: list[int] = []
    x_j_l: list[int] = []
    n_tier1 = 0
    n_tier2 = 0

    attempts = 0
    while len(masked_rows) < batch and attempts < batch * 8:
        attempts += 1

        # Choose pool: synth (high-MI templates) or main (OWT).
        if tok_docs_synth and rng.random() < p_synth:
            ids_full = rng.choice(tok_docs_synth)
        else:
            ids_full = rng.choice(tok_docs_main)

        if len(ids_full) < seq_len + 2:
            continue
        start = rng.randrange(0, len(ids_full) - seq_len + 1)
        ids = ids_full[start : start + seq_len]
        if len(ids) != seq_len:
            continue

        a: int | None = None
        b: int | None = None

        if mine_repeats:
            # Tier 1: scan for repeated non-stopword token pairs.
            repeat_pairs: list[tuple[int, int]] = []
            seen: dict[int, list[int]] = {}
            for idx, tok_id in enumerate(ids):
                if tok_id not in stop_ids:
                    if tok_id in seen:
                        for prev in seen[tok_id]:
                            if idx - prev >= min_pair_dist:
                                repeat_pairs.append((prev, idx))
                    seen.setdefault(tok_id, []).append(idx)
            if repeat_pairs:
                a, b = rng.choice(repeat_pairs)
                n_tier1 += 1

        if a is None:
            # Tier 2 (random fallback): same logic as v1.
            if seq_len <= min_pair_dist + 1:
                continue
            a = rng.randrange(0, seq_len - min_pair_dist - 1)
            b = rng.randrange(a + min_pair_dist, seq_len)
            if a == b:
                continue
            n_tier2 += 1

        x_i_l.append(ids[a])
        x_j_l.append(ids[b])
        masked = list(ids)
        masked[a] = MASK_TOKEN_ID
        masked[b] = MASK_TOKEN_ID
        masked_rows.append(torch.tensor(masked, dtype=torch.long))
        pos_a_l.append(a)
        pos_b_l.append(b)

    if len(masked_rows) < batch:
        return None

    pb = PairBatch(
        masked_ids=torch.stack(masked_rows, dim=0),
        pos_a=torch.tensor(pos_a_l, dtype=torch.long),
        pos_b=torch.tensor(pos_b_l, dtype=torch.long),
        x_i_pos=torch.tensor(x_i_l, dtype=torch.long),
        x_j_pos=torch.tensor(x_j_l, dtype=torch.long),
    )
    return pb, {"tier1": n_tier1, "tier2": n_tier2}


@torch.no_grad()
def _encode_and_topk(
    mdlm: MDLM,
    pb: PairBatch,
    top_k: int,
) -> dict:
    """Forward frozen MDLM; return hidden states and top-k support per position.

    Returns a dict with:
      h_a, h_b          [B, d_h] — hidden state at masked positions
      topk_a_ids,
      topk_b_ids        [B, top_k] long — top-k token IDs from LM marginal
      pos_a_idx,
      pos_b_idx         [B] long — rank of x_i_pos / x_j_pos in topk;
                                   -1 when absent from support
      in_topk           [B] bool — True when both tokens are in their top-k
      topk_recall       float — fraction of batch items with in_topk=True
    """
    dev = mdlm.device
    masked_ids = pb.masked_ids.to(dev)
    logits, hidden = mdlm.forward_hidden(masked_ids)
    logits = logits.float()
    logits[..., MASK_TOKEN_ID] = float("-inf")
    B = masked_ids.shape[0]
    rows = torch.arange(B, device=dev)

    h_a = hidden[rows, pb.pos_a.to(dev)]   # [B, d_h]
    h_b = hidden[rows, pb.pos_b.to(dev)]
    logits_a = logits[rows, pb.pos_a.to(dev)]  # [B, V]
    logits_b = logits[rows, pb.pos_b.to(dev)]

    _, topk_a_ids = torch.topk(logits_a, top_k, dim=-1)  # [B, k]
    _, topk_b_ids = torch.topk(logits_b, top_k, dim=-1)

    x_i_pos = pb.x_i_pos.to(dev)
    x_j_pos = pb.x_j_pos.to(dev)

    match_a = topk_a_ids == x_i_pos.unsqueeze(1)  # [B, k]
    match_b = topk_b_ids == x_j_pos.unsqueeze(1)
    in_topk_a = match_a.any(dim=1)
    in_topk_b = match_b.any(dim=1)
    in_topk = in_topk_a & in_topk_b

    # argmax returns 0 when all False; masked_fill fixes out-of-topk rows.
    pos_a_idx = match_a.int().argmax(dim=1).masked_fill(~in_topk_a, -1)
    pos_b_idx = match_b.int().argmax(dim=1).masked_fill(~in_topk_b, -1)

    return {
        "h_a": h_a,
        "h_b": h_b,
        "topk_a_ids": topk_a_ids,
        "topk_b_ids": topk_b_ids,
        "pos_a_idx": pos_a_idx,
        "pos_b_idx": pos_b_idx,
        "in_topk": in_topk,
        "topk_recall": float(in_topk.float().mean()),
    }


def _tabular_nce(
    scorer: PairwiseScorer,
    feats: dict,
) -> torch.Tensor | None:
    """Cross-entropy over the flattened [k, k] score table.

    Skips batch rows where the positive token is outside the top-k support.
    Returns None if no valid rows remain (rare; only when topk_recall=0).
    """
    mask = feats["in_topk"]
    if not mask.any():
        return None
    h_a = feats["h_a"][mask]
    h_b = feats["h_b"][mask]
    ids_a = feats["topk_a_ids"][mask]  # [B', k]
    ids_b = feats["topk_b_ids"][mask]
    pos_a = feats["pos_a_idx"][mask]   # [B'] long, valid (≥ 0)
    pos_b = feats["pos_b_idx"][mask]
    k = ids_a.shape[1]
    table = scorer.forward_table(h_a, h_b, ids_a, ids_b)  # [B', k, k]
    targets = pos_a * k + pos_b                            # [B'] long
    return F.cross_entropy(table.flatten(1), targets)


def train(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print("[m5c] loading MDLM…", flush=True)
    mdlm = MDLM.load()
    mdlm.model.eval()
    for p in mdlm.model.parameters():
        p.requires_grad_(False)

    hidden_dim = (
        mdlm.model.config.hidden_dim
        if hasattr(mdlm.model.config, "hidden_dim")
        else 768
    )
    scorer = PairwiseScorer(
        hidden_dim=hidden_dim,
        embed_dim=args.embed_dim,
        head_dim=args.head_dim,
        mlp_dim=args.mlp_dim,
        vocab_size=mdlm.vocab_size,
    ).to(mdlm.device)
    n_params = sum(p.numel() for p in scorer.parameters())
    print(f"[m5c] scorer params: {n_params / 1e6:.2f}M", flush=True)

    eff_batch = args.batch * args.grad_accum
    print(
        f"[m5c] effective batch: {eff_batch} ({args.batch} × {args.grad_accum} accum)",
        flush=True,
    )

    opt = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.lr_warmup_steps,
    )

    start_step = 1
    if args.resume:
        print(f"[m5c] loading checkpoint {args.resume}…", flush=True)
        blob = torch.load(args.resume, map_location=mdlm.device, weights_only=False)
        scorer.load_state_dict(blob["state_dict"])
        if "opt_state" in blob:
            opt.load_state_dict(blob["opt_state"])
        start_step = blob["step"] + 1
        # Skip the warmup phase: LinearLR.get_lr() is multiplicative relative
        # to the current LR, so re-running warmup from a restored LR would
        # overshoot by end_factor/start_factor (10×). Fast-forwarding
        # last_epoch past total_iters makes all future sched.step() calls
        # no-ops, and we restore args.lr directly.
        sched.last_epoch = args.lr_warmup_steps
        for pg in opt.param_groups:
            pg["lr"] = args.lr
        print(f"[m5c] resuming from step {start_step}", flush=True)

    # Build stop-word ID set for repeat-token pair mining.
    stop_ids = _make_stop_ids(mdlm.tokenizer) if args.mine_repeats else set()

    # ---- Corpus ----------------------------------------------------------------
    if args.corpus == "synthetic":
        docs_raw = _synthetic_docs()
        doc_iter = None
    elif args.corpus == "owt":
        docs_raw = []
        doc_iter = _stream_owt()
        print(f"[m5c] buffering up to {args.initial_buffer} OWT docs…", flush=True)
        for _ in range(args.initial_buffer):
            try:
                docs_raw.append(next(doc_iter))
            except StopIteration:
                doc_iter = None
                break
        print(f"[m5c] buffered {len(docs_raw)} raw docs", flush=True)
    else:
        raise ValueError(args.corpus)

    print(f"[m5c] pre-tokenising {len(docs_raw)} docs…", flush=True)
    tok_docs_main: list[list[int]] = _pretokenize_docs(
        mdlm.tokenizer, docs_raw, args.seq_len
    )
    del docs_raw
    print(
        f"[m5c] {len(tok_docs_main)} valid docs (>= {args.seq_len + 2} tokens)",
        flush=True,
    )
    if not tok_docs_main:
        raise RuntimeError(
            "No valid docs after length filter — check corpus / seq_len"
        )

    # Synthetic pool (separate from OWT; annealed in during training).
    tok_docs_synth: list[list[int]] = []
    if args.synth_frac_start > 0 or args.synth_frac_end > 0:
        synth_raw = _synthetic_docs()
        tok_docs_synth = _pretokenize_docs(mdlm.tokenizer, synth_raw, args.seq_len)
        print(
            f"[m5c] synthetic pool: {len(tok_docs_synth)} docs "
            f"(p_synth {args.synth_frac_start:.0%} → {args.synth_frac_end:.0%} "
            f"over {args.synth_anneal_steps} steps)",
            flush=True,
        )

    # ---- Training loop ---------------------------------------------------------
    log: list[dict] = []
    t0 = time.time()
    losses_window: list[float] = []
    window_tier1 = 0
    window_tier2 = 0
    last_topk_recall = 0.0

    for step in range(start_step, args.steps + 1):
        scorer.train()
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_tier1 = 0
        step_tier2 = 0

        p_synth = _synth_frac(
            step, args.synth_frac_start, args.synth_frac_end, args.synth_anneal_steps
        )

        for _acc in range(args.grad_accum):
            result = _build_batch_pretok(
                tok_docs_main=tok_docs_main,
                tok_docs_synth=tok_docs_synth,
                p_synth=p_synth,
                seq_len=args.seq_len,
                batch=args.batch,
                min_pair_dist=args.min_pair_dist,
                rng=rng,
                mine_repeats=args.mine_repeats,
                stop_ids=stop_ids,
            )
            if result is None:
                continue
            pb, tc = result
            step_tier1 += tc.get("tier1", 0)
            step_tier2 += tc.get("tier2", 0)

            feats = _encode_and_topk(mdlm, pb, args.top_k)
            last_topk_recall = feats["topk_recall"]

            loss = _tabular_nce(scorer, feats)
            if loss is None:
                continue

            (loss / args.grad_accum).backward()
            step_loss += float(loss.item()) / args.grad_accum

        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
        opt.step()
        sched.step()
        if args.temp_clamp_max is not None or args.temp_clamp_min is not None:
            with torch.no_grad():
                scorer.log_temp.data.clamp_(
                    min=args.temp_clamp_min,
                    max=args.temp_clamp_max,
                )

        losses_window.append(step_loss)
        window_tier1 += step_tier1
        window_tier2 += step_tier2

        if step % args.log_every == 0:
            avg_loss = sum(losses_window) / max(len(losses_window), 1)
            losses_window.clear()
            total_pairs = window_tier1 + window_tier2
            tier1_frac = window_tier1 / max(total_pairs, 1)
            window_tier1 = 0
            window_tier2 = 0
            temp = float(scorer.log_temp.detach().exp())
            lr = opt.param_groups[0]["lr"]
            elapsed = time.time() - t0
            rate = (step - start_step + 1) / max(elapsed, 1e-6)
            print(
                f"[m5c] step {step:6d}  loss={avg_loss:.4f}"
                f"  topk_recall={last_topk_recall:.3f}"
                f"  tier1={tier1_frac:.2f}"
                f"  temp={temp:.3f}"
                f"  lr={lr:.2e}"
                f"  {rate:.1f} steps/s",
                flush=True,
            )
            log.append(
                dict(
                    step=step,
                    loss=avg_loss,
                    topk_recall=last_topk_recall,
                    tier1_frac=tier1_frac,
                    temp=temp,
                    lr=lr,
                    time_s=elapsed,
                )
            )

        if step % args.ckpt_every == 0 or step == args.steps:
            ckpt_path = os.path.join(
                args.ckpt_dir, f"scorer_step{step:08d}.pt"
            )
            torch.save(
                dict(
                    state_dict=scorer.state_dict(),
                    opt_state=opt.state_dict(),
                    config=dict(
                        hidden_dim=scorer.hidden_dim,
                        embed_dim=scorer.embed_dim,
                        head_dim=scorer.head_dim,
                        vocab_size=scorer.vocab_size,
                    ),
                    args=vars(args),
                    step=step,
                ),
                ckpt_path,
            )
            print(f"[m5c] wrote {ckpt_path}", flush=True)

        # Optionally refresh the pre-tokenised buffer.
        if (
            doc_iter is not None
            and args.refresh_every > 0
            and step % args.refresh_every == 0
        ):
            new_raw: list[str] = []
            try:
                for _ in range(2048):
                    new_raw.append(next(doc_iter))
            except StopIteration:
                doc_iter = None
            new_tok = _pretokenize_docs(mdlm.tokenizer, new_raw, args.seq_len)
            tok_docs_main.extend(new_tok)
            if len(tok_docs_main) > args.initial_buffer * 2:
                tok_docs_main[:] = tok_docs_main[-args.initial_buffer :]
            print(f"[m5c] refreshed buffer: {len(tok_docs_main)} docs", flush=True)

    log_path = os.path.join(args.ckpt_dir, "log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(
        f"[m5c] wrote {log_path}; total {time.time() - t0:.1f}s", flush=True
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=("synthetic", "owt"), default="synthetic")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--top-k", type=int, default=64,
                   help="LM-marginal top-k support; defines the k×k tabular NCE table")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--min-pair-dist", type=int, default=4)
    p.add_argument("--mine-repeats", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="tier-1 mining: prefer repeated-token pairs over random")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="accumulate gradients over N micro-batches before optimizer step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--lr-warmup-steps", type=int, default=1000,
                   help="linear LR warmup from lr/10 to lr over this many steps")
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--mlp-dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--initial-buffer", type=int, default=50000,
                   help="docs to buffer and pre-tokenise at startup")
    p.add_argument("--refresh-every", type=int, default=0,
                   help="re-fetch N docs from OWT every this many steps (0=disabled)")
    p.add_argument("--synth-frac-start", type=float, default=0.20,
                   help="initial synthetic-pool sampling probability")
    p.add_argument("--synth-frac-end", type=float, default=0.05,
                   help="final synthetic-pool sampling probability after anneal")
    p.add_argument("--synth-anneal-steps", type=int, default=50_000,
                   help="steps over which synth-frac anneals from start to end")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--ckpt-dir", type=str,
                   default=os.path.join(_REPO_ROOT, "results/m5c"))
    p.add_argument("--resume", type=str, default=None,
                   help="path to a checkpoint to resume from")
    p.add_argument("--temp-clamp-min", type=float, default=None,
                   help="lower bound on log_temp after each optimizer step (default: no clamp)")
    p.add_argument("--temp-clamp-max", type=float, default=None,
                   help="upper bound on log_temp after each optimizer step (default: no clamp)")
    args = p.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
