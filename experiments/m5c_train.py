"""M5c — train the learned pairwise scorer (Joint-Transition NCE).

One self-contained training script. Builds (positive, hard-negative)
pairs from a text corpus by:

  1. Tokenising a document and picking two non-special-token positions
     ``a < b`` with min separation.
  2. Masking those two positions and forwarding **frozen MDLM** to get
     ``(logits, last_hidden_state)`` at both positions.
  3. Drawing ``K`` hard negatives per pair: each ``(x_i_neg, x_j_neg)``
     is sampled independently from the top-``k`` softmax of the LM
     marginals at the two positions, with ``(x_i_neg, x_j_neg) ==
     (x_i_pos, x_j_pos)`` rejected. Each negative is **individually
     high-probability** under ``p_LM`` but jointly resampled, so the
     scorer learns to discriminate jointly-coherent from
     marginally-resampled pairs.

Loss: InfoNCE, ``-log(exp(ψ_pos) / (exp(ψ_pos) + Σ exp(ψ_neg)))``.

Data sources:
  * ``--corpus synthetic`` — replays the M3/M5b templates in a loop;
    only useful for the local smoke test.
  * ``--corpus owt`` — streams OpenWebText via HF datasets. Requires
    network. Default for the long training run.

Run (long training, A100):

    LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH" \\
    TRITON_LIBCUDA_PATH="/run/opengl-driver/lib" \\
        uv run python experiments/m5c_train.py \\
            --corpus owt --steps 200000 --batch 256 --neg-k 16 \\
            --ckpt-dir results/m5c

Run (8 GB laptop GPU, overnight):

    LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH" \\
    TRITON_LIBCUDA_PATH="/run/opengl-driver/lib" \\
        uv run python experiments/m5c_train.py \\
            --corpus owt --steps 100000 --batch 32 --grad-accum 4 \\
            --neg-k 8 --embed-dim 128 --head-dim 64 --mlp-dim 256 \\
            --ckpt-dir results/m5c_local

``--grad-accum N`` accumulates gradients across N micro-batches before each
optimizer step, so ``effective_batch = batch × grad_accum``.  Peak VRAM is
determined by ``--batch``; effective signal by the product.
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
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from diffusion_ebm.backbones.mdlm import MASK_TOKEN_ID, MDLM  # noqa: E402
from diffusion_ebm.factors.learned import PairwiseScorer  # noqa: E402


def _synthetic_docs() -> list[str]:
    """Tiny corpus for smoke testing — no network. Each "document" is a
    concatenation of short factual sentences so the per-doc token count
    is comfortably above smoke's seq_len.
    """
    sentences = [
        "Alice's favorite color is blue. Bob's favorite color is also blue.",
        "The variable x was assigned the value 7. Later, x was used in a loop.",
        "My name is Sam. You can call me Sam. I said Sam three times.",
        "The cat sat on the mat. The cat looked at the dog later that day.",
        "She packed her bag into the car's boot. Then she put a sturdy leather boot on her foot.",
        "He needed cash from the bank. He sat down by the river bank to think.",
        "The astronomer saw a distant star. The actor walked the carpet like a star.",
        "Alice greeted Bob warmly. Bob waved at Alice across the crowded room.",
        "The book was on the shelf. The shelf was made of dark wood and stood by the window.",
        "She wrote her name on the form. The form was sent to the office downtown.",
    ]
    docs: list[str] = []
    for i in range(512):
        # 4 random sentences per doc → ~50–80 GPT-2 tokens per doc.
        rng = random.Random(i)
        doc = " ".join(rng.sample(sentences, k=4))
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
    x_i_pos: torch.Tensor     # [B] long — token at pos_a in the *unmasked* doc
    x_j_pos: torch.Tensor     # [B]


def _build_batch_pretok(
    tok_docs: list[list[int]],
    seq_len: int,
    batch: int,
    min_pair_dist: int,
    rng: random.Random,
) -> PairBatch | None:
    """Sample one ``[batch]`` of (masked_ids, positions, true tokens).

    tok_docs is a list of already-tokenised documents (list[int], each with
    length >= seq_len+2).  No tokeniser call is made here.
    """
    masked_rows: list[torch.Tensor] = []
    pos_a_l: list[int] = []
    pos_b_l: list[int] = []
    x_i_l: list[int] = []
    x_j_l: list[int] = []

    attempts = 0
    while len(masked_rows) < batch and attempts < batch * 8:
        attempts += 1
        ids_full = rng.choice(tok_docs)
        if len(ids_full) < seq_len + 2:
            continue
        start = rng.randrange(0, len(ids_full) - seq_len + 1)
        ids = ids_full[start : start + seq_len]
        if len(ids) != seq_len:
            continue

        # pick two positions with min separation
        if seq_len <= min_pair_dist + 1:
            continue
        a = rng.randrange(0, seq_len - min_pair_dist - 1)
        b = rng.randrange(a + min_pair_dist, seq_len)
        if a == b:
            continue
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
    return PairBatch(
        masked_ids=torch.stack(masked_rows, dim=0),
        pos_a=torch.tensor(pos_a_l, dtype=torch.long),
        pos_b=torch.tensor(pos_b_l, dtype=torch.long),
        x_i_pos=torch.tensor(x_i_l, dtype=torch.long),
        x_j_pos=torch.tensor(x_j_l, dtype=torch.long),
    )


@torch.no_grad()
def _encode_and_negatives(
    mdlm: MDLM,
    pb: PairBatch,
    neg_k: int,
    top_k: int,
) -> dict[str, torch.Tensor]:
    """Forward frozen MDLM, gather (h_a, h_b), and draw hard negatives.

    Hard negatives = independent draws from the top-``top_k`` softmax of
    the per-position LM marginal at the two masked positions. Rejection
    of (x_i_pos, x_j_pos) is done with one resample per slot.
    """
    masked_ids = pb.masked_ids.to(mdlm.device)
    logits, hidden = mdlm.forward_hidden(masked_ids)
    logits = logits.float()
    logits[..., MASK_TOKEN_ID] = float("-inf")
    B = masked_ids.shape[0]

    rows = torch.arange(B, device=mdlm.device)
    h_a = hidden[rows, pb.pos_a.to(mdlm.device)]   # [B, d_h]
    h_b = hidden[rows, pb.pos_b.to(mdlm.device)]
    logits_a = logits[rows, pb.pos_a.to(mdlm.device)]  # [B, V]
    logits_b = logits[rows, pb.pos_b.to(mdlm.device)]

    topk_a_logits, topk_a_ids = torch.topk(logits_a, top_k, dim=-1)
    topk_b_logits, topk_b_ids = torch.topk(logits_b, top_k, dim=-1)
    p_a = F.softmax(topk_a_logits, dim=-1)
    p_b = F.softmax(topk_b_logits, dim=-1)

    sampled_a_idx = torch.multinomial(p_a, neg_k, replacement=True)  # [B, K]
    sampled_b_idx = torch.multinomial(p_b, neg_k, replacement=True)
    x_i_neg = topk_a_ids.gather(1, sampled_a_idx)  # [B, K]
    x_j_neg = topk_b_ids.gather(1, sampled_b_idx)

    # Rejection: while any negative still coincides exactly with the positive
    # pair, resample x_i_neg only at those slots. p ~ (1/k_eff)^2 so usually
    # one pass; at most a handful of iterations.
    pos_i = pb.x_i_pos.to(mdlm.device).unsqueeze(1)
    pos_j = pb.x_j_pos.to(mdlm.device).unsqueeze(1)
    for _ in range(8):
        same = (x_i_neg == pos_i) & (x_j_neg == pos_j)
        if not bool(same.any()):
            break
        resamp_a = torch.multinomial(p_a, neg_k, replacement=True)
        replace_ids = topk_a_ids.gather(1, resamp_a)
        x_i_neg = torch.where(same, replace_ids, x_i_neg)

    return {
        "h_a": h_a,
        "h_b": h_b,
        "x_i_pos": pb.x_i_pos.to(mdlm.device),
        "x_j_pos": pb.x_j_pos.to(mdlm.device),
        "x_i_neg": x_i_neg,
        "x_j_neg": x_j_neg,
    }


def _info_nce(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """``-log(exp(pos) / (exp(pos) + Σ exp(neg)))``, mean over batch."""
    cat = torch.cat([pos.unsqueeze(1), neg], dim=1)  # [B, 1+K]
    logsumexp = torch.logsumexp(cat, dim=1)
    return -(pos - logsumexp).mean()


def train(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"[m5c] device check; loading MDLM…", flush=True)
    mdlm = MDLM.load()
    mdlm.model.eval()
    for p in mdlm.model.parameters():
        p.requires_grad_(False)

    scorer = PairwiseScorer(
        hidden_dim=mdlm.model.config.hidden_dim
        if hasattr(mdlm.model.config, "hidden_dim")
        else 768,
        embed_dim=args.embed_dim,
        head_dim=args.head_dim,
        mlp_dim=args.mlp_dim,
        vocab_size=mdlm.vocab_size,
    ).to(mdlm.device)
    n_params = sum(p.numel() for p in scorer.parameters())
    print(f"[m5c] scorer params: {n_params/1e6:.2f}M", flush=True)
    eff_batch = args.batch * args.grad_accum
    print(f"[m5c] effective batch: {eff_batch} ({args.batch} × {args.grad_accum} accum)", flush=True)

    opt = torch.optim.AdamW(
        scorer.parameters(), lr=args.lr, weight_decay=args.wd
    )

    # Corpus — buffer upfront and pre-tokenise so no tokeniser calls in the loop
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
    tok_docs: list[list[int]] = _pretokenize_docs(
        mdlm.tokenizer, docs_raw, args.seq_len
    )
    del docs_raw
    print(
        f"[m5c] {len(tok_docs)} valid docs (>= {args.seq_len + 2} tokens)",
        flush=True,
    )
    if not tok_docs:
        raise RuntimeError("No valid docs after length filter — check corpus / seq_len")

    log: list[dict] = []
    t0 = time.time()
    losses_window: list[float] = []
    accum_loss = 0.0
    last_pos_scores = last_neg_scores = None
    for step in range(1, args.steps + 1):
        scorer.train()
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _acc in range(args.grad_accum):
            pb = _build_batch_pretok(
                tok_docs,
                seq_len=args.seq_len,
                batch=args.batch,
                min_pair_dist=args.min_pair_dist,
                rng=rng,
            )
            if pb is None:
                continue
            feats = _encode_and_negatives(mdlm, pb, args.neg_k, args.top_k)
            pos_scores, neg_scores = scorer.forward_pos_neg(
                feats["h_a"],
                feats["h_b"],
                feats["x_i_pos"],
                feats["x_j_pos"],
                feats["x_i_neg"],
                feats["x_j_neg"],
            )
            loss = _info_nce(pos_scores, neg_scores) / args.grad_accum
            loss.backward()
            step_loss += float(loss.item())
            last_pos_scores, last_neg_scores = pos_scores.detach(), neg_scores.detach()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
        opt.step()

        losses_window.append(step_loss)
        if step % args.log_every == 0:
            avg = sum(losses_window) / len(losses_window)
            losses_window.clear()
            # NCE accuracy proxy: fraction of batch where pos > max(neg)
            acc = 0.0
            if last_pos_scores is not None and last_neg_scores is not None:
                with torch.no_grad():
                    acc = float(
                        (last_pos_scores > last_neg_scores.max(dim=1).values).float().mean()
                    )
            elapsed = time.time() - t0
            rate = step / elapsed
            print(
                f"[m5c] step {step:6d}  loss={avg:.4f}  pos>max_neg={acc:.3f} "
                f"  {rate:.1f} steps/s",
                flush=True,
            )
            log.append(
                dict(step=step, loss=avg, acc=acc, time_s=elapsed)
            )

        if step % args.ckpt_every == 0 or step == args.steps:
            ckpt_path = os.path.join(args.ckpt_dir, f"scorer_step{step:08d}.pt")
            torch.save(
                dict(
                    state_dict=scorer.state_dict(),
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

        # Optionally refresh the pre-tokenised buffer (disabled by default)
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
            tok_docs.extend(new_tok)
            if len(tok_docs) > args.initial_buffer * 2:
                tok_docs[:] = tok_docs[-args.initial_buffer :]
            print(f"[m5c] refreshed buffer: {len(tok_docs)} docs", flush=True)

    log_path = os.path.join(args.ckpt_dir, "log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[m5c] wrote {log_path}; total {time.time() - t0:.1f}s", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=("synthetic", "owt"), default="synthetic")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--neg-k", type=int, default=8, help="negatives per positive")
    p.add_argument("--top-k", type=int, default=64,
                   help="LM-marginal top-k pool to sample negatives from")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--min-pair-dist", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="accumulate gradients over N micro-batches before each optimizer step")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--mlp-dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--initial-buffer", type=int, default=50000,
                   help="docs to buffer and pre-tokenise at startup")
    p.add_argument("--refresh-every", type=int, default=0,
                   help="re-fetch N docs from OWT every this many steps (0=disabled)")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument(
        "--ckpt-dir",
        type=str,
        default=os.path.join(_REPO_ROOT, "results/m5c"),
    )
    args = p.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
