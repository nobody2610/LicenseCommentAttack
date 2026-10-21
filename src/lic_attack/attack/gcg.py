"""GCG-style coordinate-gradient attack on the comment region.

Comment tokens are discrete, so we cannot gradient-descend them directly. Instead
(Zou et al. 2023):

  1. Represent the comment as a one-hot matrix C in {0,1}^{m x V}.
  2. Compute dLoss/dC via the differentiable embedding path (scoring.objective_from_onehot).
  3. At each position, the most promising replacement tokens are those with the most
     negative gradient (lower loss). Take top-k per position, restricted to the
     comment-safe vocab.
  4. Sample a batch of single-token substitutions from those candidates, evaluate the
     true (discrete) loss of each by a forward pass, and keep the best.
  5. Repeat until the true-label margin goes negative (argmax flips) or steps run out.

The license header and source are frozen — only comment_ids changes — so the result
is, by construction, a comment-only perturbation that does not touch the source.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from tqdm import tqdm

from ..model.base import DetectorModel, PromptSegments
from .scoring import (
    LabelScores,
    objective_from_onehot,
    score_labels_batch,
    score_labels_from_embeds,
)


@dataclass
class AttackConfig:
    objective: str = "margin"
    comment_tokens: int = 20
    top_k: int = 256
    batch_size: int = 64    # wider search; the filter in step 3b costs ~2% of candidates
    n_steps: int = 200
    margin_success: float = 0.0
    seed: int = 0


@dataclass
class AttackTrace:
    step: int
    loss: float
    margin: float
    argmax: str
    comment: str


@dataclass
class AttackResult:
    true_label: str
    success: bool
    steps_used: int
    clean: LabelScores
    final: LabelScores
    final_comment_ids: torch.Tensor
    final_comment_text: str
    n_filtered: int = 0      # candidates rejected as unrealizable, summed over steps
    trace: list[AttackTrace] = field(default_factory=list)


def _onehot(ids: torch.Tensor, vocab_size: int, device, dtype) -> torch.Tensor:
    oh = torch.zeros(len(ids), vocab_size, device=device, dtype=dtype)
    oh[torch.arange(len(ids)), ids.to(device)] = 1.0
    return oh


def _init_comment_ids(model: DetectorModel, safe_ids: torch.Tensor, m: int,
                      gen: torch.Generator) -> torch.Tensor:
    """Initialize the comment with a benign, REALIZABLE run of m tokens.

    The comment must start from a non-flipped state so any flip is attributable to
    the optimization. It must also survive a text round trip, because the
    invertibility filter in run_gcg rejects candidates that do not: starting from an
    unrealizable sequence leaves every single-token substitution unrealizable too, and
    the search stalls with an empty candidate set.

    A run of m space tokens -- the previous initializer -- is exactly such a sequence:
    the tokenizer has tokens for runs of whitespace, so [space] * 20 decodes to a
    20-space string that re-encodes to ONE token. Repeating a short word with
    separating spaces keeps each token at a boundary the tokenizer will reproduce.
    """
    for filler in ("x ", "the "):
        ids = model.encode((filler * m).strip(), add_special=False)
        if len(ids) >= m:
            ids = ids[:m]
            if torch.equal(model.encode(model.decode(ids), add_special=False).cpu(),
                           ids.cpu()):
                return ids.clone()
    # Fall back to random safe tokens; still checked by the filter on every step.
    pick = safe_ids[torch.randint(0, len(safe_ids), (m,), generator=gen)]
    return pick.clone()


def run_gcg(
    model: DetectorModel,
    seg: PromptSegments,
    labels: list[str],
    true_label: str,
    cfg: AttackConfig,
    calib: torch.Tensor | None = None,
) -> AttackResult:
    device = model.device
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    n_filtered = 0          # candidates dropped by the invertibility filter (3b)
    safe_mask = model.comment_safe_mask().to(device)
    safe_ids = torch.nonzero(safe_mask, as_tuple=False).squeeze(-1)
    label_token_ids = [model.encode(lbl, add_special=False) for lbl in labels]

    comment_ids = _init_comment_ids(model, safe_ids.cpu(), cfg.comment_tokens, gen).to(device)

    clean = score_labels_from_embeds(model, seg, comment_ids, labels, label_token_ids, calib)
    V = model.vocab_size
    emb_dtype = model.embedding_matrix.dtype

    trace: list[AttackTrace] = []
    best = score_labels_from_embeds(model, seg, comment_ids, labels, label_token_ids, calib)
    best_margin = float(best.margin(true_label))
    success = best.argmax_label() != true_label

    pbar = tqdm(range(cfg.n_steps), desc=f"GCG[{true_label}]", unit="step",
                leave=False, dynamic_ncols=True)
    for step in pbar:
        if success and best_margin < cfg.margin_success:
            break

        # 1) gradient of loss wrt one-hot
        onehot = _onehot(comment_ids, V, device, emb_dtype).requires_grad_(True)
        loss = objective_from_onehot(
            model, seg, onehot, labels, true_label, label_token_ids, cfg.objective,
            calib=calib,
        )
        grad = torch.autograd.grad(loss, onehot)[0]  # [m, V]

        # 2) restrict to comment-safe vocab; pick top-k most loss-reducing tokens/pos
        grad = grad.masked_fill(~safe_mask.unsqueeze(0), float("inf"))
        topk = (-grad).topk(min(cfg.top_k, int(safe_mask.sum())), dim=1).indices  # [m, k]

        # 3) sample a batch of single-position substitutions
        m = comment_ids.shape[0]
        cand = comment_ids.unsqueeze(0).repeat(cfg.batch_size, 1)  # [B, m]
        pos = torch.randint(0, m, (cfg.batch_size,), generator=gen)
        kcol = torch.randint(0, topk.shape[1], (cfg.batch_size,), generator=gen)
        new_tok = topk[pos, kcol]
        cand[torch.arange(cfg.batch_size), pos] = new_tok.to(cand.device)

        # 3b) drop candidates whose token sequence does not survive a text round trip.
        #
        # The threat model has the attacker append TEXT to a source file; the detector
        # re-tokenises whatever was written. BPE is a function from text to tokens, but
        # many token sequences decode to the same string and encode() returns only the
        # canonical one, so a sequence GCG reaches in token space is often unreachable
        # from text. Measured on the completed runs, only 7-16% of accepted comments
        # still re-tokenise to their nominal length -- i.e. most of what the search was
        # optimising could never be deployed as a comment.
        #
        # This is the invertibility filter from the original GCG implementation
        # (Zou et al., get_filtered_cands), which compares the re-tokenisation of each
        # decoded candidate against the candidate itself. It was missing here. The cost
        # is one decode+encode per candidate on CPU, negligible beside the forward pass,
        # and it shrinks the effective search width -- which is why batch_size is raised
        # to compensate rather than left at its pre-filter value.
        keep = [i for i in range(cand.shape[0])
                if torch.equal(
                    model.encode(model.decode(cand[i]), add_special=False).cpu(),
                    cand[i].cpu())]
        if not keep:
            # Every substitution this step was unrealizable; skip to the next proposal
            # round rather than accepting something that cannot be written to a file.
            continue
        cand = cand[keep]
        n_filtered += cfg.batch_size - len(keep)

        # 4) evaluate true discrete loss of each candidate in ONE batched pass,
        #    keep the best. (Was a Python loop of B separate forward passes, each
        #    looping over labels — the dominant cost. score_labels_batch runs L
        #    forwards over the full candidate batch instead of B*L single forwards.)
        cand_logp = score_labels_batch(model, seg, cand, labels, label_token_ids, calib)  # [B, L]
        ti = labels.index(true_label)
        mask = torch.ones(len(labels), dtype=torch.bool, device=cand_logp.device)
        mask[ti] = False
        if cfg.objective == "margin":
            # margin(true) = logp[true] - max_{y!=true} logp[y], per candidate
            losses = cand_logp[:, ti] - cand_logp[:, mask].max(dim=1).values  # [B]
        else:
            lp = torch.log_softmax(cand_logp, dim=1)
            ent = -(lp.exp() * lp).sum(dim=1)        # [B] entropy per candidate
            losses = -ent
        best_idx = int(losses.argmin())
        best_b_loss = float(losses[best_idx])
        best_b = cand[best_idx].clone()
        best_b_scores = LabelScores(logprobs=cand_logp[best_idx].clone(), labels=labels)

        # 5) accept if it improves over current
        cur_loss = best_margin if cfg.objective == "margin" else float(-best.entropy())
        if best_b_loss < cur_loss:
            comment_ids = best_b
            best = best_b_scores
            best_margin = float(best.margin(true_label))
            success = best.argmax_label() != true_label

        trace.append(AttackTrace(
            step=step, loss=best_b_loss, margin=best_margin,
            argmax=best.argmax_label(), comment=model.decode(comment_ids),
        ))
        pbar.set_postfix(margin=f"{best_margin:.3f}", flipped=success)
    pbar.close()

    return AttackResult(
        true_label=true_label,
        success=best.argmax_label() != true_label,
        steps_used=len(trace),
        clean=clean,
        final=best,
        final_comment_ids=comment_ids,
        final_comment_text=model.decode(comment_ids),
        n_filtered=n_filtered,
        trace=trace,
    )
