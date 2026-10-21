"""M4 rewriter: multi-operator simulated-annealing rewrite of a successful attack comment.

M1 (delete-minimize) can only SHRINK a successful comment to a sub-sequence; it cannot
escape the load-bearing semantic phrase that does the flipping (e.g. ``closed-source``,
``no requirement to publish source``). Deleting it breaks the attack; keeping it leaves a
readable license assertion. So deletion alone never reaches a state that "still flips the
model but no longer reads as a license claim".

M4 adds three more edit operators and searches with simulated annealing under a hard
constraint plus a soft objective:

  hard constraint : margin(c') < tau          (the attack must still flip the model)
  soft objective  : minimize S(c')            (semantic residue — make it un-deniable-free)

Operators on the comment token sequence (state):
  * delete     — drop a token (M1's move).
  * substitute — replace a token with a gradient-proposed alternative (GCG top-k).
  * insert     — insert a gradient-proposed token (lets the search route AROUND a phrase
                 that breaks when a token is deleted).
  * reorder    — swap two adjacent tokens / move one (breaks fixed detector-matchable
                 phrases without changing the bag of tokens).

Semantic-residue score S = w1*S1 + w2*S2 + w3*S3 (lower = better washed):
  S1 = number of competing licenses NAMED (eval.conflict.licenses_named_in_comment).
  S2 = number of license-semantic keyword hits (source/proprietary/closed/redistribute/
       patent/license/...): a regex bag that catches the residue S1 misses.
  S3 = coherence-as-a-license-claim: we re-use the VICTIM model with a Yes/No probe
       ("Does the following comment assert or describe software license terms?") and read
       the probability of "Yes". Gibberish (GCG-like) -> ~0; "closed-source" -> high.
       S3 is the expensive term, so it is computed lazily, only for feasible neighbours.

The result we want is the bottom-left corner: margin < tau (still flips) AND S ~ 0 (reads
as gibberish, asserts no license), i.e. a comment that satisfies the threat model's
"cannot change the original license's legal meaning" the way GCG output already does, but
reached by washing a semantic seed instead.

M3 (2026-08-20) — the transfer-oriented variant
-----------------------------------------------
Measuring M4-washed comments against six frontier black-box models showed they barely
transfer (identification 0-6%, clause 0.9-23%) while the UN-washed seeds transfer at
19-94% on the same rows. Washing, not seed quality, is what destroys transferability.
M3 is M4 with three changes, each targeting one cause:

  * ``use_reorder=False`` — the reorder operator scrambles token order, the most
    source-model-specific and least fluent edit available. Dropping it is what makes an
    M3 run an M3 run.
  * ``tau`` below 0 — M4 accepts a comment that clears the decision boundary by a hair,
    leaving no confidence slack for a different model. Requiring ``margin < -delta``
    keeps only confident flips. Rather than one search per delta, ``pareto_buckets``
    archives the most-negative margin per residue bucket, so a single run yields the
    whole (S*, delta) grid.
  * ``eot_segs`` — a comment tuned against one prompt rendering overfits that rendering.
    Scoring under several (see ``data.prompts``, which follows ClauseBench's basic /
    few-shot / contextual / full strategies) pushes the comment to flip all of them.

NOTE ON THE NAME: an earlier design doc used "M3" for an abandoned LLM-rewrite washer that
was never built or run. M3 here means M4-minus-reorder, per the author's 2026-08-20 call.

Every M3 knob defaults to the M4 behaviour, so existing M4 results stay reproducible.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

import torch

from ..model.base import DetectorModel, PromptSegments
from ..eval.conflict import licenses_named_in_comment
from .scoring import objective_from_onehot, score_labels_from_embeds

# --- S2: license-semantic keyword bag (catches residue S1's name-matching misses) ---
_SEMANTIC_KEYWORDS = re.compile(
    r"\b("
    r"licen[cs]\w*|copyright|copyleft|proprietary|closed[- ]?source|open[- ]?source|"
    r"redistribut\w*|distribut\w*|sublicen\w*|relicen\w*|patent|trademark|warrant\w*|"
    r"liabilit\w*|disclos\w*|source\s+code|permission|permit\w*|grant\w*|"
    r"public\s+domain|royalty|gpl|lgpl|agpl|mpl|epl|bsd|mit|apache|cddl|"
    r"obligation|comply|compliance|notice|terms\s+and\s+conditions"
    r")\b",
    re.IGNORECASE,
)

# A compact Yes/No probe for S3, scored on the victim model itself.
_S3_PROBE_PREFIX = (
    "Comment:\n\"\"\"\n"
)
_S3_PROBE_SUFFIX = (
    "\n\"\"\"\n"
    "Does the comment above assert, grant, or describe software license terms "
    "(permissions, obligations, or a license name)? Answer Yes or No.\n"
)
_S3_LABELS = ["Yes", "No"]


@dataclass
class RewriteSearchConfig:
    tau: float = 0.0                 # attack stays valid while margin < tau
    n_steps: int = 300               # annealing steps
    top_k: int = 64                  # gradient candidates per position for sub/insert
    t0: float = 1.0                  # initial temperature
    t_end: float = 0.02              # final temperature (geometric schedule)
    w1: float = 3.0                  # weight on S1 (named competing licenses)
    w2: float = 1.0                  # weight on S2 (semantic keywords)
    w3: float = 2.0                  # weight on S3 (coherence-as-license-claim)
    use_s3: bool = True              # compute S3 on the victim model (expensive)
    max_len: int = 64                # cap comment length (insert can grow it)
    min_len: int = 1
    seed: int = 0
    p_delete: float = 0.30
    p_substitute: float = 0.30
    p_insert: float = 0.20
    p_reorder: float = 0.20
    # early stop: once the attack still flips AND residue is at/below target, stop after
    # `patience` further steps with no improvement (saves the empty tail of the schedule).
    early_stop: bool = True          # set False to always run the full schedule (Pareto runs)
    early_stop_S: float = 0.5        # residue target; <= this counts as "washed enough"
    patience: int = 8                # consecutive no-improvement steps before stopping

    # --- M3 additions (all default to M4 behaviour; see module docstring) ---
    use_reorder: bool = True         # M3 = M4 minus the reorder operator
    # Pareto archive: bucket edges on the semantic residue S. When set, every FEASIBLE
    # candidate is filed under the smallest bucket >= its S, keeping the most-negative
    # margin seen in that bucket. One run then yields the whole (S*, delta) grid without
    # a separate search per tau. Also disable `early_stop` so the frontier gets filled.
    pareto_buckets: tuple[float, ...] | None = None
    # Prompt-EOT: when rewrite_search gets `eot_segs`, the comment is also scored under those
    # alternative prompt renderings.
    #   "soft"   (default) hard constraint stays on the PRIMARY segment, and the other
    #            variants enter the objective as a hinge penalty. No seed is lost, and the
    #            all-variant-robust subset is recoverable post-hoc via `n_eot_flip`.
    #   "strict" hard constraint is max_i margin_i < tau (a seed that does not already flip
    #            every variant is returned unchanged, success=False).
    eot_mode: str = "soft"
    w_eot: float = 1.0               # weight on the EOT hinge penalty (soft mode)
    # Margin-seeking term. The hard constraint only demands margin < tau, so the search
    # has no reason to go FURTHER below it -- and measurement shows it does not: archived
    # states at low residue sit at margin ~-0.55 while high-residue ones reach ~-1.24, i.e.
    # washing spends the confidence margin that a different model would need. Setting
    # w_margin > 0 adds a hinge that actively pushes margin below -margin_target, which
    # tests whether that coupling is fundamental or just an objective that never asked.
    # 0.0 (default) reproduces M4/M3 as first run.
    w_margin: float = 0.0
    margin_target: float = 1.0       # push margin below -margin_target


@dataclass
class RewriteSearchResult:
    true_label: str
    start_text: str
    washed_text: str
    washed_ids: torch.Tensor
    start_margin: float
    final_margin: float
    start_S: float
    final_S: float
    s1: int
    s2: int
    s3: float
    success: bool          # still flips (margin < tau) at the end
    steps_used: int
    trace: list = field(default_factory=list)
    # --- M3 additions ---
    # bucket label -> {text, margin, margins_eot, n_eot_flip, S, s1, s2, s3, step}
    pareto: dict = field(default_factory=dict)
    margins_eot: list = field(default_factory=list)   # per-variant margin of the returned comment
    n_eot_flip: int = 0                               # how many variants it flips (incl. primary)


def _residue_s1(text: str, header_license: str | None) -> int:
    named = licenses_named_in_comment(text)
    if header_license is not None:
        named = {l for l in named if l != header_license}
    return len(named)


def _residue_s2(text: str) -> int:
    return len(_SEMANTIC_KEYWORDS.findall(text))


def _residue_s3(
    model: DetectorModel,
    text: str,
) -> float:
    """P('Yes' | the comment asserts/describes license terms), via the victim model.

    Built as its own tiny prompt (probe prefix + comment + probe suffix), scored with
    the same teacher-forced label readout the attack uses. Returns a probability in
    [0, 1]; high = still reads as a license claim, low = gibberish/neutral.
    """
    seg = model.build_segments(_S3_PROBE_PREFIX, _S3_PROBE_SUFFIX, "")
    empty = model.encode(text, add_special=False).to(model.device)
    sc = score_labels_from_embeds(model, seg, empty, _S3_LABELS)
    # logprobs are length-normalized mean per-token; turn the 2-way into P(Yes)
    p = torch.softmax(sc.logprobs, dim=0)
    return float(p[_S3_LABELS.index("Yes")])


def _cheap_residue(text: str, header_license: str | None, cfg: RewriteSearchConfig) -> tuple[float, int, int]:
    """The free (regex-only) part of S: w1*S1 + w2*S2. No model call."""
    s1 = _residue_s1(text, header_license)
    s2 = _residue_s2(text)
    return cfg.w1 * s1 + cfg.w2 * s2, s1, s2


def _semantic_residue(
    model: DetectorModel,
    ids: torch.Tensor,
    header_license: str | None,
    cfg: RewriteSearchConfig,
) -> tuple[float, int, int, float]:
    text = model.decode(ids)
    cheap, s1, s2 = _cheap_residue(text, header_license, cfg)
    s3 = _residue_s3(model, text) if cfg.use_s3 else 0.0
    S = cheap + cfg.w3 * s3
    return S, s1, s2, s3


def _margin_of(
    model: DetectorModel,
    seg: PromptSegments,
    ids: torch.Tensor,
    labels: list[str],
    true_label: str,
    label_token_ids: list[torch.Tensor],
    calib=None,
) -> float:
    if len(ids) == 0:
        ids = model.encode("", add_special=False).to(ids.device)
    sc = score_labels_from_embeds(model, seg, ids, labels, label_token_ids, calib)
    return float(sc.margin(true_label))


def _margins_over_segs(
    model: DetectorModel,
    segs: list[PromptSegments],
    ids: torch.Tensor,
    labels: list[str],
    true_label: str,
    label_token_ids: list[torch.Tensor],
    calib=None,
) -> list[float]:
    """Margin under each alternative prompt rendering (prompt-EOT).

    Cost is one forward per segment, so callers evaluate the PRIMARY segment first and
    only pay for these once the primary constraint has already passed.
    """
    return [_margin_of(model, s, ids, labels, true_label, label_token_ids, calib) for s in segs]


def _eot_penalty(margins: list[float], tau: float) -> float:
    """Hinge on the alternative renderings: 0 once they all flip, else how far short."""
    return sum(max(0.0, m - tau) for m in margins)


def _margin_penalty(margin: float, cfg: "RewriteSearchConfig") -> float:
    """Hinge that rewards margin below -margin_target; 0 once the surplus is reached."""
    if cfg.w_margin <= 0:
        return 0.0
    return cfg.w_margin * max(0.0, margin + cfg.margin_target)


def _bucket_of(S: float, buckets: tuple[float, ...]) -> str:
    """Smallest bucket edge >= S, as a label; 'inf' when S exceeds every edge."""
    for b in sorted(buckets):
        if S <= b + 1e-9:
            return f"{b:g}"
    return "inf"


def _archive(
    pareto: dict,
    buckets: tuple[float, ...],
    S: float,
    margin: float,
    text: str,
    stats: tuple[int, int, float],
    margins_eot: list[float],
    tau: float,
    step: int,
) -> None:
    """File a FEASIBLE candidate under its residue bucket, keeping the most negative
    margin seen there. Called for candidates the Metropolis rule later rejects too —
    those are still valid attacks, and the frontier is about what was reachable, not
    about what the chain happened to walk through."""
    key = _bucket_of(S, buckets)
    prev = pareto.get(key)
    if prev is not None and prev["margin"] <= margin:
        return
    pareto[key] = {
        "text": text, "margin": margin, "S": S,
        "s1": stats[0], "s2": stats[1], "s3": stats[2],
        "margins_eot": list(margins_eot),
        "n_eot_flip": 1 + sum(1 for m in margins_eot if m < tau),
        "step": step,
    }


def _grad_topk(
    model: DetectorModel,
    seg: PromptSegments,
    ids: torch.Tensor,
    labels: list[str],
    true_label: str,
    label_token_ids: list[torch.Tensor],
    safe_mask: torch.Tensor,
    top_k: int,
    calib=None,
) -> torch.Tensor:
    """Per-position top-k loss-reducing substitution candidates (GCG-style), restricted
    to the comment-safe vocab. Returns [m, k] token ids. Used to PROPOSE substitute /
    insert tokens that are likely to keep the attack alive (so washing edits don't just
    randomly break the flip)."""
    V = model.vocab_size
    E = model.embedding_matrix
    m = len(ids)
    onehot = torch.zeros(m, V, device=model.device, dtype=E.dtype)
    onehot[torch.arange(m), ids.to(model.device)] = 1.0
    onehot.requires_grad_(True)
    loss = objective_from_onehot(model, seg, onehot, labels, true_label,
                                 label_token_ids, "margin", calib=calib)
    grad = torch.autograd.grad(loss, onehot)[0]                 # [m, V]
    grad = grad.masked_fill(~safe_mask.unsqueeze(0), float("inf"))
    k = min(top_k, int(safe_mask.sum()))
    return (-grad).topk(k, dim=1).indices                       # [m, k]


def _propose(
    ids: torch.Tensor,
    op: str,
    topk: torch.Tensor | None,
    safe_ids: torch.Tensor,
    cfg: RewriteSearchConfig,
    gen: torch.Generator,
) -> torch.Tensor:
    """Apply one edit operator, returning a NEW id sequence (CPU long tensor)."""
    ids = ids.clone()
    m = len(ids)
    if op == "delete" and m > cfg.min_len:
        pos = int(torch.randint(0, m, (1,), generator=gen))
        return torch.cat([ids[:pos], ids[pos + 1:]])
    if op == "substitute" and m > 0 and topk is not None:
        pos = int(torch.randint(0, m, (1,), generator=gen))
        kcol = int(torch.randint(0, topk.shape[1], (1,), generator=gen))
        ids[pos] = topk[pos, kcol].to(ids.device)
        return ids
    if op == "insert" and m < cfg.max_len:
        pos = int(torch.randint(0, m + 1, (1,), generator=gen))
        # propose from a random position's top-k if available, else a random safe token
        if topk is not None and m > 0:
            src = int(torch.randint(0, m, (1,), generator=gen))
            kcol = int(torch.randint(0, topk.shape[1], (1,), generator=gen))
            tok = topk[src, kcol].to(ids.device).view(1)
        else:
            tok = safe_ids[torch.randint(0, len(safe_ids), (1,), generator=gen)]
        return torch.cat([ids[:pos], tok, ids[pos:]])
    if op == "reorder" and m > 1:
        pos = int(torch.randint(0, m - 1, (1,), generator=gen))
        ids[pos], ids[pos + 1] = ids[pos + 1].clone(), ids[pos].clone()
        return ids
    return ids  # no-op (operator not applicable at this length)


def rewrite_search(
    model: DetectorModel,
    seg: PromptSegments,
    comment_ids: torch.Tensor,
    labels: list[str],
    true_label: str,
    cfg: RewriteSearchConfig,
    header_license: str | None = None,
    eot_segs: list[PromptSegments] | None = None,
    calib=None,
) -> RewriteSearchResult:
    """Simulated-annealing multi-operator washer.

    Starts from an ALREADY-SUCCESSFUL comment (margin < tau). Each step proposes one
    edit (delete/substitute/insert[/reorder]); a proposal is REJECTED outright if it
    breaks the attack (margin >= tau). Among attack-preserving proposals, we accept by
    the Metropolis rule on the objective: always accept if it drops, else accept with
    prob exp(-dObj / T). Temperature cools geometrically so early steps explore (escape
    the load-bearing-phrase local optimum) and late steps refine.

    The objective is the semantic residue S, plus — when `eot_segs` is given in "soft"
    mode — ``w_eot`` times a hinge on the alternative prompt renderings, so the search
    actively drives those below tau as well without ever discarding a seed that only
    flips the primary rendering.
    """
    device = model.device
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    label_token_ids = [model.encode(l, add_special=False) for l in labels]
    safe_mask = model.comment_safe_mask().to(device)
    safe_ids = torch.nonzero(safe_mask, as_tuple=False).squeeze(-1).cpu()

    eot_segs = list(eot_segs or [])
    strict_eot = bool(eot_segs) and cfg.eot_mode == "strict"

    cur = comment_ids.detach().cpu().clone()
    start_text = model.decode(cur)
    start_margin = _margin_of(model, seg, cur.to(device), labels, true_label, label_token_ids, calib)
    cur_eot = (_margins_over_segs(model, eot_segs, cur.to(device), labels, true_label,
                                  label_token_ids, calib) if eot_segs else [])

    # If the seed isn't actually a successful attack, return it unchanged (nothing to wash).
    # In strict-EOT mode the bar is higher: it must already flip every prompt rendering.
    infeasible = start_margin >= cfg.tau or (
        strict_eot and any(m >= cfg.tau for m in cur_eot))
    if infeasible:
        S0, s1, s2, s3 = _semantic_residue(model, cur, header_license, cfg)
        return RewriteSearchResult(
            true_label=true_label, start_text=start_text, washed_text=start_text,
            washed_ids=cur, start_margin=start_margin, final_margin=start_margin,
            start_S=S0, final_S=S0, s1=s1, s2=s2, s3=s3, success=False, steps_used=0,
            margins_eot=cur_eot,
            n_eot_flip=int(start_margin < cfg.tau) + sum(1 for m in cur_eot if m < cfg.tau),
        )

    cur_S, cur_s1, cur_s2, cur_s3 = _semantic_residue(model, cur, header_license, cfg)
    start_S = cur_S
    cur_obj = (cur_S + (cfg.w_eot * _eot_penalty(cur_eot, cfg.tau) if eot_segs else 0.0)
               + _margin_penalty(start_margin, cfg))
    best = cur.clone()
    best_S = cur_S
    best_obj = cur_obj
    best_stats = (cur_s1, cur_s2, cur_s3)
    best_eot = list(cur_eot)

    pareto: dict = {}
    if cfg.pareto_buckets:
        _archive(pareto, cfg.pareto_buckets, cur_S, start_margin, start_text,
                 (cur_s1, cur_s2, cur_s3), cur_eot, cfg.tau, -1)

    # Keeping the 4-operator list byte-identical when reorder is on preserves the RNG
    # draw sequence, so M4 runs reproduce exactly.
    if cfg.use_reorder:
        ops = ["delete", "substitute", "insert", "reorder"]
        probs = [cfg.p_delete, cfg.p_substitute, cfg.p_insert, cfg.p_reorder]
    else:
        ops = ["delete", "substitute", "insert"]
        probs = [cfg.p_delete, cfg.p_substitute, cfg.p_insert]
    trace = []
    # refresh gradient top-k periodically (it depends on current ids); cheap to do lazily
    topk = None
    topk_age = 0
    no_improve = 0   # consecutive steps without best-objective improvement (for early stop)

    for step in range(cfg.n_steps):
        T = cfg.t0 * (cfg.t_end / cfg.t0) ** (step / max(1, cfg.n_steps - 1))

        if os.environ.get("WASH_STEP_LOG") and step % 10 == 0:
            print(f"    [wash step {step}/{cfg.n_steps}] len={len(cur)} S={cur_S:.2f}", flush=True)

        # (re)compute gradient proposals occasionally for sub/insert moves. Gradients only
        # PROPOSE candidates, so they stay on the primary segment even under EOT — paying
        # k backward passes here would buy nothing the hard constraint doesn't already give.
        if topk is None or topk_age >= 10:
            if len(cur) > 0:
                topk = _grad_topk(model, seg, cur.to(device), labels, true_label,
                                  label_token_ids, safe_mask, cfg.top_k, calib).cpu()
            topk_age = 0
        topk_age += 1

        op = ops[int(torch.multinomial(torch.tensor(probs), 1, generator=gen))]
        cand = _propose(cur, op, topk, safe_ids, cfg, gen)
        if len(cand) == 0 or torch.equal(cand, cur):
            continue

        # hard constraint: must still flip the model. Primary segment first — most
        # proposals die here, and that keeps the EOT forwards off the hot path.
        cand_margin = _margin_of(model, seg, cand.to(device), labels, true_label, label_token_ids, calib)
        if cand_margin >= cfg.tau:
            continue  # reject: breaks the attack
        cand_eot = (_margins_over_segs(model, eot_segs, cand.to(device), labels, true_label,
                                       label_token_ids, calib) if eot_segs else [])
        if strict_eot and any(m >= cfg.tau for m in cand_eot):
            continue  # reject: breaks the attack under some prompt rendering

        # soft objective: Metropolis on semantic residue S = cheap(S1,S2) + w3*S3.
        # S3 needs a model forward, so gate it: compute the FREE cheap residue first and
        # only pay for S3 when the candidate is not already worse on the cheap axis than
        # the current state's UPPER bound (cur cheap part). This skips the model call for
        # the many proposals that regress on named-license / keyword count anyway.
        cand_text = model.decode(cand)
        cheap, c_s1, c_s2 = _cheap_residue(cand_text, header_license, cfg)
        cur_cheap = cfg.w1 * cur_s1 + cfg.w2 * cur_s2
        if cfg.use_s3 and cheap <= cur_cheap + cfg.w3:  # within one S3-unit of plausible
            c_s3 = _residue_s3(model, cand_text)
        else:
            c_s3 = 0.0 if not cfg.use_s3 else 1.0  # pessimistic stand-in when skipped
        cand_S = cheap + cfg.w3 * c_s3
        cand_obj = (cand_S + (cfg.w_eot * _eot_penalty(cand_eot, cfg.tau) if eot_segs else 0.0)
                    + _margin_penalty(cand_margin, cfg))

        # The candidate is a valid attack whether or not Metropolis keeps it, so it belongs
        # on the frontier either way. Note S3 may be the pessimistic stand-in here, which
        # only ever files a candidate in a HIGHER bucket than it deserves.
        if cfg.pareto_buckets:
            _archive(pareto, cfg.pareto_buckets, cand_S, cand_margin, cand_text,
                     (c_s1, c_s2, c_s3), cand_eot, cfg.tau, step)

        dS = cand_obj - cur_obj
        if dS <= 0 or torch.rand(1, generator=gen).item() < math.exp(-dS / max(1e-6, T)):
            cur, cur_S, cur_s1, cur_s2, cur_s3 = cand, cand_S, c_s1, c_s2, c_s3
            cur_obj, cur_eot = cand_obj, cand_eot
            topk = None  # ids changed -> stale gradient proposals
            if cur_obj < best_obj - 1e-9:
                best, best_S, best_obj = cur.clone(), cur_S, cur_obj
                best_stats, best_eot = (c_s1, c_s2, c_s3), list(cand_eot)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        trace.append((step, op, cur_S, cand_margin))

        # Early stop: the best comment already flips (it was accepted under margin<tau) and
        # its residue is at/below target, with no improvement for `patience` steps. The rest
        # of the schedule would only shuffle an already-washed comment, so stop. Pareto runs
        # turn this off — the tail of the schedule is where the frontier gets filled in.
        if cfg.early_stop and best_obj <= cfg.early_stop_S and no_improve >= cfg.patience:
            if os.environ.get("WASH_STEP_LOG"):
                print(f"    [early stop @ step {step}] best_S={best_S:.2f}", flush=True)
            break

    final_margin = _margin_of(model, seg, best.to(device), labels, true_label, label_token_ids, calib)
    return RewriteSearchResult(
        true_label=true_label, start_text=start_text, washed_text=model.decode(best),
        washed_ids=best, start_margin=start_margin, final_margin=final_margin,
        start_S=start_S, final_S=best_S, s1=best_stats[0], s2=best_stats[1], s3=best_stats[2],
        success=final_margin < cfg.tau, steps_used=len(trace),
        pareto=pareto, margins_eot=best_eot,
        n_eot_flip=int(final_margin < cfg.tau) + sum(1 for m in best_eot if m < cfg.tau),
    )
