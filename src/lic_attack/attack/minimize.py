"""Gradient-guided greedy backward elimination (deletion-only minimization).

Given an ALREADY-SUCCESSFUL adversarial comment (margin < tau), shrink it to a
*minimal sufficient subset*: a subsequence such that deleting any further token
would break the attack. This turns a long, human-readable-ish comment into a short
token soup while preserving the attack — the short survivors can then be hand-
screened for ones that carry no legal semantics.

Design (route A), faithful to the experiment's purity:
  * The ONLY acceptance criterion is margin(c) < tau, checked by REALLY deleting the
    token and re-running the forward pass (delete != zero-out the embedding).
  * Gradient saliency (input x gradient on the comment embeddings) is used ONLY to
    ORDER which token to try deleting first. It never enters the objective, never
    biases toward/away from any token class (no legal-keyword bias). It only affects
    efficiency — try the least-salient token first, it is the most likely to be
    safely removable, so we converge to a smaller subset faster.

The objective being optimized is pure L0: minimize the number of retained tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..model.base import DetectorModel, PromptSegments
from .scoring import objective_from_onehot, score_labels_from_embeds


@dataclass
class MinimizeConfig:
    tau: float = 0.0          # attack stays valid while margin < tau
    use_gradient_order: bool = True   # rank deletions by saliency (efficiency only)
    seed: int = 0             # randomized order tie-break / fallback when grad off
    max_passes: int = 3       # extra full sweeps after first, to catch newly-removable


@dataclass
class MinimizeResult:
    true_label: str
    start_len: int
    final_len: int
    kept_ids: torch.Tensor          # the minimal sufficient subset (token ids)
    kept_text: str
    final_margin: float
    success: bool                   # margin < tau at the end (attack preserved)
    order_used: str                 # "gradient" | "random"


def _saliency(
    model: DetectorModel,
    seg: PromptSegments,
    comment_ids: torch.Tensor,
    labels: list[str],
    true_label: str,
    label_token_ids: list[torch.Tensor],
    objective: str,
) -> torch.Tensor:
    """Per-token importance for MAINTAINING the attack, via input x gradient.

    We take d(margin_loss)/d(comment_emb) and contract with the embedding itself.
    A token whose removal would most RAISE the margin (undo the attack) has high
    |saliency|; we will try to delete LOW-saliency tokens first.
    Returns a 1-D tensor [m] aligned with comment_ids. (Ordering only — not a loss.)
    """
    E = model.embedding_matrix
    onehot = torch.zeros(len(comment_ids), model.vocab_size,
                         device=model.device, dtype=E.dtype)
    onehot[torch.arange(len(comment_ids)), comment_ids.to(model.device)] = 1.0
    onehot.requires_grad_(True)
    loss = objective_from_onehot(model, seg, onehot, labels, true_label,
                                 label_token_ids, objective)
    grad_oh = torch.autograd.grad(loss, onehot)[0]          # [m, V]
    # gradient wrt this token's embedding, contracted with the embedding -> scalar/token
    comment_emb = (onehot @ E).detach()                     # [m, H]
    grad_emb = grad_oh @ E                                  # [m, H] (chain via E)
    sal = (grad_emb * comment_emb).sum(dim=1)               # [m]
    return sal.detach()


def minimize_comment(
    model: DetectorModel,
    seg: PromptSegments,
    comment_ids: torch.Tensor,
    labels: list[str],
    true_label: str,
    cfg: MinimizeConfig,
    objective: str = "margin",
) -> MinimizeResult:
    """Greedy backward elimination to a minimal sufficient subset (deletion-only)."""
    device = comment_ids.device
    label_token_ids = [model.encode(l, add_special=False) for l in labels]
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    def margin_of(ids: torch.Tensor) -> float:
        if len(ids) == 0:
            # empty comment: fall back to clean prompt margin
            ids = model.encode("", add_special=False).to(device)
        sc = score_labels_from_embeds(model, seg, ids, labels, label_token_ids)
        return float(sc.margin(true_label))

    kept = comment_ids.clone()
    start_len = len(kept)
    order_used = "gradient" if cfg.use_gradient_order else "random"

    # sanity: the starting comment must actually be a successful attack
    if margin_of(kept) >= cfg.tau:
        return MinimizeResult(
            true_label=true_label, start_len=start_len, final_len=start_len,
            kept_ids=kept, kept_text=model.decode(kept), final_margin=margin_of(kept),
            success=False, order_used=order_used,
        )

    changed = True
    passes = 0
    while changed and passes < cfg.max_passes:
        changed = False
        passes += 1

        # rank current positions: least-salient first (most likely safely removable)
        if cfg.use_gradient_order and len(kept) > 0:
            sal = _saliency(model, seg, kept, labels, true_label, label_token_ids, objective)
            order = torch.argsort(sal.abs(), descending=False).tolist()
        else:
            order = torch.randperm(len(kept), generator=gen).tolist()

        # try deletions in that order; positions shift after each accepted delete, so
        # we operate on a mask over the CURRENT kept sequence and rebuild once per pass
        removed = set()
        for pos in order:
            if pos in removed:
                continue
            trial_mask = torch.ones(len(kept), dtype=torch.bool)
            trial_mask[pos] = False
            for r in removed:
                trial_mask[r] = False
            trial = kept[trial_mask]
            if margin_of(trial) < cfg.tau:    # attack still valid -> accept deletion
                removed.add(pos)
                changed = True
        if removed:
            keep_mask = torch.ones(len(kept), dtype=torch.bool)
            for r in removed:
                keep_mask[r] = False
            kept = kept[keep_mask]

    final_margin = margin_of(kept)
    return MinimizeResult(
        true_label=true_label, start_len=start_len, final_len=len(kept),
        kept_ids=kept, kept_text=model.decode(kept), final_margin=final_margin,
        success=final_margin < cfg.tau, order_used=order_used,
    )
