"""Clause-level semantic attack coordinator.

Where the license attack flips the model's *license identification* (GPL -> MIT),
this attack flips the model's understanding of a *binary clause* — e.g. making it
answer "No" to "Does this license permit commercial use?" when the truth is "Yes",
WITHOUT necessarily changing the perceived license name. That is a stealthier and
more directly harmful failure: the model still says "this is GPL" but now believes
GPL allows closed-source linking.

This module is a thin coordinator. The heavy lifting (GCG search, deletion
minimization, scoring, margin) is fully reused from gcg.py / minimize.py /
scoring.py — the only change is the label set ({Yes, No}) and the ground-truth
(the clause answer) handed to those parameterized functions.

Untargeted by design: in a binary task, "flip away from the truth" is identical to
"flip to the single other answer", so margin = logp[truth] - logp[other] and we
push it below tau. We only attack clauses the model gets RIGHT and CONFIDENT on a
clean prompt (require_clean_correct + clean_margin_min), otherwise a "flip" is
meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.licenses import ClauseProbe, Sample
from ..data.handcrafted import clause_handcrafted_for_style
from ..model.base import DetectorModel
from .gcg import AttackConfig, run_gcg
from .minimize import MinimizeConfig, minimize_comment
from .scoring import score_labels_from_embeds

YES_NO = ["Yes", "No"]


@dataclass
class ClauseCandidate:
    name: str            # candidate id (handcrafted key or "gcg")
    category: str
    comment_ids: torch.Tensor
    argmax: str          # "Yes"/"No" after this comment
    margin: float        # logp[truth] - logp[other]; <tau means flipped
    success: bool
    comment_text: str


@dataclass
class ClauseAttackResult:
    license: str
    probe_key: str
    truth: str                       # ground-truth answer "Yes"/"No"
    clean_answer: str
    clean_margin: float
    skipped: str | None              # reason if skipped (clean wrong / not confident)
    candidates: list[ClauseCandidate]


def attack_clause(
    model: DetectorModel,
    sample: Sample,
    probe: ClauseProbe,
    use_gcg: bool,
    attack_cfg: AttackConfig,
    min_cfg: MinimizeConfig,
    comment_style: str = "line",
    success_tau: float = 0.0,
    require_clean_correct: bool = True,
    clean_margin_min: float = 0.0,
    do_minimize: bool = True,
    llm_comments: list[str] | None = None,
):
    """Attack one (sample, clause) pair. Returns (ClauseAttackResult, minimize_records).

    minimize_records is a list parallel to the successful candidates, each a dict
    ready for JSON (start_len/final_len/kept_text/...), or None for non-minimized.
    """
    truth = probe.truth.get(sample.license_id)
    seg = model.build_segments(sample.source_before, sample.source_after, probe.question)
    yn_ids = [model.encode(t, add_special=False) for t in YES_NO]
    empty = model.encode("", add_special=False)

    if truth is None:
        return ClauseAttackResult(sample.license_id, probe.key, "n/a", "n/a", 0.0,
                                  "no ground truth for this license", []), []

    clean = score_labels_from_embeds(model, seg, empty, YES_NO, yn_ids)
    clean_ans = clean.argmax_label()
    clean_margin = float(clean.margin(truth))

    # gate: only attack clauses the model already gets right and is confident about
    if require_clean_correct and (clean_ans != truth or clean_margin < clean_margin_min):
        reason = ("clean answer wrong" if clean_ans != truth
                  else f"clean margin {clean_margin:.3f} < {clean_margin_min}")
        return ClauseAttackResult(sample.license_id, probe.key, truth, clean_ans,
                                  clean_margin, reason, []), []

    # ---- collect candidates: clause-specific handcrafted (+ optional LLM, + optional GCG) ----
    candidates: list[tuple[str, str, torch.Tensor]] = []
    for hc in clause_handcrafted_for_style(comment_style):
        candidates.append((hc.key, hc.category, model.encode(hc.text, add_special=False).to(model.device)))
    # LLM-generated comments for this clause (offline cache); each is its own candidate so
    # the per-candidate minimize (below) runs on the successful ones, same as handcrafted.
    for i, text in enumerate(llm_comments or []):
        candidates.append((f"llm_{i}", "llm", model.encode(text, add_special=False).to(model.device)))
    if use_gcg:
        res = run_gcg(model, seg, YES_NO, truth, attack_cfg)
        candidates.append(("gcg", "gcg", res.final_comment_ids))

    out_cands: list[ClauseCandidate] = []
    min_records: list = []
    for name, category, comment_ids in candidates:
        sc = score_labels_from_embeds(model, seg, comment_ids, YES_NO, yn_ids)
        margin = float(sc.margin(truth))
        success = margin < success_tau
        out_cands.append(ClauseCandidate(
            name=name, category=category, comment_ids=comment_ids,
            argmax=sc.argmax_label(), margin=margin, success=success,
            comment_text=model.decode(comment_ids),
        ))
        if success and do_minimize:
            mr = minimize_comment(model, seg, comment_ids, YES_NO, truth,
                                  min_cfg, attack_cfg.objective)
            min_records.append({
                "candidate": name, "category": category,
                "start_len": mr.start_len, "final_len": mr.final_len,
                "final_margin": mr.final_margin, "success": mr.success,
                "order": mr.order_used, "kept_text": mr.kept_text,
                "kept_ids": mr.kept_ids.tolist(),
            })
        else:
            min_records.append(None)

    return ClauseAttackResult(sample.license_id, probe.key, truth, clean_ans,
                              clean_margin, None, out_cands), min_records
