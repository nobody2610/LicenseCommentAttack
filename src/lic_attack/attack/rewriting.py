"""Washing methods (洗白) for a successful attack comment.

A comment that flips the identifier is only half the attack: an overt comment like
"This file is relicensed under MIT" is trivially flagged by a human reviewer or a
header-vs-comment conflict detector. WASHING reduces a successful comment to a form that
still flips the model (effectiveness) AND names no competing license token (evasion) AND
reads plausibly (deniability). This module provides the washers; effectiveness must be
re-checked on the model (GPU) after washing, evasion is checked here via
eval.conflict.licenses_named_in_comment.

Washers (increasing sophistication):
  M0 none            : identity — the original successful comment (baseline).
  M1 delete-minimize : pure token deletion (see attack.minimize) — shrinks, does not
                       rewrite, so a named license token survives. Implemented elsewhere;
                       referenced here for the comparison.
  M2 keyword-strip   : deterministically remove competing-license names / SPDX tokens,
                       keeping the surrounding narrative. Directly targets evasion. This
                       module implements M2.
  M3 paraphrase      : LLM-rewrite to fluent, keyword-free prose preserving intent
                       (needs a generator call; left as a hook).

M2 is deterministic and model-free, so it runs in seconds; effectiveness (does the
washed comment still flip the model?) is verified separately on the detector.
"""
from __future__ import annotations

import re

from ..eval.conflict import _NAME_TO_SPDX, _SPDX_LINE, licenses_named_in_comment


def wash_none(comment: str) -> str:
    """M0 baseline: the original comment, unchanged."""
    return comment


def wash_keyword_strip(comment: str, header_license: str | None = None) -> str:
    """M2: remove competing-license evidence a token comparator keys on, while leaving
    the misleading narrative intact.

    Removes (a) whole SPDX-License-Identifier lines/clauses and (b) explicit license
    names (MIT, "Apache License", GPL, ...). If header_license is given, an occurrence of
    the file's OWN license is kept (re-stating the true license is not a competing token
    and not what we strip). The goal is a comment that still asserts "this isn't really
    governed by the header" without ever naming what it supposedly is.
    """
    out = _SPDX_LINE.sub("", comment)  # drop forged SPDX lines entirely
    for pat, spdx in _NAME_TO_SPDX.items():
        if header_license is not None and spdx == header_license:
            continue  # keep the file's own license name; only competing names are evidence
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    # tidy whitespace/punctuation left by removals
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([.,;:])", r"\1", out)
    out = re.sub(r"([.,;:])\1+", r"\1", out)
    return out.strip(" \t,;:.-")


# registry so the runner / report can iterate washers uniformly.
# NOTE: M4 (multi-operator simulated-annealing search) is NOT a pure str->str function
# like M0/M2 — it needs the victim model + prompt segments to enforce the still-flips
# constraint and to score S3. It lives in attack.wash_search.wash_search and is invoked
# directly by the runner, not through this str-only registry.
WASHERS = {
    "none": wash_none,
    "keyword_strip": wash_keyword_strip,
}


def evades_conflict(comment: str, header_license: str) -> bool:
    """True if the (washed) comment names no license competing with the header — i.e. a
    header-vs-comment token comparator finds nothing to flag."""
    named = licenses_named_in_comment(comment)
    return not any(l != header_license for l in named)
