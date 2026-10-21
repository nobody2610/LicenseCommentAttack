"""Prompt builders for the license-ID task and clause probes.

Two identification prompt styles (the SCORING is the same for both — teacher-forced
logprob ranking over a candidate SPDX set, see attack/scoring.py — so neither relies
on free-form generation):

  * open_ended : "What license governs this file? Answer with an SPDX identifier."
      No options listed → closer to a real deployment query. Main-table default.
  * closed_set : lists the candidate SPDX ids in the prompt. For ablation.
"""
from __future__ import annotations


def open_ended_question() -> str:
    """Realistic, no-options query. Answer is read via forced-choice logprob ranking
    over the candidate SPDX set, not by parsing free generation."""
    return "What license governs this file? Answer with an SPDX identifier."


def closed_set_question(labels: list[str]) -> str:
    """The identification instruction that lists the candidate set (ablation)."""
    opts = ", ".join(labels)
    return (
        f"Which open-source license governs this file? "
        f"Choose exactly one of: {opts}. "
        f"Answer with the SPDX identifier only."
    )


def identification_question(prompt_style: str, labels: list[str]) -> str:
    """Dispatch on prompt_style: 'open_ended' (default) or 'closed_set'."""
    if prompt_style == "closed_set":
        return closed_set_question(labels)
    return open_ended_question()


def clause_question(base_question: str) -> str:
    """Wrap a clause probe; the file is shown via the prefix/suffix segments."""
    return base_question


# ---------------------------------------------------------------------------
# Prompt-EOT variants (ClauseBench-grounded)
#
# A washed comment tuned against ONE prompt rendering overfits that rendering, which is
# a large part of why M4-washed comments barely transfer to black-box models that phrase
# the query differently. To make the washer's hard constraint hold across renderings, we
# need a set of prompt variants that is defensible rather than invented.
#
# ClauseBench (Ke et al., ICSE-Companion 2025, ref_papers/ClauseBench_*.pdf) benchmarks
# exactly four prompting strategies for license-clause analysis — basic, few-shot,
# contextual, and full — and reports contextual as the strongest. Its prompt is built from
# four components (task description, simplified markers, input, output format) over a
# 14-term taxonomy with CAN / MUST / CANNOT / NOT SPECIFIED markers. We reuse those four
# strategies as the EOT set, so "robust under prompt variation" means robust under the
# prompting a published clause-analysis framework would actually use.
#
# Every variant must still end in a question the teacher-forced scorer can answer with the
# SAME label set (Yes/No for clauses, SPDX ids for identification) — the EOT set varies the
# framing, never the answer space.
# ---------------------------------------------------------------------------

# ClauseBench Table I: the 14 license terms, split into rights and obligations.
CLAUSEBENCH_RIGHTS = (
    "Distribute", "Modify", "Commercial Use", "Relicense", "Hold Liable",
    "Use Patent Claims", "Sublicense", "Use Trademark",
)
CLAUSEBENCH_OBLIGATIONS = (
    "Include Copyright", "Include License", "Disclose Source", "Give Credit",
    "Rename", "Contact Author",
)
CLAUSEBENCH_MARKERS = ("CAN", "CANNOT", "MUST", "NOT SPECIFIED")

# ClauseBench builds its input around ONE term at a time — "you will be provided with only
# those sections that are directly relevant to the specified term" — pairing the term with
# its definition rather than dumping the whole taxonomy into every prompt. Mapping our six
# binary probes onto that taxonomy keeps the contextual/full variants faithful AND short:
# enumerating all 14 terms cost 185 tokens of the 700-token single-card budget, which was
# enough to push every GPL-2.0-classpath file out of the clause pool.
CLAUSEBENCH_TERM_FOR_PROBE = {
    "disclose_source": ("Disclose Source",
                        "whether source code must be made available to recipients"),
    "network_use_is_distribution": ("Distribute",
                                    "what counts as conveying the work to others"),
    "patent_grant": ("Use Patent Claims",
                     "whether the license grants rights under contributors' patents"),
    "permits_proprietary_linking": ("Sublicense",
                                    "whether the work may be relicensed into a closed-source"
                                    " combined work"),
    "permit_commercial_use": ("Commercial Use",
                              "whether the work may be used for commercial purposes"),
    "permit_redistribution": ("Distribute",
                              "whether copies of the work may be shared with others"),
}

_CB_TASK = (
    "You are assigned to determine and interpret the presence and meaning of a specific "
    "licensing term for the source file above, using the license evidence the file carries."
)

_CB_MARKERS_BLOCK = (
    "Rights terms are marked CAN or CANNOT, obligation terms MUST, and a term the license "
    "leaves unclear is marked NOT SPECIFIED."
)


def _cb_term_block(probe_key: str | None) -> str:
    """The 'clause name and explanation' component of the ClauseBench input, for one term."""
    entry = CLAUSEBENCH_TERM_FOR_PROBE.get(probe_key or "")
    if entry is None:
        return ""
    term, desc = entry
    return f"Term under analysis: {term} — {desc}.\n"


def clause_prompt_variants(base_question: str, probe_key: str | None = None) -> list[str]:
    """The four ClauseBench prompting strategies for one binary clause probe.

    Returned in a fixed order — basic, few-shot, contextual, full — so that a stored
    per-variant margin vector is interpretable across runs. Index 0 (basic) is the
    rendering the existing clause experiments use, so it stays the primary segment.
    """
    basic = base_question

    few_shot = (
        "Example: for a file under MIT, \"Does this license require that derivative "
        "works be released as open source?\" -> No.\n"
        "Example: for a file under GPL-3.0, the same question -> Yes.\n"
        + base_question
    )

    contextual = (
        _CB_TASK + "\n" + _CB_MARKERS_BLOCK + "\n" + _cb_term_block(probe_key)
        + base_question
    )

    full = (
        _CB_TASK + "\n" + _CB_MARKERS_BLOCK + "\n" + _cb_term_block(probe_key)
        + "Example: for a file under MIT, \"Does this license require that derivative "
        "works be released as open source?\" -> No.\n"
        "Work from the license evidence in the file, not from its name or path. "
        "Answer with the single word only.\n"
        + base_question
    )

    return [basic, few_shot, contextual, full]


def ident_prompt_variants(labels: list[str]) -> list[str]:
    """Prompt renderings for the identification objective.

    Index 0 is `open_ended` — the main-table default and therefore the primary segment.
    """
    contextual = (
        "You are assigned to identify which open-source license governs the source file "
        "above, using only the license evidence the file carries (header comments, notices, "
        "SPDX tags) and not the file name or path.\n"
        "Report the license as an SPDX identifier. If the evidence does not identify a "
        "license, report NOASSERTION.\n"
        + open_ended_question()
    )
    return [open_ended_question(), closed_set_question(labels), contextual]
