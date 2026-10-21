"""Load LLM-generated attack comments from the offline cache produced by
scripts/gen_llm_comments.py. Kept separate from licenses.py/handcrafted.py because
this source is OPTIONAL (only present when an llm_comments_*.json cache exists) and
must never make the GPU attack path depend on network or an API key.

Cache formats (v2 adds a direction axis; v1 fields are still written and still read):
  license: {"target":"license",
            "by_direction": {"from_copyleft":[str,...], "from_permissive":[str,...]},
            "comments": [str, ...]}                      # v1 union, fallback
  clause:  {"target":"clause",
            "by_probe_truth": {probe: {"Yes":[...], "No":[...]}},
            "by_probe": {probe: [str, ...]}}             # v1 copyleft-only, fallback

WHY THE DIRECTION AXIS EXISTS: the v1 cache was generated entirely from a
copyleft-source prompt, so every comment argues "this file is really permissive". On a
file that is ALREADY permissive that argument is vacuous, and it shows up in the numbers
-- the llm arm flipped Apache-2.0 identification 0/17, and clause flip on the four
license-dependent probes was 10.8% for permissive sources vs 38.4% for copyleft. Picking
the comment set by the source file's own license family removes that confound.

Loaders return [] when the cache file is absent, so the runner silently skips the
LLM arm unless the user generated it.
"""
from __future__ import annotations

import json
from pathlib import Path

# Families the generator produces comment sets for. Anything not listed as copyleft is
# treated as permissive, which matches how the corpus is built (9 license families,
# 6 copyleft + 3 permissive).
_COPYLEFT = {
    "GPL-2.0", "GPL-3.0", "GPL-2.0-classpath", "AGPL-3.0", "LGPL-2.1", "LGPL-3.0",
    "MPL-2.0", "EPL-2.0", "CDDL-1.0", "SSPL-1.0", "BSL-1.1", "Elastic-2.0",
    "Redis-SALv2", "CockroachDB-CSL", "Confluent-Community", "OSL-3.0", "Artistic-2.0",
}


def direction_for(license_id: str | None) -> str:
    """Which generated comment set applies to a file under `license_id`.

    A copyleft source is attacked toward permissive; a permissive source toward copyleft.
    Unknown/None defaults to the copyleft direction, which is what the v1 cache held.
    """
    if license_id and license_id not in _COPYLEFT:
        return "from_permissive"
    return "from_copyleft"


def _cache(path: str | None) -> dict:
    if not path or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())


def load_llm_license_comments(path: str | None,
                              license_id: str | None = None) -> list[str]:
    """Identification comments for a source file under `license_id`.

    license_id=None keeps the v1 behaviour (flat union), so callers that have not been
    made direction-aware still work.
    """
    data = _cache(path)
    if not data:
        return []
    by_dir = data.get("by_direction")
    if by_dir and license_id is not None:
        got = by_dir.get(direction_for(license_id))
        if got:
            return list(got)
    if by_dir and license_id is None:
        return [c for v in by_dir.values() for c in v]
    return list(data.get("comments", []))


def load_llm_clause_comments(path: str | None, probe_key: str,
                             license_id: str | None = None) -> list[str]:
    """Clause comments for `probe_key` on a source file under `license_id`.

    Selection is by the GROUND-TRUTH answer, not by license family. Four of the six
    probes disagree within the copyleft family (network-use is Yes for AGPL-3.0 but No
    for GPL-3.0; proprietary-linking is Yes for LGPL-2.1 but No for GPL-3.0), so a
    family-keyed cache would still hand a sibling license the wrong argument direction.
    """
    data = _cache(path)
    if not data:
        return []
    bpt = data.get("by_probe_truth")
    if bpt and license_id is not None:
        from .licenses import CLAUSE_PROBES
        probe = next((p for p in CLAUSE_PROBES if p.key == probe_key), None)
        truth = probe.truth.get(license_id) if probe else None
        if truth is not None:
            got = (bpt.get(probe_key) or {}).get(truth)
            if got:
                return list(got)
    if bpt and license_id is None:
        return [c for v in (bpt.get(probe_key) or {}).values() for c in v]
    return list(data.get("by_probe", {}).get(probe_key, []))
