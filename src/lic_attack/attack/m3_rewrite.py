"""M3-lite washer: a single, forward-free license-semantic residue.

This is the simplified rewriter written for the new closed-set pipeline (seeds from
``scripts/run_m3_seed.py`` -> ``outputs/m3_seeds_v2/*.jsonl``).  It keeps the M4 idea
-- simulated-annealing edits under a "still flips" hard constraint, minimizing how much
the comment reads as a license claim -- but throws out the two expensive/indirect parts
of M4's residue:

  * S1 (number of OTHER licenses named) is DROPPED: the keyword bag below already lists
    every closed-set license short-name, so naming ``Apache-2.0`` is a keyword hit.
  * S3 (P("Yes"| "does this read as a license claim?") via the VICTIM model) is DROPPED:
    it cost one forward pass per feasible candidate. We replace the thing it was for --
    catching license semantics the exact keyword regex misses -- with an embedding-space
    similarity to the keyword bag, which is a table lookup + one matmul, NO forward.

So the residue is one concept, "license-semantic score":

    S = w_keyword * (# exact keyword-bag hits)
      + w_sim     * (# comment tokens whose embedding is near a bag word)

Both terms are forward-free.  The ONLY forward pass left in the loop is the hard
constraint ``margin(cand) < tau`` -- i.e. "does the edited comment still flip the model" --
which is the attack's definition and cannot be avoided in a white-box washer.

The ``reorder`` operator is gone (that is what makes this an M3 run); only
delete / substitute / insert remain, and substitute/insert still draw GCG gradient
top-k candidates so edits tend to preserve the flip rather than break it at random.

Cosine on RAW input embeddings is unreliable (they are anisotropic -- unrelated tokens
sit at cosine ~0.3-0.6).  We mean-center by the vocabulary mean embedding before
normalizing, which is the standard fix and makes the 0.5 threshold meaningful.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

import torch

from ..model.base import DetectorModel, PromptSegments
from .rewrite_search import _grad_topk, _margin_of, _propose

# --- Expanded license-semantic keyword bag ------------------------------------------
# Exact-match arm (regex, with stems/wildcards).  Superset of rewrite_search._SEMANTIC_KEYWORDS
# plus derivative-work / share-alike / network / ClauseBench-taxonomy terms and every
# closed-set license short-name, so a comment naming any of them is scored as residue.
_SEMANTIC_KEYWORDS = re.compile(
    r"\b("
    # core license vocabulary
    r"licen[cs]\w*|copyright|copyleft|proprietary|closed[- ]?source|open[- ]?source|"
    r"redistribut\w*|distribut\w*|sublicen\w*|relicen\w*|patent\w*|trademark\w*|"
    r"warrant\w*|liabilit\w*|disclos\w*|source\s+code|permission\w*|permit\w*|grant\w*|"
    r"public\s+domain|royalt\w*|obligation\w*|compl(?:y|ies|iance)|notice|"
    r"terms\s+and\s+conditions|"
    # derivative-work / copyleft mechanics
    r"derivativ\w*|modif\w*|share[- ]?alike|attribution|credit|waiv\w*|exception|"
    r"combin\w*|link(?:ing|ed)?|statical\w*|dynamical\w*|incorporat\w*|"
    r"terminat\w*|revok\w*|retaliat\w*|reserved\s+rights|all\s+rights\s+reserved|"
    # network / commercial reframes
    r"network\s+use|over\s+a\s+network|service|saas|commercial\s+use|non[- ]?commercial|"
    r"free\s+of\s+charge|freely|"
    # SPDX + license families / short-names (closed set)
    r"spdx|identifier|affero|lesser|general\s+public|mozilla|eclipse|"
    r"gpl|lgpl|agpl|mpl|epl|bsd|mit|apache|cddl|sspl|osl|artistic|bsl|isc|unlicense|"
    # ClauseBench 14-term taxonomy verbs not already covered
    r"hold\s+liable|use\s+trademark|give\s+credit|rename|contact\s+author"
    r")\b",
    re.IGNORECASE,
)

# Fuzzy arm (embedding similarity).  Plain surface forms used to build the bag vectors;
# these seed the semantic neighbourhood the cosine test flags, catching paraphrases the
# regex misses (e.g. "publish" ~ "distribute", "grants" ~ "permits").
BAG_WORDS: tuple[str, ...] = (
    "license", "licensed", "licensing", "copyright", "copyleft", "proprietary",
    "closed-source", "open-source", "redistribute", "distribute", "distribution",
    "sublicense", "relicense", "patent", "trademark", "warranty", "liability",
    "disclose", "source", "permission", "permit", "permits", "grant", "grants",
    "public domain", "royalty", "obligation", "comply", "compliance", "notice",
    "terms", "conditions", "derivative", "modify", "modified", "modification",
    "share-alike", "attribution", "credit", "waiver", "exception", "combine",
    "linking", "incorporate", "terminate", "revoke", "retaliation", "reserved",
    "network", "service", "commercial", "noncommercial", "freely",
    "spdx", "identifier", "affero", "lesser", "mozilla", "eclipse",
    "GPL", "LGPL", "AGPL", "MPL", "EPL", "BSD", "MIT", "Apache", "CDDL",
    "SSPL", "OSL", "Artistic", "BSL", "ISC", "Zlib",
)


def keyword_hits(text: str) -> int:
    """Exact license-semantic keyword count (forward-free)."""
    return len(_SEMANTIC_KEYWORDS.findall(text))


class BagIndex:
    """Mean-centered, normalized embeddings of the keyword bag, for cosine near-hits.

    Built once per model (the embedding matrix is fixed), then reused for every seed.
    No forward pass -- just embedding-table lookups and a matmul.
    """

    def __init__(self, model: DetectorModel, words: tuple[str, ...] = BAG_WORDS):
        E = model.embedding_matrix
        self.mu = E.mean(dim=0).detach()               # [H] vocabulary mean (anisotropy fix)
        vecs = []
        for w in words:
            ids = model.encode(w, add_special=False)
            if len(ids) == 0:
                continue
            v = (model.embed_ids(ids).detach() - self.mu).mean(dim=0)   # [H] centered
            n = v.norm()
            if n > 0:
                vecs.append(v / n)
        self.bag = torch.stack(vecs) if vecs else E.new_zeros((0, E.shape[1]))  # [B, H]

    def near_hits(self, model: DetectorModel, comment_ids: torch.Tensor, threshold: float) -> int:
        """# comment tokens whose centered embedding cosine-exceeds `threshold` to any bag word."""
        if len(comment_ids) == 0 or self.bag.shape[0] == 0:
            return 0
        with torch.no_grad():
            ce = model.embed_ids(comment_ids.to(model.device)).detach() - self.mu   # [m, H]
            ce = ce / ce.norm(dim=1, keepdim=True).clamp_min(1e-6)
            sims = ce @ self.bag.T                                                  # [m, B]
            return int((sims.max(dim=1).values > threshold).sum())


@dataclass
class M3RewriteConfig:
    tau: float = 0.0                 # hard constraint: keep only comments with margin < tau
    n_steps: int = 300
    top_k: int = 64                  # GCG gradient candidates per position (sub/insert)
    t0: float = 1.0                  # initial SA temperature
    t_end: float = 0.02              # final SA temperature (geometric schedule)
    max_len: int = 96                # cap comment length (insert can grow it)
    min_len: int = 1
    seed: int = 0
    p_delete: float = 0.375
    p_substitute: float = 0.375
    p_insert: float = 0.25
    # residue weights (single license-semantic score = keyword + embedding-similarity)
    w_keyword: float = 1.0
    use_sim: bool = True
    w_sim: float = 1.0
    sim_threshold: float = 0.5
    # early stop: STALL-based. Stop once the best residue has not improved for `patience`
    # consecutive feasible candidates, regardless of how low S actually is. This is the fix
    # for seeds whose residue floor is a load-bearing keyword (S plateaus at ~2 and can never
    # cross a fixed threshold, so a threshold-gated early stop would run the full schedule for
    # nothing). A min-steps floor lets the high-temperature phase explore before stalling counts.
    early_stop: bool = True
    patience: int = 20               # consecutive feasible candidates w/o a new best-S -> stop
    early_stop_min_steps: int = 40   # never stall-stop before this many iterations
    # Optional M4-style gate on top of the stall rule: when set, a stall may only end the
    # search once the best residue is already at or below this value. Seeds that plateau
    # above it keep searching to the end of the schedule, which is what makes the share
    # reaching the threshold a property of the budget rather than of where the seed
    # happened to stall. None keeps the stall-only behaviour.
    early_stop_S: float | None = None
    # OOM guard: the gradient proposal (_grad_topk) runs BACKWARD over the whole sequence, whose
    # activation memory blows up an 8B on a 24GB card for long source files. The no_grad margin
    # checks are safe at any length, so we only gate the gradient: for a sequence longer than
    # grad_max_len tokens, skip gradient proposals and fall back to delete + random safe-token
    # edits (delete, the main residue-reducer, needs no gradient). The full file is still scored,
    # so seed validity and the margin constraint are unchanged.
    grad_max_len: int = 700


@dataclass
class M3RewriteResult:
    true_label: str
    start_text: str
    washed_text: str
    washed_ids: torch.Tensor
    start_margin: float
    final_margin: float
    start_S: float
    final_S: float
    start_kw: int
    final_kw: int
    start_sim: int
    final_sim: int
    success: bool
    steps_used: int
    trace: list = field(default_factory=list)


def _residue(model: DetectorModel, ids: torch.Tensor, cfg: M3RewriteConfig,
             bag: BagIndex | None) -> tuple[float, int, int]:
    """Single license-semantic residue (forward-free): keyword hits + embedding near-hits."""
    text = model.decode(ids)
    kw = keyword_hits(text)
    sim = bag.near_hits(model, ids, cfg.sim_threshold) if (cfg.use_sim and bag is not None) else 0
    S = cfg.w_keyword * kw + cfg.w_sim * sim
    return S, kw, sim


def m3_rewrite_search(
    model: DetectorModel,
    seg: PromptSegments,
    comment_ids: torch.Tensor,
    labels: list[str],
    true_label: str,
    cfg: M3RewriteConfig,
    bag: BagIndex | None = None,
    calib=None,
) -> M3RewriteResult:
    """Simulated-annealing washer with a single forward-free license-semantic residue.

    Starts from an ALREADY-FLIPPING comment (margin < tau).  Each step proposes one edit
    (delete / substitute / insert; substitute/insert from GCG gradient top-k).  A proposal
    is rejected outright if it breaks the flip (margin >= tau).  Among flip-preserving
    proposals, accept by Metropolis on the residue S (always if S drops, else with prob
    exp(-dS/T)); temperature cools geometrically so early steps explore and late steps
    refine.  The best (lowest-S, still-flipping) comment seen is returned.
    """
    device = model.device
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    label_token_ids = [model.encode(l, add_special=False) for l in labels]
    safe_mask = model.comment_safe_mask().to(device)
    safe_ids = torch.nonzero(safe_mask, as_tuple=False).squeeze(-1).cpu()
    if cfg.use_sim and bag is None:
        bag = BagIndex(model)

    cur = comment_ids.detach().cpu().clone()
    start_text = model.decode(cur)
    start_margin = _margin_of(model, seg, cur.to(device), labels, true_label, label_token_ids, calib)

    # Nothing to wash if the seed doesn't actually flip the model.
    if start_margin >= cfg.tau:
        S0, kw0, sim0 = _residue(model, cur, cfg, bag)
        return M3RewriteResult(
            true_label=true_label, start_text=start_text, washed_text=start_text,
            washed_ids=cur, start_margin=start_margin, final_margin=start_margin,
            start_S=S0, final_S=S0, start_kw=kw0, final_kw=kw0, start_sim=sim0,
            final_sim=sim0, success=False, steps_used=0)

    cur_S, cur_kw, cur_sim = _residue(model, cur, cfg, bag)
    start_S, start_kw, start_sim = cur_S, cur_kw, cur_sim
    best, best_S, best_stats = cur.clone(), cur_S, (cur_kw, cur_sim)

    ops = ["delete", "substitute", "insert"]              # M3: no reorder
    probs = [cfg.p_delete, cfg.p_substitute, cfg.p_insert]
    trace: list = []
    topk = None
    topk_age = 0
    no_improve = 0
    base_len = len(seg.prefix_ids) + len(seg.suffix_ids)   # fixed frozen context length
    grad_disabled = base_len > cfg.grad_max_len            # too long to backward safely
    grad_oom = False                                       # set once a backward OOMs this seed

    for step in range(cfg.n_steps):
        T = cfg.t0 * (cfg.t_end / cfg.t0) ** (step / max(1, cfg.n_steps - 1))

        if os.environ.get("WASH_STEP_LOG") and step % 20 == 0:
            print(f"    [m3 step {step}/{cfg.n_steps}] len={len(cur)} S={cur_S:.1f} "
                  f"(kw={cur_kw} sim={cur_sim})", flush=True)

        # (re)compute gradient proposals occasionally; they only PROPOSE, so primary seg only.
        # Skipped entirely for over-long sequences (grad_disabled) or after a backward OOM
        # (grad_oom) -- the search then uses delete + random safe-token proposals, which need
        # no gradient. This keeps long files washing (mainly via delete) without OOM.
        if not grad_disabled and not grad_oom and (topk is None or topk_age >= 10):
            if len(cur) > 0 and base_len + len(cur) <= cfg.grad_max_len:
                try:
                    topk = _grad_topk(model, seg, cur.to(device), labels, true_label,
                                      label_token_ids, safe_mask, cfg.top_k, calib).cpu()
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    topk, grad_oom = None, True
            topk_age = 0
        topk_age += 1

        op = ops[int(torch.multinomial(torch.tensor(probs), 1, generator=gen))]
        cand = _propose(cur, op, topk, safe_ids, cfg, gen)
        if len(cand) == 0 or torch.equal(cand, cur):
            continue

        # hard constraint: must still flip the model (the only forward in the loop)
        cand_margin = _margin_of(model, seg, cand.to(device), labels, true_label, label_token_ids, calib)
        if cand_margin >= cfg.tau:
            continue

        cand_S, c_kw, c_sim = _residue(model, cand, cfg, bag)

        dS = cand_S - cur_S
        if dS <= 0 or torch.rand(1, generator=gen).item() < math.exp(-dS / max(1e-6, T)):
            cur, cur_S, cur_kw, cur_sim = cand, cand_S, c_kw, c_sim
            topk = None                                  # ids changed -> stale proposals
            if cur_S < best_S - 1e-9:
                best, best_S, best_stats = cur.clone(), cur_S, (c_kw, c_sim)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        trace.append((step, op, cur_S, cand_margin))

        # Stall-based early stop: best-S has not improved for `patience` feasible candidates.
        # No S-threshold gate, so a seed stuck at a keyword floor stops instead of burning the
        # whole schedule; the min-steps floor protects the high-temperature exploration phase.
        if (cfg.early_stop and step >= cfg.early_stop_min_steps
                and no_improve >= cfg.patience
                and (cfg.early_stop_S is None or best_S <= cfg.early_stop_S)):
            if os.environ.get("WASH_STEP_LOG"):
                print(f"    [m3 stall-stop @ step {step}] best_S={best_S:.1f} "
                      f"no_improve={no_improve}", flush=True)
            break

    final_margin = _margin_of(model, seg, best.to(device), labels, true_label, label_token_ids, calib)
    return M3RewriteResult(
        true_label=true_label, start_text=start_text, washed_text=model.decode(best),
        washed_ids=best, start_margin=start_margin, final_margin=final_margin,
        start_S=start_S, final_S=best_S, start_kw=start_kw, final_kw=best_stats[0],
        start_sim=start_sim, final_sim=best_stats[1],
        success=final_margin < cfg.tau, steps_used=len(trace), trace=trace)
