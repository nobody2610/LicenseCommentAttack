"""Scoring: closed-set label log-probabilities and untargeted objectives.

Given the prompt segments and the current comment ids, we score each closed-set
label by teacher-forcing its tokens after the assistant header and summing their
log-probs. From the per-label logprob vector we derive:

  * margin(true)   = logp[true] - max_{y != true} logp[y]   (CW-style, untargeted)
  * entropy        = H(softmax(logp))                        (confusion objective)

The attack minimizes margin (push true label out of first place) or maximizes
entropy (flatten the distribution). Both are untargeted: we never specify which
wrong label to land on.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model.base import DetectorModel, PromptSegments


@dataclass
class LabelScores:
    logprobs: torch.Tensor   # [L] summed teacher-forced logprob per label
    labels: list[str]

    def margin(self, true_label: str) -> torch.Tensor:
        ti = self.labels.index(true_label)
        mask = torch.ones(len(self.labels), dtype=torch.bool, device=self.logprobs.device)
        mask[ti] = False
        other_max = self.logprobs[mask].max()
        return self.logprobs[ti] - other_max

    def argmax_label(self) -> str:
        return self.labels[int(self.logprobs.argmax())]

    def entropy(self) -> torch.Tensor:
        logp = torch.log_softmax(self.logprobs, dim=0)
        p = logp.exp()
        return -(p * logp).sum()


def _label_token_ids(model: DetectorModel, labels: list[str]) -> list[torch.Tensor]:
    return [model.encode(lbl, add_special=False) for lbl in labels]


def _reduce(total: torch.Tensor, n_tok: int, idx: int,
            calib: torch.Tensor | None) -> torch.Tensor:
    """Turn a label's summed teacher-forced logprob into its comparable score.

    calib is None -> length-normalised mean, the historical behaviour.
    calib given   -> PMI: the summed logprob minus what the same label scores on a
                     content-free prompt.

    Why the default is wrong for a closed set, and why PMI is the fix:

    Averaging is a beam-search heuristic for comparing generation candidates of
    different lengths. Here it rewards long labels whose continuation is nearly
    deterministic once the first token is fixed ("LGPL" -> "-2.1" costs almost
    nothing per token) and punishes single-token labels that must carry all their
    mass at once. With MIT and ISC at 1 token, Zlib at 2, and most of the rest at
    5-6, that is not a rounding error: measured over 268 files, the mean rule gives
    MIT, Zlib and Artistic-2.0 a 0% clean accuracy on Llama-3-8B while PMI gives
    them 1%, 100% and 100%.

    Plain summation is not the fix either -- it just moves the bias, favouring
    whatever surface form the model likes a priori (BSL-1.1 and LGPL-3.0 collapse
    to 0% and 40%). Subtracting the label's score on a null prompt removes that
    prior, which is the standard calibration for closed-set LM classification
    (Holtzman et al. 2021, surface form competition; Zhao et al. 2021, contextual
    calibration). Across the four detectors it raises clean accuracy by up to 33
    points (License-Llama 23% -> 56%) and cuts licences with a 0% gate rate from
    5/3/6/12 to 1/2/3/2.
    """
    if calib is None:
        return total / max(1, n_tok)
    return total - calib[idx]


def null_calibration(model: DetectorModel, question: str, labels: list[str],
                     label_token_ids: list[torch.Tensor] | None = None) -> torch.Tensor:
    """Per-label summed logprob on a content-free prompt: the PMI reference term.

    The premise is the same question asked about an empty file, so what remains in
    each label's score is the model's unconditional pull toward that surface form.
    Constant for a given (model, question, label set) -- compute it once per run.
    """
    if label_token_ids is None:
        label_token_ids = _label_token_ids(model, labels)
    seg = model.build_segments("", "", question)
    empty = model.encode("", add_special=False)
    mean = score_labels_from_embeds(model, seg, empty, labels, label_token_ids).logprobs
    return torch.tensor([float(mean[i]) * max(1, len(label_token_ids[i]))
                         for i in range(len(labels))], device=mean.device)


def score_labels_from_embeds(
    model: DetectorModel,
    seg: PromptSegments,
    comment_ids: torch.Tensor,
    labels: list[str],
    label_token_ids: list[torch.Tensor] | None = None,
    calib: torch.Tensor | None = None,
) -> LabelScores:
    """Score every label by teacher forcing, sharing the prefix forward pass.

    Builds, per label, the sequence [prefix ++ comment ++ suffix ++ label_tokens]
    and reads the logprob the model assigns to each label token at the preceding
    position. comment_ids enters as ids here (no grad); the differentiable path
    used by the attack lives in ``objective_from_onehot`` below.

    This is a pure forward evaluation (argmax/margin readout), so it runs under
    ``torch.no_grad()`` — no autograd graph is built. Wrapping this loop in
    no_grad is the single biggest speedup for the candidate search, which calls
    it hundreds of times per GCG step without ever needing the graph.
    """
    if label_token_ids is None:
        label_token_ids = _label_token_ids(model, labels)
    with torch.no_grad():
        base = torch.cat([seg.prefix_ids, comment_ids.to(seg.prefix_ids.device), seg.suffix_ids])
        logps = []
        for lbl_ids in label_token_ids:
            l = len(lbl_ids)
            full = torch.cat([base, lbl_ids.to(base.device)])
            emb = model.embed_ids(full).unsqueeze(0)
            # Only the last (l+1) positions carry label-token predictors. Projecting the
            # FULL [T, V] logits is what OOMs GPU1 on long source files (T can be 2000+,
            # V~152k); logits_to_keep makes the LM head project just the tail. Label token
            # j (predicted at global position len(base)+j-1) maps to local index j here,
            # since the kept window starts at global position len(base)-1.
            keep = l + 1
            logits = model.logits_from_embeds(emb, logits_to_keep=keep)[0]  # [keep, V]
            logp = torch.log_softmax(logits, dim=-1)
            total = torch.zeros((), device=logp.device)
            for j, tok in enumerate(lbl_ids.tolist()):
                total = total + logp[j, tok]
            logps.append(_reduce(total, l, len(logps), calib))
    return LabelScores(logprobs=torch.stack(logps), labels=labels)


def score_labels_batch(
    model: DetectorModel,
    seg: PromptSegments,
    cand_comment_ids: torch.Tensor,   # [B, m] candidate comments (ids, no grad)
    labels: list[str],
    label_token_ids: list[torch.Tensor] | None = None,
    calib: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched discrete scorer: score B candidate comments against all labels at once.

    Returns logprobs [B, L] — the length-normalized mean per-token logprob each
    candidate assigns to each label, numerically identical to calling
    ``score_labels_from_embeds`` once per candidate.

    The GCG inner loop evaluates ``batch_size`` single-token substitutions every
    step; doing that as ``B`` separate forward passes (each looping over labels)
    is the dominant cost. Here we instead run, per label, ONE forward over the
    whole candidate batch — L forwards total instead of B*L — so the GPU sees a
    real batch dimension and the per-call launch/overhead is amortized.

    Memory: a [B, T, V] logits tensor is enormous (V~152k). We pass
    ``logits_to_keep`` so the LM head projects only the trailing positions that
    carry label-token predictions, keeping the logits tensor at [B, N, V] with N
    = max label length + 1.
    """
    if label_token_ids is None:
        label_token_ids = _label_token_ids(model, labels)
    device = seg.prefix_ids.device
    B, m = cand_comment_ids.shape
    L = len(labels)
    cand = cand_comment_ids.to(device)

    # Per-label, all label sequences share the same base length (prefix+comment+suffix),
    # so each label's tokens occupy a fixed trailing block — we only need logits over
    # the last (max_lbl_len + 1) positions to read every label token's predictor.
    base_len = len(seg.prefix_ids) + m + len(seg.suffix_ids)
    max_lbl = max(len(t) for t in label_token_ids)
    keep = max_lbl + 1  # +1: token 0 of a label is predicted at the position before base

    with torch.no_grad():
        prefix_emb = model.embed_ids(seg.prefix_ids)            # [P, H]
        suffix_emb = model.embed_ids(seg.suffix_ids)            # [S, H]
        cand_emb = model.embed_ids(cand)                        # [B, m, H]
        prefix_b = prefix_emb.unsqueeze(0).expand(B, -1, -1)
        suffix_b = suffix_emb.unsqueeze(0).expand(B, -1, -1)

        out = torch.empty(B, L, device=device)
        for li, lbl_ids in enumerate(label_token_ids):
            lbl_ids = lbl_ids.to(device)
            lbl_emb = model.embed_ids(lbl_ids).unsqueeze(0).expand(B, -1, -1)  # [B, l, H]
            emb = torch.cat([prefix_b, cand_emb, suffix_b, lbl_emb], dim=1)    # [B, T, H]
            logits = model.logits_from_embeds(emb, logits_to_keep=keep)        # [B, keep, V]
            logp = torch.log_softmax(logits, dim=-1)
            # Global predictor of label token j is at position base_len+j-1; the kept
            # window is the last `keep` positions, i.e. global index t maps to local
            # index t-(T-keep) where T = base_len+len(lbl_ids).
            T = base_len + len(lbl_ids)
            offset = T - keep
            total = torch.zeros(B, device=device)
            for j, tok in enumerate(lbl_ids.tolist()):
                total = total + logp[:, (base_len + j - 1) - offset, tok]
            out[:, li] = _reduce(total, len(lbl_ids), li, calib)
    return out


def objective_from_onehot(
    model: DetectorModel,
    seg: PromptSegments,
    comment_onehot: torch.Tensor,   # [m, V] requires_grad
    labels: list[str],
    true_label: str,
    label_token_ids: list[torch.Tensor],
    objective: str = "margin",
    calib: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable objective wrt the comment one-hot, for GCG gradients.

    Returns a scalar LOSS to minimize:
      * margin  -> loss = margin(true)            (lower = true label loses)
      * entropy -> loss = -entropy                (lower = more confused)

    MEMORY (margin path): the margin loss = logp[true] - max_{y!=true} logp[y] only
    depends on TWO labels — the true label and its strongest competitor. The other
    labels contribute ZERO gradient (they are not selected by the max). So we first
    find the competitor under ``torch.no_grad()`` (no autograd graph), then build the
    backward graph for ONLY {true, competitor} — 2 resident graphs instead of L (~10).
    That is what lets the 8B model's GCG backward fit on a single 24GB card. The result
    is numerically identical to scoring all labels with grad and taking the margin.
    The entropy objective needs the full distribution over all labels, so it keeps the
    all-labels path.
    """
    E = model.embedding_matrix
    comment_emb = comment_onehot @ E  # [m, H], differentiable into one-hot
    prefix_emb = model.embed_ids(seg.prefix_ids)
    suffix_emb = model.embed_ids(seg.suffix_ids)

    def label_logp(lbl_ids: torch.Tensor, idx: int) -> torch.Tensor:
        """Length-normalized teacher-forced logprob of one label, as a differentiable
        scalar in the comment one-hot's graph. Projects only the trailing label
        positions (logits_to_keep) — same as score_labels_batch, no gradient change."""
        lbl_emb = model.embed_ids(lbl_ids)
        emb = torch.cat([prefix_emb, comment_emb, suffix_emb, lbl_emb], dim=0).unsqueeze(0)
        keep = len(lbl_ids) + 1
        logits = model.logits_from_embeds(emb, logits_to_keep=keep)[0]  # [keep, V]
        logp = torch.log_softmax(logits, dim=-1)
        # With keep = len(label)+1, global predictor position (start+j-1) maps to the
        # local index j within the kept window (offset = (start+len) - keep = start-1).
        total = torch.zeros((), device=logp.device)
        for j, tok in enumerate(lbl_ids.tolist()):
            total = total + logp[j, tok]
        return _reduce(total, len(lbl_ids), idx, calib)

    ti = labels.index(true_label)

    if objective == "margin":
        # Pass 1 (no_grad): score every label cheaply, no autograd graph, to find the
        # strongest non-true competitor.
        with torch.no_grad():
            vals = [float(label_logp(lbl, i).item()) for i, lbl in enumerate(label_token_ids)]
        ci, cbest = -1, None
        for i, v in enumerate(vals):
            if i == ti:
                continue
            if cbest is None or v > cbest:
                cbest, ci = v, i
        # Pass 2 (grad): build graphs for ONLY {true, competitor}. margin = lp_true - lp_comp.
        lp_true = label_logp(label_token_ids[ti], ti)
        lp_comp = label_logp(label_token_ids[ci], ci)
        return lp_true - lp_comp

    if objective == "entropy":
        # entropy needs the full distribution over all labels -> all-labels grad path.
        logp_vec = torch.stack([label_logp(lbl, i) for i, lbl in enumerate(label_token_ids)])
        lp = torch.log_softmax(logp_vec, dim=0)
        ent = -(lp.exp() * lp).sum()
        return -ent
    raise ValueError(f"unknown objective {objective!r}")
