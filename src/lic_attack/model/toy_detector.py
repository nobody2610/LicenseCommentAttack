"""Tiny deterministic offline backend for validating the pipeline on CPU.

ToyDetector is NOT a language model. It exists so the entire attack pipeline —
embedding gradients, GCG candidate search, margin/entropy objectives, probe
parsing — can be exercised end-to-end with no model download, in seconds, on a
laptop. Its "logits" are a fixed differentiable function of the input embeddings,
constructed so that:

  * each closed-set license label has a learned-free, deterministic key vector,
  * the score for a label is the mean input embedding dotted with that key,
  * therefore changing comment tokens *does* move label scores (so GCG has a real
    gradient signal to follow), and the result is fully reproducible.

The vocabulary is a small fixed set of printable ASCII chunks, so tokenization is
trivial and every token is comment-safe by construction.
"""
from __future__ import annotations

import torch

from .base import DetectorModel, PromptSegments

# Fixed vocab: pad, printable ASCII chars, a few license words, the full SPDX label
# strings as single tokens, AND "Yes"/"No" as atomic tokens (for clause attacks).
# Label strings cover the Experiment-A scoring set (copyleft + permissive + unknown).
_LABEL_STRINGS = [
    "GPL-3.0", "AGPL-3.0", "LGPL-3.0", "MPL-2.0", "EPL-2.0", "CDDL-1.0",
    "MIT", "Apache-2.0", "BSD-3-Clause", "NOASSERTION",
]
_YESNO = ["Yes", "No"]
_VOCAB = [
    "<pad>",  # 0 — benign filler / not comment-safe
] + [chr(c) for c in range(32, 127)] + [
    "License", "Copyright", "SPDX", "identifier", "permit", "require", "source", "free",
] + _LABEL_STRINGS + _YESNO
_STOI = {tok: i for i, tok in enumerate(_VOCAB)}
_HIDDEN = 32


class ToyDetector(DetectorModel):
    """Deterministic surrogate. NOT a language model.

    Mechanism (faithful to the real attack's lever): the first L embedding
    dimensions are dedicated one-per-label. Each license's full SPDX token
    ("GPL-3.0", "MIT", ...) activates its own dimension, and a label's score is the
    running COUNT of that token in the prompt. So:

      * a GPL header (whose SPDX line contains the atomic token "GPL-3.0") gives the
        GPL label one more hit than any other -> clean detection is CORRECT,
      * inserting a competing license's SPDX token into the comment raises that
        label's count and can flip the argmax -> exactly the comment-perturbation
        lever GCG is meant to find, with a real gradient pointing at it.
    """

    def __init__(self, labels: list[str], seed: int = 0):
        self._device = torch.device("cpu")
        self._labels = list(labels)
        L = len(labels)
        # dims [0..L) are per-label channels; dims [L], [L+1] are the Yes / No channels.
        assert L + 2 <= _HIDDEN, "need one dim per label plus 2 for Yes/No"
        for lbl in labels:
            assert lbl in _STOI, f"label {lbl!r} missing from toy vocab"
        g = torch.Generator().manual_seed(seed)
        # Small random base so every token has a non-degenerate embedding (keeps
        # gradients well-defined).
        self._embed = 0.05 * torch.randn(len(_VOCAB), _HIDDEN, generator=g)
        # The first L dims are RESERVED label channels: zero them everywhere so that
        # only a label's own SPDX token writes to its channel. Otherwise the +3 count
        # signal is swamped by random noise summed over hundreds of prompt tokens.
        self._embed[:, :L] = 0.0
        self._label_ids = [_STOI[lbl] for lbl in labels]   # full-string token per label
        for li, lbl in enumerate(labels):
            self._embed[_STOI[lbl], li] = 3.0  # label's SPDX token lights up its dim

        # --- Yes/No channels (for clause attacks) ---
        # dims L (Yes) and L+1 (No), also zeroed so only designated tokens write to them.
        self._yes_dim, self._no_dim = L, L + 1
        self._embed[:, self._yes_dim] = 0.0
        self._embed[:, self._no_dim] = 0.0
        # Every license SPDX token contributes a baseline to the Yes channel, so a clean
        # file (header carries one SPDX token) reads as "Yes" by default — matching the
        # OSI-permissive clause truths. Inserting "No" tokens into the comment raises the
        # No channel and can flip the answer: exactly the comment lever GCG should find.
        for lbl in labels:
            self._embed[_STOI[lbl], self._yes_dim] = 2.0
        self._embed[_STOI["No"], self._no_dim] = 3.0
        self._embed[_STOI["Yes"], self._yes_dim] = 3.0
        self._yesno_ids = {"Yes": _STOI["Yes"], "No": _STOI["No"]}

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def vocab_size(self) -> int:
        return len(_VOCAB)

    @property
    def embedding_matrix(self) -> torch.Tensor:
        return self._embed

    @property
    def labels(self) -> list[str]:
        return self._labels

    def encode(self, text: str, add_special: bool = False) -> torch.Tensor:
        ids = []
        i = 0
        # Greedy longest-match over multi-char vocab words, else fall back to chars.
        words = sorted([w for w in _VOCAB if len(w) > 1], key=len, reverse=True)
        while i < len(text):
            matched = None
            for w in words:
                if text.startswith(w, i):
                    matched = w
                    break
            if matched is not None:
                ids.append(_STOI[matched])
                i += len(matched)
            else:
                ch = text[i]
                ids.append(_STOI.get(ch, _STOI[" "]))
                i += 1
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return "".join(_VOCAB[int(i)] if 0 <= int(i) < len(_VOCAB) else "" for i in ids)

    def logits_from_embeds(self, inputs_embeds: torch.Tensor,
                           logits_to_keep: int = 0) -> torch.Tensor:
        # inputs_embeds [B, T, H]. The first L dims are per-label "identifying-token"
        # activations; a label's score at position t is the running SUM of its dim up
        # to t (i.e. how many of its identifying tokens have been seen). This is a
        # differentiable function of the embeddings, so GCG gets a real gradient.
        # Dims L (Yes) and L+1 (No) drive the Yes/No answer tokens the same way, for
        # clause attacks.
        B, T, H = inputs_embeds.shape
        L = len(self._labels)
        running_sum = torch.cumsum(inputs_embeds[:, :, :L + 2], dim=1)  # [B, T, L+2]
        logits = torch.full((B, T, self.vocab_size), -10.0, dtype=inputs_embeds.dtype,
                            device=inputs_embeds.device)
        for li, vid in enumerate(self._label_ids):
            logits[:, :, vid] = running_sum[:, :, li] * 2.0  # scale for sharper argmax
        logits[:, :, self._yesno_ids["Yes"]] = running_sum[:, :, self._yes_dim] * 2.0
        logits[:, :, self._yesno_ids["No"]] = running_sum[:, :, self._no_dim] * 2.0
        # Mirror the HF backend: logits_to_keep>0 returns only the last N positions.
        if logits_to_keep > 0:
            logits = logits[:, -logits_to_keep:, :]
        return logits

    def build_segments(self, source_before: str, source_after: str, question: str) -> PromptSegments:
        prefix = self.encode(source_before)
        suffix = self.encode(source_after + "\n" + question)
        return PromptSegments(prefix_ids=prefix, suffix_ids=suffix)

    def comment_safe_mask(self) -> torch.Tensor:
        mask = super().comment_safe_mask()
        mask[_STOI["<pad>"]] = False
        return mask

    # Convenience for scoring: the vocab id that represents each label's answer token.
    def label_token_id(self, label: str) -> int:
        return self._label_ids[self._labels.index(label)]
