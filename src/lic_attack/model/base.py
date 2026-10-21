"""Device-agnostic detector-model interface for white-box license attacks.

The attack needs three things a normal generate() API does not expose:
  1. the input-embedding matrix E (to turn a one-hot over the vocab into an
     embedding and get gradients back onto the one-hot),
  2. a forward pass that accepts ``inputs_embeds`` (so the mutable comment can be
     a differentiable one-hot @ E spliced between frozen prefix/suffix embeds),
  3. a "comment-safe" vocab mask (candidate tokens must decode to text that is
     legal inside a source-code comment: printable, single-line, no specials).

Two implementations share this interface:
  * HFDetector  — a real HF causal LM (Qwen2.5-Coder-7B by default), for the A100.
  * ToyDetector — a tiny deterministic offline model, so the whole pipeline can be
     validated on a CPU with no model download.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class PromptSegments:
    """Token ids around the mutable comment region.

    Full input = prefix_ids ++ comment_ids ++ suffix_ids, with label tokens
    appended after suffix_ids for teacher-forced scoring. Only comment_ids is
    optimized; prefix/suffix (system prompt, license text, source, question,
    assistant header) are frozen.
    """

    prefix_ids: torch.Tensor  # [P]
    suffix_ids: torch.Tensor  # [S]


class DetectorModel(ABC):
    """Minimal white-box surface the GCG attack depends on."""

    # --- properties every backend must provide ---
    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def embedding_matrix(self) -> torch.Tensor:
        """The input token-embedding weight, shape [vocab_size, hidden]."""

    # --- tokenization ---
    @abstractmethod
    def encode(self, text: str, add_special: bool = False) -> torch.Tensor:
        """Text -> 1-D LongTensor of ids (no batch dim)."""

    @abstractmethod
    def decode(self, ids: torch.Tensor) -> str: ...

    # --- forward ---
    @abstractmethod
    def logits_from_embeds(self, inputs_embeds: torch.Tensor,
                           logits_to_keep: int = 0) -> torch.Tensor:
        """inputs_embeds [B, T, H] -> logits.

        logits_to_keep == 0 returns logits for every position ([B, T, V]); a
        positive value returns only the last ``logits_to_keep`` positions
        ([B, logits_to_keep, V]). The discrete batched scorer needs only the
        label positions, so it passes a small value to avoid materializing a
        [B, T, V] tensor (which for a 152k vocab is tens of GB at batch 128)."""

    # --- prompt assembly (handles chat template where relevant) ---
    @abstractmethod
    def build_segments(self, source_before: str, source_after: str, question: str) -> PromptSegments:
        """Render the full prompt with the comment region split out.

        source_before / source_after are the file text on either side of the
        mutable comment; question is the closed-set license-ID instruction.
        """

    # --- comment-safe candidate vocabulary ---
    def comment_safe_mask(self) -> torch.Tensor:
        """Bool mask [vocab_size]; True = token may appear in the comment.

        Default policy: token decodes to a non-empty string of printable chars
        with no newline/carriage-return and no replacement char. Backends may
        override to also drop special/added tokens.
        """
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for tid in range(self.vocab_size):
            s = self.decode(torch.tensor([tid]))
            if not s:
                continue
            if "\n" in s or "\r" in s or "�" in s:
                continue
            if all(32 <= ord(ch) <= 126 or ord(ch) > 160 for ch in s):
                mask[tid] = True
        return mask

    def embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for ids [..., T] -> [..., T, H] (differentiable)."""
        # Coerce to long: empty-string tokenization can yield a float tensor, and
        # embedding indexing requires an integer dtype.
        return torch.nn.functional.embedding(ids.to(device=self.device, dtype=torch.long),
                                             self.embedding_matrix)
