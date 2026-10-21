"""Hugging Face causal-LM backend (default: Qwen3-Coder-30B-A3B-Instruct).

Device-agnostic: on multiple GPUs it shards the model with device_map="auto"
(needed for the 30B MoE across 2x A100); on a single GPU it loads to cuda; off-GPU
it falls back to cpu+fp32 so the same code path runs (slowly) on a laptop.

Weights are downloaded automatically from ModelScope when ``name`` is a repo id
(e.g. "Qwen/Qwen3-Coder-30B-A3B-Instruct") rather than a local directory — handy
for mainland China cloud hosts where the HF Hub is slow/blocked.
"""
from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import DetectorModel, PromptSegments


def _resolve_dtype(dtype: str, on_gpu: bool) -> torch.dtype:
    if not on_gpu:
        # bf16/fp16 matmul on CPU is slow and flaky; force fp32 off-GPU.
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]


def _maybe_modelscope_download(name: str) -> str:
    """If ``name`` is a repo id (not an existing local dir), fetch it from ModelScope
    and return the local snapshot path. If it's already a local path, return as-is.
    Falls back to returning ``name`` unchanged if modelscope isn't installed (so the
    HF Hub path still works)."""
    if os.path.isdir(name):
        return name
    try:
        from modelscope import snapshot_download
    except Exception:
        # modelscope not available — let from_pretrained try the HF Hub instead.
        return name
    print(f"[modelscope] downloading {name} ...")
    path = snapshot_download(name)
    print(f"[modelscope] cached at {path}")
    return path


class HFDetector(DetectorModel):
    def __init__(self, name: str, dtype: str = "bfloat16", device: str = "auto",
                 trust_remote_code: bool = False):
        n_gpu = torch.cuda.device_count()
        on_gpu = n_gpu > 0
        self._dtype = _resolve_dtype(dtype, on_gpu)

        local = _maybe_modelscope_download(name)
        self.tokenizer = AutoTokenizer.from_pretrained(local, trust_remote_code=trust_remote_code)

        load_kwargs = dict(torch_dtype=self._dtype, trust_remote_code=trust_remote_code)
        # Attention backend. Under device_map sharding the SDPA path takes a different
        # branch with vs without KV cache (is_causal fast-path vs explicit mask), and
        # the explicit mask can be built wrong across shards — which corrupts cached
        # trunk logits. "eager" forces one consistent masked path so use_cache=True and
        # use_cache=False agree numerically (needed for the KV-cache candidate scorer).
        # Override with LIC_ATTN_IMPL (sdpa|eager|flash_attention_2).
        attn_impl = os.environ.get("LIC_ATTN_IMPL")
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
        if n_gpu > 1 and device in ("auto", "cuda"):
            # shard across all visible GPUs (needed for 30B across 2x A100)
            load_kwargs["device_map"] = "auto"
            # Skew the weight split so GPU0 holds LESS than an even share. The GCG
            # gradient step runs forward+backward activations on GPU0 (embeddings,
            # the first layers, and the LM head all live there); an even ~30.5GB
            # weight split leaves too little headroom and OOMs a 48GB card during
            # the forward pass. We cap GPU0's weights low and let the other GPU(s)
            # carry more, freeing GPU0 memory for activations and the 152k-vocab
            # logits. Overridable via LIC_MAX_MEMORY="22GiB,46GiB" (per-GPU, GPU0 first).
            env_mm = os.environ.get("LIC_MAX_MEMORY")
            if env_mm:
                caps = [c.strip() for c in env_mm.split(",")]
                max_memory = {i: caps[i] for i in range(min(n_gpu, len(caps)))}
            else:
                # Near-even split, nudged so GPU0 holds slightly LESS than half.
                # An exactly-even split (~30.5GiB each, ~17GiB free) OOMed GPU0 by
                # only ~20MB during the forward; skewing weights hard to GPU1 instead
                # OOMs GPU1. The sweet spot is near-even with a small bias: GPU0 also
                # carries embeddings + LM head + activations, so give it ~4GiB more
                # headroom than GPU1.
                if n_gpu == 2:
                    max_memory = {0: "29GiB", 1: "33GiB"}
                else:
                    max_memory = {i: "30GiB" for i in range(n_gpu)}
            load_kwargs["max_memory"] = max_memory
            print(f"[shard] max_memory per GPU: {max_memory}")
            self.model = AutoModelForCausalLM.from_pretrained(local, **load_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(local, **load_kwargs)
            target = ("cuda" if (on_gpu and device in ("auto", "cuda")) else
                      (device if device not in ("auto",) else "cpu"))
            self.model = self.model.to(target)

        self.model.eval()
        self.model.requires_grad_(False)  # we differentiate wrt inputs, never weights

        # Gradient checkpointing: the GCG gradient step backprops through the 30B
        # model; storing every layer's ~600-token activations alongside ~30GB of
        # sharded weights OOMs a 48GB card. Checkpointing recomputes activations in
        # the backward pass instead of storing them, trading ~1.5-2x compute for a
        # large memory saving. use_reentrant=False is required here: it works with
        # inputs_embeds-only grad (weights are frozen) and with device_map sharding.
        # Inference-only mode (LIC_INFER_ONLY=1): the non-GCG suite (clean/random/
        # fake_spdx/handcrafted/llm scoring) never takes gradients, so gradient
        # checkpointing — which recomputes activations every forward to save backward
        # memory — is pure overhead there, and use_cache=False needlessly disables the
        # KV cache. For a pure-forward run we flip both: ~1.5-2x faster scoring, and
        # numerically identical (it only changes memory/compute scheduling, not math).
        # Leave it OFF (default) for any run that uses GCG (stage1.use_gcg) or minimize,
        # which DO need the backward pass.
        infer_only = os.environ.get("LIC_INFER_ONLY") == "1"
        if infer_only:
            self.model.config.use_cache = True
            self._grad_ckpt = False
            print("[infer-only] gradient checkpointing OFF, KV cache ON "
                  "(no-grad scoring path — do NOT use with GCG/minimize)")
        else:
            self.model.config.use_cache = False
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                self._grad_ckpt = True
            except Exception as e:
                print(f"[warn] gradient_checkpointing_enable failed ({e}); continuing without it")
                self._grad_ckpt = False

        self._embed = self.model.get_input_embeddings().weight  # [V, H]
        # With device_map sharding, the embedding layer lives on one specific GPU;
        # inputs_embeds must be placed there. Read the actual weight device rather
        # than assuming a single global device.
        self._device = self._embed.device

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def vocab_size(self) -> int:
        return self._embed.shape[0]

    @property
    def embedding_matrix(self) -> torch.Tensor:
        return self._embed

    def encode(self, text: str, add_special: bool = False) -> torch.Tensor:
        ids = self.tokenizer(text, add_special_tokens=add_special, return_tensors="pt").input_ids[0]
        # An empty string yields an empty tensor that PyTorch types as float; force long
        # so concatenation/embedding indexing never gets a FloatTensor.
        return ids.to(device=self._device, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids.tolist(), skip_special_tokens=False)

    def logits_from_embeds(self, inputs_embeds: torch.Tensor,
                           logits_to_keep: int = 0) -> torch.Tensor:
        # Under device_map sharding the embedding layer (input side) lives on
        # self._device; inputs_embeds must start there. transformers moves the
        # hidden state across shards internally. We bring logits back to
        # self._device so all downstream margin/softmax math stays on one device.
        #
        # logits_to_keep>0 makes the LM head project only the last N positions,
        # so a batched [B, T, V] logits tensor (tens of GB at B=128, V=152k) is
        # never materialized — only [B, N, V] is.
        out = self.model(inputs_embeds=inputs_embeds.to(device=self._device, dtype=self._dtype),
                         logits_to_keep=logits_to_keep)
        return out.logits.float().to(self._device)

    def build_segments(self, source_before: str, source_after: str, question: str) -> PromptSegments:
        # The mutable comment sits between source_before and source_after inside a
        # user turn that shows the file then asks the closed-set question.
        # We render the chat template in two halves so the comment region is a clean
        # token boundary: everything up to the comment is prefix, the rest is suffix.
        sentinel = "<<<COMMENT_SLOT>>>"
        user_content = (
            "Here is a source file:\n\n```\n"
            + source_before + sentinel + source_after
            + "\n```\n\n" + question
        )
        full = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )
        if sentinel not in full:
            raise RuntimeError("comment sentinel was lost during chat templating")
        before_text, after_text = full.split(sentinel, 1)
        prefix_ids = self.tokenizer(before_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        suffix_ids = self.tokenizer(after_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        return PromptSegments(prefix_ids=prefix_ids.to(self._device), suffix_ids=suffix_ids.to(self._device))

    def comment_safe_mask(self) -> torch.Tensor:
        mask = super().comment_safe_mask()
        # Drop all special/added tokens (eos, im_start, etc.) — they must never
        # appear inside the comment.
        for tid in self.tokenizer.all_special_ids:
            if 0 <= tid < self.vocab_size:
                mask[tid] = False
        added = getattr(self.tokenizer, "added_tokens_decoder", {}) or {}
        for tid in added:
            if 0 <= tid < self.vocab_size:
                mask[tid] = False
        return mask
