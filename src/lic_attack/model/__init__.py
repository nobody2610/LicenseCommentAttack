"""Model backends for the license attack."""
import os
from pathlib import Path

from .base import DetectorModel, PromptSegments


def resolve_model_path(name: str) -> str:
    """Let one set of configs work on hosts that store the weights elsewhere.

    The configs hold absolute paths from the machine the runs were developed on. A
    container, or a second machine, will not have them. Rather than edit seven configs
    per host, LIC_MODEL_ROOT names a directory to look in when the configured path is
    absent; the basename is matched against it.

    Two spellings are tried because modelscope escapes dots in its cache directory
    names -- `Qwen2.5-Coder-7B-Instruct` is stored as `Qwen2___5-Coder-7B-Instruct` --
    so weights copied to a fresh host are usually given the readable name back.

    An existing path always wins, so setting LIC_MODEL_ROOT cannot change what a host
    that already resolves its configs is running.
    """
    if os.path.exists(name):
        return name
    root = os.environ.get("LIC_MODEL_ROOT")
    if not root:
        return name
    base = Path(name).name
    for cand in (base, base.replace("___", ".")):
        p = Path(root) / cand
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"model {name!r} not found, and no match for {base!r} under "
        f"LIC_MODEL_ROOT={root!r}. Mount the weights there or fix model.name."
    )


def build_model(cfg: dict, labels: list[str]):
    """Construct a DetectorModel from the ``model`` section of a config dict."""
    kind = cfg.get("kind", "hf")
    if kind == "toy":
        from .toy_detector import ToyDetector
        return ToyDetector(labels=labels, seed=cfg.get("seed", 0))
    if kind == "hf":
        from .hf_detector import HFDetector
        return HFDetector(
            name=resolve_model_path(cfg["name"]),
            dtype=cfg.get("dtype", "bfloat16"),
            device=cfg.get("device", "auto"),
            trust_remote_code=cfg.get("trust_remote_code", False),
        )
    raise ValueError(f"unknown model.kind: {kind!r}")


__all__ = ["DetectorModel", "PromptSegments", "build_model", "resolve_model_path"]
