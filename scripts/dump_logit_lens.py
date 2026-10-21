#!/usr/bin/env python3
"""Logit-lens dump: per-LAYER closed-set label distribution, clean vs attacked.

Motivation. Every number in the paper reads the detector's decision at the LAST
layer (argmax / margin). That says the attack works, never where inside the model
it works. This dump replays already-attacked samples and reads the same closed-set
label scores out of EVERY layer, by projecting each layer's hidden state through
the final norm + LM head (the "logit lens"). The resulting trajectory shows the
layer at which the injected comment overtakes the file's real licence header.

Readout. The 19 ident labels do NOT have distinct first tokens ("GPL-2.0",
"GPL-2.0-classpath" and "GPL-3.0" all start with the token 'GPL'), so a
first-token lens would be degenerate. We instead reproduce the paper's scorer
per layer: teacher-force the whole label and reduce it exactly as
scoring.score_labels_from_embeds does -- PMI-calibrated for ident, the
length-normalised mean for clause. Layer L-1 therefore reproduces the paper's own
numbers, which the assert in main() checks on the first sample.

Conditions replayed per sample:
  clean    empty comment (the file's own licence header only)
  gcg      the GCG suffix     (unreadable, white-box)
  rewrite  the M3-rewritten comment (deniable prose)

The two run directories are arguments rather than constants: they were hard-coded to
the first pair that existed (outputs/gcg_m3, outputs/m3_rewrite_v2), and when the runs
moved to gcg_m3_cap15_pmi / m3_rewrite_v3 the script silently found zero attacked
comments, planned zero forwards, and wrote four empty files without failing.

Usage (one 8B per card):
  CUDA_VISIBLE_DEVICES=0 python -u scripts/dump_logit_lens.py \
    --config configs/m3_llama.yaml --model-key llama --target ident \
    --out outputs/logit_lens/llama_ident.jsonl

Each output row:
  {model, target, name, license, truth, probe, cond, still_flips, labels,
   layer_logprobs: [n_layers+1][n_labels]}   # index 0 = embedding output
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lic_attack.data import CLAUSE_PROBES, build_samples_from_dir, identification_question
from lic_attack.model import build_model

YN3 = ["Yes", "No", "Unknown"]


def _lens_parts(hf):
    """Return (decoder, final_norm, lm_head) for a HF causal-LM wrapper.

    The logit lens needs the two modules that sit between the last decoder block
    and the vocabulary: the final RMSNorm and the unembedding. Llama/Qwen/DeepSeek
    all expose them as model.model.norm and model.lm_head.
    """
    inner = hf.model.model
    norm = getattr(inner, "norm", None) or getattr(inner, "final_layernorm")
    head = hf.model.get_output_embeddings()
    return inner, norm, head


@torch.no_grad()
def layerwise_label_logprobs(hf, seg, comment_ids, label_token_ids, calib=None):
    """[n_hidden][n_labels] length-normalised mean per-token logprob, per layer.

    One forward per label (the labels are teacher-forced, so their token sequences
    differ and cannot share a single pass). Only the trailing positions that carry
    label-token predictions are projected through the head, which keeps the [N, V]
    logits tiny even though V ~ 128k.
    """
    _, norm, head = _lens_parts(hf)
    dev = seg.prefix_ids.device
    base = torch.cat([seg.prefix_ids, comment_ids.to(dev), seg.suffix_ids])

    per_label = []
    for lbl_ids in label_token_ids:
        lbl_ids = lbl_ids.to(dev)
        l = len(lbl_ids)
        full = torch.cat([base, lbl_ids]).unsqueeze(0)
        # use_cache=True is not an optimisation here (single pass, nothing reused):
        # it selects the same attention kernel the paper's scorer runs under, and
        # only then does the last layer of the lens match its scores bit for bit.
        # logits_to_keep stops the model projecting its own [1, T, 128k] logits.
        out = hf.model(input_ids=full, output_hidden_states=True,
                       use_cache=True, logits_to_keep=l + 1)
        # Label token j is predicted at position len(base)+j-1; keeping the last
        # l+1 positions puts that predictor at local index j.
        keep = l + 1
        scores = []
        last = len(out.hidden_states) - 1
        for i, h in enumerate(out.hidden_states):         # embeddings + every block
            tail = h[0, -keep:, :]                       # [keep, d]
            # HF appends the FINAL hidden state after self.norm has run, so only
            # entries 0..last-1 still need the lens to apply it. Norming twice
            # shifts the last layer off the paper's own scores (caught by the
            # assert in main()).
            logits = head(tail if i == last else norm(tail)).float()   # [keep, V]
            logp = torch.log_softmax(logits, dim=-1)
            tot = sum(logp[j, tok] for j, tok in enumerate(lbl_ids.tolist()))
            # Must match the rule the run being replayed was scored under. ident is
            # PMI-calibrated; reading the layers back under the length-normalised mean
            # instead silently mixes two scoring rules -- sample selection from one,
            # per-layer readout from the other -- and drags the clean final-layer
            # accuracy down to ~55% on samples that were clean-correct by construction.
            scores.append(float(tot) - float(calib[len(per_label)])
                          if calib is not None else float(tot) / max(1, l))
        per_label.append(scores)
        del out
    # per_label is [n_labels][n_hidden] -> transpose to [n_hidden][n_labels]
    return [list(col) for col in zip(*per_label)]


def _load_attacked(pattern, key_fn):
    """Index attacked comments from a run directory: key -> (text, still_flips)."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            txt = r.get("washed_text") or ""
            if not txt or not r.get("still_flips"):
                continue            # only replay attacks that actually flipped
            out.setdefault(key_fn(r), (txt, True))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-key", required=True,
                    help="prefix of the outputs/{gcg_m3,m3_rewrite_v2} filenames")
    ap.add_argument("--target", choices=["ident", "clause"], default="ident")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-license", type=int, default=15)
    ap.add_argument("--max-body-chars", type=int, default=1200)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = no cap")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--gcg-dir", default="outputs/gcg_m3_cap15_pmi",
                    help="directory holding <model>_<target>.r*.jsonl from run_gcg_m3")
    ap.add_argument("--rewrite-dir", default="outputs/m3_rewrite_v3",
                    help="directory holding <model>_<target>.r*.jsonl from run_m3_rewrite")
    ap.add_argument("--calibration", choices=["none", "pmi"], default="pmi",
                    help="must match the run being replayed: ident uses pmi, clause none")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(ROOT / args.config))
    labels = cfg["task"]["labels"] if args.target == "ident" else YN3
    dataset_dir = cfg["task"].get("dataset_dir", "data/licenses_by_name")

    gpu_list = [int(g) for g in args.gpus.split(",")]
    model_cfg = dict(cfg["model"], device=f"cuda:{gpu_list[0]}" if len(gpu_list) == 1 else "auto")
    model = build_model(model_cfg, labels)
    label_token_ids = [model.encode(l, add_special=False) for l in labels]

    calib = None
    if args.calibration == "pmi":
        if args.target != "ident":
            raise SystemExit("--calibration pmi is ident-only")
        from lic_attack.attack import null_calibration
        calib = null_calibration(model, identification_question("open_ended", labels),
                                 labels, label_token_ids)
        print(f"[calib] PMI reference over {len(calib)} labels", flush=True)

    samples = build_samples_from_dir(
        str(ROOT / dataset_dir),
        eval_licenses=cfg["task"]["labels"] if args.target == "ident" else None,
        max_per_license=args.max_per_license,
    )
    for s in samples:
        if args.max_body_chars > 0 and len(s.source_after) > args.max_body_chars:
            s.source_after = s.source_after[:args.max_body_chars]
    by_name = {s.name: s for s in samples}

    mk = args.model_key
    if args.target == "ident":
        keyf = lambda r: r["name"]
    else:
        keyf = lambda r: f'{r["name"]}||{r["probe"]}'
    gcg = _load_attacked(f"{args.gcg_dir}/{mk}_{args.target}.r*.jsonl", keyf)
    rew = _load_attacked(f"{args.rewrite_dir}/{mk}_{args.target}.r*.jsonl", keyf)
    print(f"[lens] attacked comments: gcg={len(gcg)} rewrite={len(rew)}", flush=True)
    if not gcg and not rew:
        # Writing an empty output here looks like a completed stage. Fail instead.
        raise SystemExit(
            f"no attacked comments for {mk}/{args.target} under {args.gcg_dir} or "
            f"{args.rewrite_dir} -- check the directories, not the model")

    # One work item per (sample-or-pair, condition). clean is replayed once per
    # sample so every attacked trajectory has its own baseline to be compared to.
    work = []
    seen_clean = set()
    units = sorted(set(gcg) | set(rew))
    if args.max_samples:
        units = units[:args.max_samples]
    for key in units:
        name, probe = (key.split("||", 1) + [""])[:2] if "||" in key else (key, "")
        if name not in by_name:
            continue
        if key not in seen_clean:
            work.append((key, name, probe, "clean", "", True))
            seen_clean.add(key)
        if key in gcg:
            work.append((key, name, probe, "gcg", gcg[key][0], True))
        if key in rew:
            work.append((key, name, probe, "rewrite", rew[key][0], True))

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                done.add((r["name"], r.get("probe", ""), r["cond"]))
    print(f"[lens] {len(work)} forwards planned, {len(done)} already done", flush=True)

    probe_by_key = {p.key: p for p in CLAUSE_PROBES}
    question_ident = identification_question("open_ended", labels)
    checked_last_layer = False

    with open(out_path, "a", buffering=1) as f:
        for key, name, probe, cond, text, _ in tqdm(work, desc=f"lens-{args.target}", unit="fwd"):
            if (name, probe, cond) in done:
                continue
            s = by_name[name]
            if args.target == "ident":
                q, truth = question_ident, s.license_id
            else:
                p = probe_by_key[probe]
                q = p.question
                truth = p.truth.get(s.license_id) or "Unknown"
            seg = model.build_segments(s.source_before, s.source_after, q)
            cids = model.encode(text, add_special=False) if text else model.encode("", add_special=False)
            layers = layerwise_label_logprobs(model, seg, cids, label_token_ids, calib)

            if not checked_last_layer:
                # The last hidden state IS what the LM head sees, so the lens must
                # reproduce the paper's own scorer there. Guard against a wrong
                # norm/head pick silently producing plausible-looking curves.
                from lic_attack.attack.scoring import score_labels_from_embeds
                ref = score_labels_from_embeds(model, seg, cids, labels, label_token_ids, calib)
                err = max(abs(a - b) for a, b in zip(layers[-1], ref.logprobs.tolist()))
                print(f"[lens] last-layer vs paper scorer: max abs diff = {err:.2e}", flush=True)
                assert err < 1e-2, "logit lens does not reproduce the last-layer scorer"
                checked_last_layer = True

            f.write(json.dumps({
                "model": mk, "target": args.target, "name": name, "probe": probe,
                "license": s.license_id, "truth": truth, "cond": cond,
                "labels": labels, "layer_logprobs": layers,
            }) + "\n")
    print(f"[lens] wrote {out_path}")
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main())
