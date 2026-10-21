#!/usr/bin/env python3
"""Forced-choice logprob transfer eval on a LOCAL open model (e.g. Qwen3-Coder-30B).

Black-box transfer in the attacker-capability sense (the washed comments were crafted on a
small white-box model; this target never gave gradients), but scored with the SAME
teacher-forced logprob / forced-choice protocol as the white-box eval -- so there is NO
logprob-vs-parse mismatch. Forward-only (set LIC_INFER_ONLY=1): no backward, so a 30B fits
the 8-card pipeline without OOM. This is eval-only, one score per row, not a 200-step search.

Reads washed comments (still_flips=True on the source model) from outputs/m3_rewrite_v2/*.jsonl
and asks: does the same comment, on the same (body-truncated) file, also flip THIS target?

  ident : score over the 19-label closed set (18 + NOASSERTION), flip = argmax != true license
  clause: score 3-way {Yes, No, Unknown}, flip = argmax != truth
  gate  : the target must be clean-correct on that (file[,probe]) or the flip is unattributable

Usage:
  LIC_INFER_ONLY=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/transfer_local.py \\
    --config configs/m3_qwen.yaml --target clause \\
    --wash-glob 'outputs/m3_rewrite_v2/llama_clause.r*.jsonl' --source-tag llama \\
    --out outputs/transfer_qwen3_30b/llama_clause.jsonl
"""
from __future__ import annotations

import argparse, glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lic_attack.data import CLAUSE_PROBES, build_samples_from_dir, identification_question
from lic_attack.eval import evaluate_comment
from lic_attack.model import build_model

YN3 = ["Yes", "No", "Unknown"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="target-model YAML (model.name = the 30B)")
    ap.add_argument("--target", choices=["ident", "clause"], required=True)
    ap.add_argument("--wash-glob", required=True, help="glob of source wash JSONL (r0/r1)")
    ap.add_argument("--source-tag", required=True, help="label for the source model, e.g. llama")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max-body-chars", type=int, default=1200,
                    help="same body truncation the wash used, so the file rendering is identical")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(ROOT / args.config))
    model_path = cfg["model"]["name"]
    dataset_dir = cfg["task"].get("dataset_dir", "data/licenses_by_name")

    gpu_list = [int(g) for g in args.gpus.split(",")]
    model_cfg = dict(cfg["model"], device=f"cuda:{gpu_list[0]}" if len(gpu_list) == 1 else "auto")

    if args.target == "ident":
        labels = cfg["task"]["labels"]
        question = identification_question("open_ended", labels)
    else:
        labels = YN3
        probes = {p.key: p for p in CLAUSE_PROBES}

    eval_licenses = cfg["task"]["labels"] if args.target == "ident" else None
    samples = {s.name: s for s in build_samples_from_dir(
        str(ROOT / dataset_dir), eval_licenses=eval_licenses)}
    if args.max_body_chars > 0:
        for s in samples.values():
            if len(s.source_after) > args.max_body_chars:
                s.source_after = s.source_after[:args.max_body_chars]

    # Source rows to transfer: only those that still flip on the source model.
    rows = []
    for f in sorted(glob.glob(str(ROOT / args.wash_glob))):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get("still_flips"):
                    rows.append(r)
    rows.sort(key=lambda r: (r["name"], r.get("probe", "")))
    print(f"[transfer] {len(rows)} still-flip source rows from {args.wash_glob}, "
          f"target={model_path.split('/')[-1]}, task={args.target}", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for l in open(out_path):
            if l.strip():
                d = json.loads(l); done.add((d["name"], d.get("probe", "")))
    if done:
        print(f"[resume] {len(done)} already scored; skipping", flush=True)

    print(f"[transfer] loading target={model_path} labels={len(labels)}", flush=True)
    model = build_model(model_cfg, labels)
    empty_ids = model.encode("", add_special=False)

    clean_cache: dict = {}   # (name, probe) -> (clean_argmax, clean_margin)  [comment-independent]
    n = n_gated = n_flip = 0
    with open(out_path, "a", buffering=1) as outf:
        for r in rows:
            probe_key = r.get("probe", "")
            if (r["name"], probe_key) in done:
                continue
            sample = samples.get(r["name"])
            if sample is None:
                continue
            if args.target == "ident":
                true_label = r["license"]; q = question
            else:
                probe = probes.get(probe_key)
                if probe is None:
                    continue
                true_label = r["truth"]; q = probe.question

            seg = model.build_segments(sample.source_before, sample.source_after, q)

            ck = (r["name"], probe_key)
            if ck not in clean_cache:
                csc = evaluate_comment(model, seg, empty_ids, labels, true_label, "clean")
                clean_cache[ck] = (csc.scores.argmax_label(), float(csc.scores.margin(true_label)))
            clean_arg, clean_margin = clean_cache[ck]
            clean_ok = (clean_arg == true_label)

            n += 1
            comment_ids = model.encode(r["washed_text"], add_special=False)
            asc = evaluate_comment(model, seg, comment_ids, labels, true_label, "attacked")
            att_arg = asc.scores.argmax_label()
            att_margin = float(asc.scores.margin(true_label))
            # A flip only counts where the target was clean-correct (attributable to the comment).
            flip = clean_ok and (att_arg != true_label)
            n_gated += int(clean_ok)
            n_flip += int(flip)

            outf.write(json.dumps({
                "name": r["name"], "license": r["license"], "probe": probe_key,
                "truth": r.get("truth", ""), "source": args.source_tag,
                "clean_argmax": clean_arg, "clean_correct": clean_ok, "clean_margin": clean_margin,
                "attacked_argmax": att_arg, "attacked_margin": att_margin,
                "transfer_flip": flip, "washed_text": r["washed_text"][:200],
            }) + "\n")
            tag = r["license"] if args.target == "ident" else f"{r['license']}/{probe_key}"
            print(f"[{tag:30}] clean={clean_arg}({'ok' if clean_ok else 'WRONG'}) "
                  f"att={att_arg} flip={flip}", flush=True)

    print(f"\n=== {args.source_tag}->{model_path.split('/')[-1]} {args.target}: "
          f"{n} rows, clean-correct {n_gated}, transfer-flip {n_flip}/{n_gated} "
          f"({100*n_flip//n_gated if n_gated else 0}% of gated) ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
