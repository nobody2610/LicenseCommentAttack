#!/usr/bin/env python3
"""M3-lite washer runner for the closed-set pipeline.

Consumes the seed files produced by ``scripts/run_m3_seed.py``
(``outputs/m3_seeds_v2/{model}_{ident,clause}.jsonl``) and rewrites each flipping seed
with ``m3_rewrite_search`` -- simulated annealing under a "still flips" hard constraint,
minimizing a single forward-free license-semantic residue
(keyword-bag hits + embedding-similarity near-hits).  See
``src/lic_attack/attack/m3_wash.py`` for the objective.

Front-end only (loading seeds + samples + labels/questions); the search engine and the
residue are shared library code.  Mirrors run_m3_seed.py's config / dataset conventions:
  * samples come from ``data/licenses_by_name/<license>/<file>`` via build_samples_from_dir
  * ident scores over the 19-label closed set; clause scores 3-way {Yes, No, Unknown}

Usage:
  python scripts/run_m3_rewrite.py --config configs/m3_llama.yaml --target ident \\
    --seeds outputs/m3_seeds_v2/llama_ident.jsonl \\
    --out   outputs/m3_rewrite_v2/llama_ident.jsonl --gpus 0,1

Resume-safe: skips (name, probe) already in any replica_*.jsonl / the out file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lic_attack.attack import null_calibration, BagIndex, M3RewriteConfig, m3_rewrite_search
from lic_attack.data import (
    CLAUSE_PROBES,
    build_samples_from_dir,
    identification_question,
)
from lic_attack.model import build_model

YN3 = ["Yes", "No", "Unknown"]


def load_seeds(path: Path) -> list[dict]:
    seeds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    # Deterministic order so --replica sharding is stable.
    seeds.sort(key=lambda s: (s["name"], s.get("probe", "")))
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="YAML config (model.name, task.labels, task.dataset_dir)")
    ap.add_argument("--target", choices=["ident", "clause"], required=True)
    ap.add_argument("--seeds", required=True, help="seed JSONL from run_m3_seed.py")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("--gpus", default="0", help="comma-separated GPU indices")
    ap.add_argument("--replica", type=int, default=0)
    ap.add_argument("--n-replicas", type=int, default=1)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--tau", type=float, default=0.0,
                    help="hard constraint: keep only comments with margin < tau")
    ap.add_argument("--no-sim", action="store_true",
                    help="disable the embedding-similarity residue term (keyword-only)")
    ap.add_argument("--w-keyword", type=float, default=1.0)
    ap.add_argument("--w-sim", type=float, default=1.0)
    ap.add_argument("--sim-threshold", type=float, default=0.5)
    ap.add_argument("--max-body-chars", type=int, default=1200,
                    help="truncate the file BODY to this many chars before washing (the "
                         "license header is untouched). Bounds the backward-pass memory so "
                         "the 8B does not OOM on long source files; the clause/ident judgement "
                         "reads the header + injected comment, so the body is near-irrelevant. "
                         "0 = no truncation.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--early-stop-s", type=float, default=None,
                    help="gate the stall-stop on the best residue reaching this value, "
                         "so a plateau above it keeps searching to the end of the schedule")
    ap.add_argument("--drop-op", choices=["delete", "substitute", "insert"], default=None,
                    help="ablation: zero this operator's proposal probability and "
                         "renormalise the other two, so the run exercises the rewriter "
                         "minus one operator on the same seeds")
    ap.add_argument("--calibration", choices=["none", "pmi"], default="none",
                    help="label-score comparison rule. ident uses 'pmi' because its 19 labels differ in token length (MIT/ISC 1 token vs LGPL-2.1 6), which the length-normalised mean turns into a systematic bias; clause uses 'none' because Yes/No/Unknown are all 1 token, so the two rules coincide there. MUST match whatever the GCG run used, or the gate and the search disagree.")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(ROOT / args.config))
    model_path = cfg["model"]["name"]
    dataset_dir = cfg["task"].get("dataset_dir", "data/licenses_by_name")

    gpu_list = [int(g) for g in args.gpus.split(",")]
    if len(gpu_list) == 1:
        model_cfg_override = dict(cfg["model"], device=f"cuda:{gpu_list[0]}")
    else:
        model_cfg_override = dict(cfg["model"], device="auto")

    if args.target == "ident":
        labels = cfg["task"]["labels"]
        question = identification_question("open_ended", labels)
    else:
        labels = YN3
        probes = {p.key: p for p in CLAUSE_PROBES}

    # Samples keyed by name (== seed["name"], e.g. "AGPL-3.0/03154bd27ee4").
    eval_licenses = cfg["task"]["labels"] if args.target == "ident" else None
    samples = {s.name: s for s in build_samples_from_dir(
        str(ROOT / dataset_dir), eval_licenses=eval_licenses)}

    # Truncate the file BODY (not the license header) to bound backward-pass memory. Long
    # source bodies OOM the 8B on a 24GB card during the gradient proposal; the license
    # header + injected comment carry the signal the clause/ident judgement reads, so cutting
    # the body tail leaves the flip essentially unchanged (seeds that DO break re-score as
    # still_flips=False via wash_search's own infeasible check).
    if args.max_body_chars > 0:
        n_trunc = 0
        for s in samples.values():
            if len(s.source_after) > args.max_body_chars:
                s.source_after = s.source_after[:args.max_body_chars]
                n_trunc += 1
        print(f"[m3-wash] truncated body of {n_trunc} files to <= {args.max_body_chars} chars",
              flush=True)

    seeds = load_seeds(ROOT / args.seeds)
    seeds = [s for i, s in enumerate(seeds) if i % args.n_replicas == args.replica]
    print(f"[m3-wash] {len(seeds)} seeds (replica {args.replica}/{args.n_replicas}), "
          f"target={args.target}, model={model_path.split('/')[-1]}", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in sorted(out_path.parent.glob(f"{out_path.stem}*.jsonl")):
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add((r["name"], r.get("probe", "")))
    if done:
        print(f"[resume] {len(done)} (name,probe) already done; skipping", flush=True)

    print(f"[m3-wash] loading model={model_path}  labels={len(labels)}", flush=True)
    model = build_model(model_cfg_override, labels)
    # Must match the calibration the seed stage and the GCG run used: the rewriter's
    # hard constraint is margin(cand) < tau under this same scorer, so a mismatch would
    # have it preserving a "flip" the evaluation does not agree is a flip.
    calib = None
    if args.calibration == "pmi":
        if args.target != "ident":
            # The six clause probes ask different questions, so they need six different
            # reference terms; one shared vector would calibrate five of them against
            # the wrong premise. clause is run with --calibration none by design (its
            # three labels are all 1 token, so mean and sum coincide and there is no
            # length bias to correct), so rather than half-implement it, refuse.
            raise SystemExit("--calibration pmi is ident-only; clause needs one "
                             "reference per probe and is run uncalibrated by design")
        calib = null_calibration(model, question, labels)
        print(f"[calib] PMI reference over {len(calib)} labels", flush=True)
    torch.manual_seed(args.seed)

    wc = M3RewriteConfig(
        tau=args.tau, n_steps=args.n_steps, seed=args.seed,
        use_sim=not args.no_sim, w_keyword=args.w_keyword, w_sim=args.w_sim,
        sim_threshold=args.sim_threshold, early_stop_S=args.early_stop_s)
    if args.drop_op:
        # Renormalise rather than leaving the probabilities summing to less than one:
        # wash_search samples with torch.multinomial, which would otherwise reweight
        # them implicitly and make the ablation a different search, not a smaller one.
        keep = {"delete": "p_delete", "substitute": "p_substitute", "insert": "p_insert"}
        setattr(wc, keep.pop(args.drop_op), 0.0)
        tot = sum(getattr(wc, f) for f in keep.values())
        for f in keep.values():
            setattr(wc, f, getattr(wc, f) / tot)
        print(f"[ablation] dropped {args.drop_op}; probs = "
              + ", ".join(f"{k}={getattr(wc, v):.3f}" for k, v in keep.items()), flush=True)
    bag = None if args.no_sim else BagIndex(model)
    print(f"[m3-wash] residue = {wc.w_keyword:g}*keyword"
          + ("" if args.no_sim else f" + {wc.w_sim:g}*sim(>{wc.sim_threshold:g})")
          + f"  tau={wc.tau:g}  n_steps={wc.n_steps}", flush=True)

    n = n_flip = 0
    with open(out_path, "a", buffering=1) as outf:
        for s in seeds:
            probe_key = s.get("probe", "")
            if (s["name"], probe_key) in done:
                continue
            sample = samples.get(s["name"])
            if sample is None:
                continue

            if args.target == "ident":
                true_label = s["license"]
                q = question
            else:
                probe = probes.get(probe_key)
                if probe is None:
                    continue
                true_label = s["truth"]
                q = probe.question

            seg = model.build_segments(sample.source_before, sample.source_after, q)
            seed_ids = model.encode(s["comment"], add_special=False).to(model.device)
            wr = m3_rewrite_search(model, seg, seed_ids, labels, true_label, wc, bag=bag, calib=calib)
            n += 1
            n_flip += int(wr.success)

            rec = {
                "name": s["name"], "license": s["license"], "probe": probe_key,
                "tgt_license": s.get("tgt_license", ""), "truth": s.get("truth", ""),
                "washer": "m3", "tau": wc.tau,
                "start_S": wr.start_S, "final_S": wr.final_S,
                "start_kw": wr.start_kw, "final_kw": wr.final_kw,
                "start_sim": wr.start_sim, "final_sim": wr.final_sim,
                "start_margin": wr.start_margin, "final_margin": wr.final_margin,
                "still_flips": wr.success, "steps": wr.steps_used,
                "seed_text": s["comment"][:200], "washed_text": wr.washed_text[:200],
            }
            outf.write(json.dumps(rec) + "\n")
            tag = s["license"] if args.target == "ident" else f"{s['license']}/{probe_key}"
            print(f"[{tag:34}] S {wr.start_S:.0f}->{wr.final_S:.0f} "
                  f"(kw {wr.start_kw}->{wr.final_kw}, sim {wr.start_sim}->{wr.final_sim}) "
                  f"m {wr.start_margin:.2f}->{wr.final_margin:.2f} flips={wr.success} "
                  f"| {wr.washed_text[:44]!r}", flush=True)

    print(f"\n=== {args.target} M3-wash: {n} seeds, still-flip {n_flip}/{n} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
