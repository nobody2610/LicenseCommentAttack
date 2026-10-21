#!/usr/bin/env python3
"""GCG attack under the NEW setup (18-license closed set / 19-label ident / 3-way clause).

The old GCG results were produced on the OLD setup (10-label ident, 2-way {Yes,No} clause,
the build_samples_from_jsonl manifest, the old truth table). This runner re-runs GCG on the
CURRENT setup so the white-box ASR and the black-box transfer numbers are comparable with the
M3-rewrite / transfer_local family:

  ident : 19-label closed set (18 + NOASSERTION), true_label = file's SPDX id
  clause: 3-way {Yes, No, Unknown}; truth None -> "Unknown" is ATTACKABLE (not skipped)
  gate  : only attack (file[,probe]) the white-box model is clean-correct on (attributable)

The GCG engine (run_gcg) is label-set agnostic and is NOT modified: it takes `labels` +
`true_label` and derives the target token ids internally. This runner is the outer wiring only
(dataset, label set, truth, clause 3-way), mirroring scripts/run_m3_seed.py and writing rows in
the SAME JSONL schema the transfer scripts read (still_flips + washed_text + name + license/truth
+ probe), so GCG output drops straight into transfer_local.py / ident_transfer_openai.py.

GCG needs the backward pass: LIC_INFER_ONLY MUST be unset (this runner hard-fails otherwise).

Usage (one 7-8B model per card, 8-way shard across 8 cards):
  CUDA_VISIBLE_DEVICES=0 python -u scripts/run_gcg_m3.py \\
    --config configs/m3_llama.yaml --target ident \\
    --shard-index 0 --num-shards 8 --max-per-license 15 \\
    --out outputs/gcg_m3/llama_ident.r0.jsonl

Each output row:
  {name, license, probe, truth, generator:"gcg", still_flips, seed_text, washed_text,
   final_margin, clean_argmax, clean_correct, steps_used}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lic_attack.data import CLAUSE_PROBES, build_samples_from_dir, identification_question
from lic_attack.attack import AttackConfig, null_calibration, run_gcg
from lic_attack.eval import evaluate_comment
from lic_attack.model import build_model

YN3 = ["Yes", "No", "Unknown"]


def _run_gcg_with_backoff(model, seg, labels, true_label, acfg, min_batch=8, calib=None):
    """run_gcg, halving the candidate batch on CUDA OOM instead of dying.

    The longest prompt in the dataset (a Linux kernel file whose GPL notice runs
    2000+ chars) OOMs at batch_size=32 on a 24GB card, and an uncaught OOM kills the
    whole shard -- one file cost a card 9 idle hours before this was noticed. Halving
    is preferable to skipping: skipping would silently drop exactly the files with the
    longest licence notices, i.e. the ones the detector finds easiest, which would
    bias ASR upward. The batch actually used is recorded per row so any deviation from
    the nominal search budget is visible in the results rather than implied.
    """
    import dataclasses
    batch = acfg.batch_size
    while True:
        try:
            return run_gcg(model, seg, labels, true_label,
                           dataclasses.replace(acfg, batch_size=batch), calib), batch
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= min_batch:
                raise
            batch //= 2
            print(f"[oom] retrying at batch_size={batch}", flush=True)


def _attack_and_record(model, seg, labels, true_label, acfg, empty_ids, meta, out_f, calib=None):
    """Clean-gate, then GCG. Write one row. Returns (attacked, flipped)."""
    clean_sc = evaluate_comment(model, seg, empty_ids, labels, true_label, "clean", calib)
    clean_arg = clean_sc.scores.argmax_label()
    clean_ok = (clean_arg == true_label)
    if not clean_ok:
        # not attributable -> record the gate outcome, do not run GCG
        out_f.write(json.dumps({**meta, "generator": "gcg", "still_flips": False,
                                "seed_text": "", "washed_text": "",
                                "final_margin": float(clean_sc.scores.margin(true_label)),
                                "clean_argmax": clean_arg, "clean_correct": False,
                                "steps_used": 0}) + "\n")
        return False, False

    try:
        res, batch_used = _run_gcg_with_backoff(model, seg, labels, true_label, acfg, calib=calib)
    except torch.OutOfMemoryError:
        # Backoff exhausted. The OOM is in objective_from_onehot's backward graph
        # (19 teacher-forced labels over a ~900-token prompt), not in the candidate
        # batch, so shrinking the batch cannot save it and gradient checkpointing is
        # already on. Record the file as skipped and keep the shard alive: losing one
        # sample beats losing the 20+ files queued behind it. Analysis MUST exclude
        # oom_skipped rows from the ASR denominator -- counting them as failed attacks
        # would understate ASR, counting them as attacked would be a fabrication.
        torch.cuda.empty_cache()
        print(f"[oom] giving up on {meta.get('name')}; recording as skipped", flush=True)
        out_f.write(json.dumps({**meta, "generator": "gcg", "still_flips": False,
                                "seed_text": "", "washed_text": "",
                                "final_margin": float(clean_sc.scores.margin(true_label)),
                                "clean_argmax": clean_arg, "clean_correct": True,
                                "steps_used": 0, "oom_skipped": True}) + "\n")
        return False, False

    flipped = bool(res.success)
    # Realizability. GCG optimises a TOKEN sequence, but the threat model has the
    # attacker append TEXT to a source file; the detector then re-tokenises whatever
    # was written. Those are not the same thing: many token sequences decode to the
    # same string and encode() returns only the canonical one, so a sequence the
    # search reached is often unreachable from text. On the runs made BEFORE the
    # invertibility filter was added to run_gcg, only 7-16% of accepted comments still
    # re-tokenised to their nominal length -- i.e. most of what was being optimised
    # could never have been deployed. With the filter in place this should hold for
    # every accepted comment; the field is recorded anyway so the claim is checkable
    # rather than assumed.
    text = res.final_comment_text
    reencoded = model.encode(text, add_special=False).cpu()
    realizable = bool(torch.equal(reencoded, res.final_comment_ids.cpu()))
    out_f.write(json.dumps({
        **meta, "generator": "gcg", "still_flips": flipped,
        "seed_text": res.final_comment_text, "washed_text": res.final_comment_text,
        "final_margin": float(res.final.margin(true_label)),
        "clean_argmax": clean_arg, "clean_correct": True,
        "steps_used": res.steps_used, "batch_used": batch_used,
        "realizable": realizable,
    }) + "\n")
    return True, flipped


def run_ident(model, samples, labels, acfg, out_path, done_keys, calib=None):
    question = identification_question("open_ended", labels)
    empty_ids = model.encode("", add_special=False)
    n = n_flip = 0
    with open(out_path, "a", buffering=1) as out_f:
        for s in tqdm(samples, desc="gcg-ident", unit="file"):
            if s.name in done_keys:
                continue
            seg = model.build_segments(s.source_before, s.source_after, question)
            meta = {"name": s.name, "license": s.license_id, "probe": "", "truth": s.license_id}
            att, flip = _attack_and_record(model, seg, labels, s.license_id, acfg, empty_ids, meta, out_f, calib)
            n += int(att); n_flip += int(flip)
            done_keys.add(s.name)
    print(f"[gcg-ident] attacked {n}, flipped {n_flip}/{n}")


def run_clause(model, samples, acfg, out_path, done_keys, calib=None):
    empty_ids = model.encode("", add_special=False)
    n = n_flip = 0
    with open(out_path, "a", buffering=1) as out_f:
        for s in tqdm(samples, desc="gcg-clause", unit="file"):
            for probe in CLAUSE_PROBES:
                pair_key = f"{s.name}||{probe.key}"
                if pair_key in done_keys:
                    continue
                truth = probe.truth.get(s.license_id)          # str or None
                truth_label = "Unknown" if truth is None else truth   # None IS attackable
                seg = model.build_segments(s.source_before, s.source_after, probe.question)
                meta = {"name": s.name, "license": s.license_id,
                        "probe": probe.key, "truth": truth_label}
                pc = calib[probe.key] if isinstance(calib, dict) else calib
                att, flip = _attack_and_record(model, seg, YN3, truth_label, acfg, empty_ids, meta, out_f, pc)
                n += int(att); n_flip += int(flip)
                done_keys.add(pair_key)
    print(f"[gcg-clause] attacked {n}, flipped {n_flip}/{n}")


class _PersistingKeys(set):
    """Resume: every processed key (clean-wrong / non-flip / flip alike) is persisted to a
    sidecar so restarts skip it. File-level for ident (s.name), pair-level for clause."""
    def __init__(self, path):
        super().__init__()
        self.path = path
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    if line.strip():
                        super().add(line.strip())
        self._f = open(self.path, "a", buffering=1)

    def add(self, key):
        if key not in self:
            super().add(key)
            self._f.write(key + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="white-box model YAML (configs/m3_*.yaml)")
    ap.add_argument("--target", choices=["ident", "clause"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--max-per-license", type=int, default=15,
                    help="cap files per license (per-license subsample; default 15)")
    ap.add_argument("--max-body-chars", type=int, default=1200,
                    help="body truncation matching the M3 / transfer runs")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gpus", default="0")
    # GCG hyperparams (match configs/default.yaml except margin_success)
    ap.add_argument("--comment-tokens", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--margin-success", type=float, default=-1.0,
                    help="keep optimizing until margin < this (confident flip; avoids the "
                         "boundary -0.006 garbage flips the old pool had)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibration", choices=["none", "pmi"], default="pmi",
                    help="how closed-set label scores are compared. 'none' is the "
                         "length-normalised mean the earlier runs used; it gives "
                         "single-token labels (MIT, ISC, Zlib) a 0%% clean accuracy "
                         "and so drops them from the ASR denominator entirely. 'pmi' "
                         "subtracts each label's score on a content-free prompt "
                         "(see scoring._reduce) and is the default.")
    args = ap.parse_args()

    if os.environ.get("LIC_INFER_ONLY") == "1":
        raise SystemExit("LIC_INFER_ONLY=1 is set but GCG needs the backward pass. Unset it.")

    import yaml
    cfg = yaml.safe_load(open(ROOT / args.config))
    model_path = cfg["model"]["name"]
    dataset_dir = args.dataset_dir or cfg["task"].get("dataset_dir", "data/licenses_by_name")
    labels = cfg["task"]["labels"]

    gpu_list = [int(g) for g in args.gpus.split(",")]
    model_cfg = dict(cfg["model"], device=f"cuda:{gpu_list[0]}" if len(gpu_list) == 1 else "auto")
    model_cfg["seed"] = args.seed

    samples = build_samples_from_dir(
        str(ROOT / dataset_dir),
        eval_licenses=labels if args.target == "ident" else None,
        max_per_license=args.max_per_license,
    )
    if args.max_body_chars > 0:
        for s in samples:
            if len(s.source_after) > args.max_body_chars:
                s.source_after = s.source_after[:args.max_body_chars]
    if args.num_shards > 1:
        samples = [s for i, s in enumerate(samples) if i % args.num_shards == args.shard_index]
    print(f"[gcg] {len(samples)} samples, shard {args.shard_index}/{args.num_shards}, "
          f"model={model_path.split('/')[-1]}, target={args.target}, "
          f"labels={len(labels) if args.target=='ident' else 3}", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = _PersistingKeys(out_path.with_suffix(out_path.suffix + ".processed"))
    print(f"[resume] {len(done_keys)} keys already processed", flush=True)

    torch.manual_seed(args.seed)
    model = build_model(model_cfg, labels)
    acfg = AttackConfig(objective=cfg["attack"].get("objective", "margin"),
                        comment_tokens=args.comment_tokens, top_k=args.top_k,
                        batch_size=args.batch_size, n_steps=args.n_steps,
                        margin_success=args.margin_success, seed=args.seed)

    # One forward pass, reused for every sample: the reference term depends only on
    # (model, question, label set), none of which vary within a run.
    # The reference term depends on (model, question, label set). ident asks one
    # question so one vector suffices; clause asks a different question per probe, so
    # reusing one probe's reference for all six would calibrate against the wrong
    # premise. Computed once up front either way -- it is a single forward per question.
    calib = None
    if args.calibration == "pmi":
        if args.target == "ident":
            calib = null_calibration(model, identification_question("open_ended", labels),
                                     labels)
            print(f"[calib] PMI reference over {len(calib)} ident labels", flush=True)
        else:
            calib = {p.key: null_calibration(model, p.question, YN3) for p in CLAUSE_PROBES}
            print(f"[calib] PMI reference for {len(calib)} clause probes", flush=True)

    if args.target == "ident":
        run_ident(model, samples, labels, acfg, out_path, done_keys, calib)
    else:
        run_clause(model, samples, acfg, out_path, done_keys, calib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
