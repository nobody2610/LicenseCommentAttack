#!/usr/bin/env python3
"""M3 seed-finding stage: evaluate instantiated comment templates on a white-box model.

This is the simplified seed-finding pass designed for the M3 experiment.  It does
one thing: for every (file, template) pair, instantiate the template with the file's
real license and a target license from the closed set, evaluate the margin on the
white-box model, and record rows that are (a) clean-correct and (b) flipped.

What it does NOT do (compare run_attack.py):
  - No random/fake_spdx/handcrafted/GCG candidate arms
  - No deletion minimization
  - No clause collateral evaluation during this stage

The output is a JSONL seed file consumed by the M3 rewriter (run_m4_wash.py --washer m3).

Usage:
  python scripts/run_m3_seed.py \\
    --model-path /path/to/model \\
    --templates data/comment_templates.json \\
    --dataset-dir data/licenses_by_name \\
    --labels-config configs/llama_ident_expanded.yaml \\
    --out outputs/m3_seeds_ident/seeds.jsonl \\
    --target ident                         # ident | clause
    --gpus 0,1,2,3                         # GPU indices for this shard
    --shard-index 0 --num-shards 1         # for multi-GPU sharding

Each output row:
  {
    "name":        str,   # relative file path
    "license":     str,   # source file's SPDX id
    "tgt_license": str,   # attack target SPDX id (ident only)
    "probe":       str,   # probe key (clause only, empty for ident)
    "template":    str,   # the raw template string (with {placeholders})
    "comment":     str,   # the instantiated comment text
    "gen_margin":  float, # margin after inserting comment (negative = flipped)
    "clean_margin":float, # margin on the clean file (positive = was correct)
    "generator":   str,   # always "template"
  }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lic_attack.data import (
    CLAUSE_PROBES,
    build_samples_from_dir,
)
from lic_attack.data.licenses import CLOSED_SET
from lic_attack.attack import null_calibration
from lic_attack.eval import evaluate_comment, best_comment_by_margin
from lic_attack.model import build_model


# ── Template instantiation ────────────────────────────────────────────────────

COPYLEFT_SET = {"GPL-2.0","GPL-2.0-classpath","GPL-3.0","AGPL-3.0","LGPL-2.1","LGPL-3.0",
                "MPL-2.0","EPL-2.0","CDDL-1.0","SSPL-1.0","OSL-3.0","Artistic-2.0","BSL-1.1"}
PERMISSIVE_SET = {"MIT","ISC","Apache-2.0","BSD-3-Clause","Unlicense"}


def _ident_target(src_license: str, rng) -> str:
    """Pick ONE attack target from the OPPOSITE license class, at random (A).

    A copyleft file is attacked toward a random permissive target and vice versa.
    The exact target barely changes the teacher-forced margin (which only measures
    whether argmax leaves the true label), so one random pick per file keeps the
    attack surface diverse without the 5x combinatorial cost of enumerating all.
    """
    pool = sorted(PERMISSIVE_SET if src_license in COPYLEFT_SET else COPYLEFT_SET)
    return rng.choice(pool)


def instantiate_ident(template: str, src_license: str, tgt_license: str) -> str:
    return template.replace("{src_license}", src_license).replace("{tgt_license}", tgt_license)


def instantiate_clause(template: str, src_license: str) -> str:
    return template.replace("{src_license}", src_license).replace("{tgt_license}", "")


def _clause_attack_templates(clause_templates: dict, probe_key: str, truth: str) -> list[str]:
    """Templates that push a clause answer AWAY from `truth`.

    The template file has two direction buckets per probe:
      Yes_to_No  — asserts the clause answer is "No"
      No_to_Yes  — asserts the clause answer is "Yes"

    For a 3-way ground truth {Yes, No, None} the attack succeeds if the model's argmax
    leaves `truth` for ANY other label, so we inject every template whose assertion
    contradicts `truth`:
      truth == "Yes"  -> Yes_to_No templates (assert No)
      truth == "No"   -> No_to_Yes templates (assert Yes)
      truth == "Unknown" -> BOTH buckets (assert Yes or No; either leaves Unknown)
    """
    d = clause_templates.get(probe_key, {})
    if truth == "Yes":
        return list(d.get("Yes_to_No", []))
    if truth == "No":
        return list(d.get("No_to_Yes", []))
    # truth == "Unknown": any assertion (Yes or No) moves the answer off Unknown
    return list(d.get("Yes_to_No", [])) + list(d.get("No_to_Yes", []))


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_ident(model, samples, templates_data, labels, success_tau, out_path, done_keys, rng, calib=None):
    """For each clean-correct file, batch-score all instantiated templates and record
    the single most margin-reducing one if it flips (C: one batched pass per file)."""
    ident_templates = templates_data.get("identification", {})
    all_templates = (
        [(t, "from_copyleft") for t in ident_templates.get("from_copyleft", [])]
        + [(t, "from_permissive") for t in ident_templates.get("from_permissive", [])]
    )
    if not all_templates:
        print("[warn] no identification templates found in template file")
        return

    question = _build_ident_question(labels)
    label_ids = [model.encode(l, add_special=False) for l in labels]
    empty_ids = model.encode("", add_special=False)

    with open(out_path, "a", buffering=1) as out_f:
        for s in tqdm(samples, desc="ident-seed", unit="file"):
            if s.name in done_keys:
                continue
            seg = model.build_segments(s.source_before, s.source_after, question)
            clean_sc = evaluate_comment(model, seg, empty_ids, labels, s.license_id, "clean", calib)
            if clean_sc.scores.argmax_label() != s.license_id:
                done_keys.add(s.name)          # record as processed (clean-wrong)
                continue
            clean_margin = float(clean_sc.scores.margin(s.license_id))

            tgt = _ident_target(s.license_id, rng)
            comments = [instantiate_ident(t, s.license_id, tgt) for t, _ in all_templates]
            comment_ids = [model.encode(c, add_special=False).to(model.device) for c in comments]

            best_i, _, best_margin = best_comment_by_margin(
                model, seg, comment_ids, labels, s.license_id, label_ids, calib)

            if best_margin < success_tau:
                out_f.write(json.dumps({
                    "name": s.name, "license": s.license_id, "tgt_license": tgt,
                    "probe": "", "template": all_templates[best_i][0],
                    "comment": comments[best_i], "gen_margin": best_margin,
                    "clean_margin": clean_margin, "generator": "template",
                }) + "\n")
            done_keys.add(s.name)


def run_clause(model, samples, templates_data, labels, success_tau, out_path, done_keys, rng):
    """For each clean-correct (file, probe), batch-score all attack templates and record
    the best flipping one (C: one batched pass per pair). Uses 3-way {Yes, No, None}
    scoring so every ground-truth value -- including None -- is a valid attack target."""
    clause_templates = templates_data.get("clause", {})
    yn3 = ["Yes", "No", "Unknown"]                 # 3-way label set (matches question wording)
    yn3_ids = [model.encode(t, add_special=False) for t in yn3]
    empty_ids = model.encode("", add_special=False)

    with open(out_path, "a", buffering=1) as out_f:
        for s in tqdm(samples, desc="clause-seed", unit="file"):
            for probe in CLAUSE_PROBES:
                truth = probe.truth.get(s.license_id)          # str or None
                truth_label = "Unknown" if truth is None else truth
                pair_key = f"{s.name}||{probe.key}"
                if pair_key in done_keys:
                    continue

                templates = _clause_attack_templates(clause_templates, probe.key, truth_label)
                if not templates:
                    continue

                seg = model.build_segments(s.source_before, s.source_after, probe.question)
                clean_sc = evaluate_comment(model, seg, empty_ids, yn3, truth_label, "clean")
                if clean_sc.scores.argmax_label() != truth_label:
                    done_keys.add(pair_key)        # clean-wrong -> not attackable
                    continue
                clean_margin = float(clean_sc.scores.margin(truth_label))

                comments = [instantiate_clause(t, s.license_id) for t in templates]
                comment_ids = [model.encode(c, add_special=False).to(model.device) for c in comments]

                best_i, _, best_margin = best_comment_by_margin(
                    model, seg, comment_ids, yn3, truth_label, yn3_ids)

                if best_margin < success_tau:
                    out_f.write(json.dumps({
                        "name": s.name, "license": s.license_id, "tgt_license": "",
                        "probe": probe.key, "truth": truth_label,
                        "template": templates[best_i], "comment": comments[best_i],
                        "gen_margin": best_margin, "clean_margin": clean_margin,
                        "generator": "template",
                    }) + "\n")
                done_keys.add(pair_key)


# ── Helpers ───────────────────────────────────────────────────────────────────

class _PersistingKeys(set):
    """A set subclass that persists each added key to a sidecar file (append-only).

    Used for resume: done_keys.add(...) writes both to memory and disk, so restarts
    skip already-evaluated files/pairs (including clean-wrong and non-flipping ones).
    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        self._f = None
        # Load existing keys on init
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        super().add(line)
        # Open for append (buffered=1 means line-buffered)
        self._f = open(self.path, "a", buffering=1)

    def add(self, key):
        if key not in self:
            super().add(key)
            self._f.write(key + "\n")

    def __del__(self):
        if self._f:
            self._f.close()


def _build_ident_question(labels: list[str]) -> str:
    from lic_attack.data import identification_question
    return identification_question("open_ended", labels)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="YAML config file (e.g. configs/m3_llama.yaml); "
                         "reads model.name, task.labels, task.dataset_dir, task.templates_path")
    ap.add_argument("--model-path", default=None,
                    help="Override model path from config")
    ap.add_argument("--templates", default=None,
                    help="Override template file path from config")
    ap.add_argument("--dataset-dir", default=None,
                    help="Override dataset dir from config")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Override SPDX labels from config")
    ap.add_argument("--out", required=True, help="output JSONL seed file")
    ap.add_argument("--target", choices=["ident", "clause"], default="ident")
    ap.add_argument("--gpus", default="0",
                    help="comma-separated GPU indices to use, e.g. 0,1")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-per-license", type=int, default=None,
                    help="cap files per license (for quick smoke tests)")
    ap.add_argument("--success-tau", type=float, default=0.0,
                    help="margin threshold for a flip (default 0.0)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibration", choices=["none", "pmi"], default="none",
                    help="label-score comparison rule. ident uses 'pmi' because its 19 labels differ in token length (MIT/ISC 1 token vs LGPL-2.1 6), which the length-normalised mean turns into a systematic bias; clause uses 'none' because Yes/No/Unknown are all 1 token, so the two rules coincide there. MUST match whatever the GCG run used, or the gate and the search disagree.")
    args = ap.parse_args()

    # Load config
    import yaml
    cfg = yaml.safe_load(open(ROOT / args.config))

    model_path = args.model_path or cfg["model"]["name"]
    templates_path = args.templates or cfg["task"].get("templates_path", "data/comment_templates.json")
    dataset_dir   = args.dataset_dir or cfg["task"].get("dataset_dir", "data/licenses_by_name")
    labels        = args.labels or cfg["task"]["labels"]

    # GPU setup — with device: auto HuggingFace handles multi-card; just set CUDA_VISIBLE_DEVICES
    gpu_list = [int(g) for g in args.gpus.split(",")]
    # Override the device setting in config if explicit gpus given
    if len(gpu_list) == 1:
        model_cfg_override = dict(cfg["model"], device=f"cuda:{gpu_list[0]}")
    else:
        model_cfg_override = dict(cfg["model"], device="auto")

    # Load templates
    tmpl_path = ROOT / templates_path
    if not tmpl_path.exists():
        raise SystemExit(f"Template file not found: {tmpl_path}\n"
                         "Run scripts/gen_comment_templates.py first.")
    templates_data = json.loads(tmpl_path.read_text())

    # Load samples
    samples = build_samples_from_dir(
        str(ROOT / dataset_dir),
        eval_licenses=labels if args.target == "ident" else None,
        max_per_license=args.max_per_license,
    )

    # Shard
    if args.num_shards > 1:
        samples = [s for i, s in enumerate(samples) if i % args.num_shards == args.shard_index]
    print(f"[seed] {len(samples)} samples, shard {args.shard_index}/{args.num_shards}, "
          f"model={model_path.split('/')[-1]}, target={args.target}")

    # Resume: a file/pair is "done" once EVALUATED, whether or not it yielded a flip.
    # Successful seeds land in out_path, but clean-wrong and non-flipping files produce
    # no output row, so we track every processed key in a sidecar to skip them on
    # restart (re-evaluating them on the 30B model would waste hours). Keys are
    # file-level for ident (s.name) and pair-level for clause (s.name||probe.key) —
    # exactly what run_ident/run_clause add to done_keys.
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path = out_path.with_suffix(out_path.suffix + ".processed")
    done_keys = _PersistingKeys(processed_path)
    print(f"[resume] {len(done_keys)} keys already processed (from {processed_path.name})")

    # Build model
    print(f"[seed] model={model_path}  labels={len(labels)}")
    model = build_model(model_cfg_override, labels)
    # Same rule as the GCG run, for the same reason: the seed stage decides both which
    # files are attributable (clean argmax) and which template instances count as
    # flipping seeds, both through this scorer. Seeding under one rule and attacking
    # under another would silently change what "a seed" means.
    calib = None
    if args.calibration == "pmi":
        if args.target != "ident":
            raise SystemExit("--calibration pmi is ident-only; clause needs one "
                             "reference per probe and is run uncalibrated by design")
        calib = null_calibration(model, _build_ident_question(labels), labels)
        print(f"[calib] PMI reference over {len(calib)} labels", flush=True)

    import random
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)     # reproducible random target selection (A)
    if args.target == "ident":
        run_ident(model, samples, templates_data, labels, args.success_tau, out_path, done_keys, rng, calib)
    else:
        run_clause(model, samples, templates_data, labels, args.success_tau, out_path, done_keys, rng)

    total = sum(1 for _ in open(out_path))
    print(f"[seed] done. {total} seeds in {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
