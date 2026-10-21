#!/bin/bash
# Supervise the GCG cap=15 ident runs: one model at a time, 7 shards, restart on death.
#
# Scoring is PMI-calibrated (--calibration pmi). The earlier mean-per-token rule gave
# single-token labels (MIT, ISC, Zlib) a 0% clean accuracy, so those licences never
# entered the ASR denominator at all, and it held License-Llama's clean accuracy to
# 23% when calibration puts it at 56%. Results from that run are kept for comparison
# under outputs/gcg_m3_cap15_meannorm/.
#
# Replaces the plain `wait`-on-pids loop in rebuild_and_run_cap15.sh, which had two
# faults that cost real time:
#   * a shard that dies (CUDA OOM) is never noticed -- one card sat idle 9 hours
#   * `wait` returns for the dead pid, so the driver advances to the next model while
#     a manually restarted shard is still holding that card, and the new shard for the
#     same card OOMs on contention
#
# Here a model is finished only when every shard has processed every sample assigned to
# it, which is checked against the .processed sidecar rather than against process
# liveness. Restarts are free: run_gcg_m3.py resumes from that sidecar.
#
# Usage:  setsid nohup bash scripts/supervise_gcg_cap15.sh > logs/gcg_m3_cap15/supervisor.log 2>&1 &
set -u
# Locate the repo from the script itself, and activate the host conda env only if it
# is there. In the container the code is bind-mounted at /workspace and the
# interpreter is already the right one, so both steps must be skipped rather than
# hard-coded to the 4090 host's paths.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f /opt/anaconda3/etc/profile.d/conda.sh ]; then
  source /opt/anaconda3/etc/profile.d/conda.sh && conda activate licatk
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset LIC_INFER_ONLY          # GCG needs the backward pass

OUTDIR=outputs/gcg_m3_cap15_pmi
GCGLOG=logs/gcg_m3_cap15_pmi
mkdir -p "$OUTDIR" "$GCGLOG"

# Cards to shard over. Default skips card 0: on the 4090 host a ~2.6GB non-pytorch
# process lives there and the GCG backward needs the full 24GB. Override with
# LIC_CARDS="0 1 2 3 4 5 6 7" on a host where every card is free.
read -r -a CARDS <<< "${LIC_CARDS:-1 2 3 4 5 6 7}"
NSHARD=${#CARDS[@]}
CAP=15; NSTEPS=100; MARGIN=-1.0; BATCH=64
POLL=180                       # seconds between liveness checks

declare -A CFG=(
  [llama]=configs/m3_llama.yaml
  [qwen25coder]=configs/m3_qwen25coder.yaml
  [deepseek_coder]=configs/m3_deepseek_coder.yaml
  [license_llama]=configs/m3_license_llama.yaml
)

say() { echo "[sup $(date '+%m-%d %H:%M:%S')] $*"; }

# How many samples is shard $2 of model $1 supposed to process? Derived from the same
# loader + sharding rule the runner uses, so it cannot drift from the runner's view.
expected() {
  python - "$1" <<'PY'
import sys, yaml
sys.path.insert(0, "src")
from lic_attack.data import build_samples_from_dir
cfg = yaml.safe_load(open(sys.argv[1]))["task"]
ss = build_samples_from_dir("data/licenses_by_name",
                            eval_licenses=cfg["labels"], max_per_license=15)
print(len(ss))
PY
}

launch() {   # $1 tag  $2 shard
  local tag=$1 s=$2 card=${CARDS[$2]}
  CUDA_VISIBLE_DEVICES=$card nohup python -u scripts/run_gcg_m3.py \
    --config "${CFG[$tag]}" --target ident \
    --max-per-license "$CAP" --n-steps "$NSTEPS" --margin-success "$MARGIN" \
    --batch-size "$BATCH" --shard-index "$s" --num-shards "$NSHARD" --gpus 0 \
    --calibration pmi \
    --out "$OUTDIR/${tag}_ident.r${s}.jsonl" \
    >> "$GCGLOG/${tag}_ident.r${s}.log" 2>&1 &
  say "  launched $tag shard $s on card $card (pid $!)"
}

alive() {    # $1 tag  $2 shard  -- is a process for this exact shard running?
  pgrep -f "run_gcg_m3.py .*--shard-index $2 .*${1}_ident.r${2}.jsonl" > /dev/null
}

done_count() {   # $1 tag  $2 shard -- rows in the .processed sidecar
  local f="$OUTDIR/${1}_ident.r${2}.jsonl.processed"
  [ -f "$f" ] && wc -l < "$f" || echo 0
}

run_model() {
  local tag=$1
  local total; total=$(expected "${CFG[$tag]}")
  # ceil split: shard s takes every NSHARD-th sample starting at s
  say "=== $tag: $total samples over $NSHARD shards ==="
  while true; do
    local pending=0 running=0
    for s in $(seq 0 $((NSHARD-1))); do
      local want=$(( (total - s + NSHARD - 1) / NSHARD ))
      local have; have=$(done_count "$tag" "$s")
      if [ "$have" -ge "$want" ]; then continue; fi
      pending=$((pending+1))
      if alive "$tag" "$s"; then
        running=$((running+1))
      else
        say "$tag shard $s not running ($have/$want done) -- (re)starting"
        launch "$tag" "$s"
        sleep 25            # stagger model loads so 7 cards do not page in at once
      fi
    done
    if [ "$pending" -eq 0 ]; then
      local att flip oom
      att=$(cat "$OUTDIR/${tag}_ident".r*.jsonl 2>/dev/null | grep -c '"clean_correct": true')
      flip=$(cat "$OUTDIR/${tag}_ident".r*.jsonl 2>/dev/null | grep -c '"still_flips": true')
      oom=$(cat "$OUTDIR/${tag}_ident".r*.jsonl 2>/dev/null | grep -c '"oom_skipped": true')
      say "=== $tag DONE -- clean-correct $att, flipped $flip, oom-skipped $oom ==="
      return 0
    fi
    sleep "$POLL"
  done
}

for tag in llama qwen25coder deepseek_coder license_llama; do
  run_model "$tag"
done
say "ALL MODELS DONE. results in $OUTDIR"
