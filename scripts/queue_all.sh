#!/bin/bash
# Full experiment queue on the rebuilt dataset. Runs after the live GCG ident job.
#
# Calibration policy, applied consistently at every stage of a branch:
#   ident  --calibration pmi   (19 labels of unequal token length; the mean rule gives
#                               MIT/ISC/Zlib a 0% clean accuracy)
#   clause --calibration none  (Yes/No/Unknown are all 1 token, so mean == sum and
#                               there is no length bias to correct)
# Mixing rules within a branch would mean seeding under one definition of "flips" and
# attacking under another, so each stage below is pinned explicitly.
#
# Dependencies are real: rewrite consumes the seed stage's output, transfer consumes
# rewrite's. Each stage refuses to start if its input is missing or empty.
#
# Usage: setsid nohup bash scripts/queue_all.sh > logs/queue/driver.log 2>&1 &
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

LOG=logs/queue
mkdir -p "$LOG" outputs/gcg_m3_cap15_pmi outputs/m3_seeds_v3 outputs/m3_rewrite_v3 outputs/transfer_v3
say() { echo "[queue $(date '+%m-%d %H:%M:%S')] $*"; }

# Stages to skip, space-separated, matched against the labels passed to run_sharded
# (e.g. LIC_SKIP="gcg_*_ident" to run GCG ident on a different host). Skipping is
# explicit and logged rather than silent: a stage that does not run must be visible in
# the log, because its absence is otherwise indistinguishable from a stage whose
# output nobody checked.
skipped() {
  local label=$1 pat
  for pat in ${LIC_SKIP:-}; do
    # shellcheck disable=SC2053
    if [[ $label == $pat ]]; then say "SKIP $label (matches LIC_SKIP pattern '$pat')"; return 0; fi
  done
  return 1
}

# Cards to shard over. Default skips card 0: on the 4090 host a ~2.6GB non-pytorch
# process lives there and the GCG backward needs the full 24GB. Override with
# LIC_CARDS="0 1 2 3 4 5 6 7" on a host where every card is free.
read -r -a CARDS <<< "${LIC_CARDS:-1 2 3 4 5 6 7}"
NSHARD=${#CARDS[@]}
MODELS=(llama qwen25coder deepseek_coder license_llama)
declare -A CFG=(
  [llama]=configs/m3_llama.yaml
  [qwen25coder]=configs/m3_qwen25coder.yaml
  [deepseek_coder]=configs/m3_deepseek_coder.yaml
  [license_llama]=configs/m3_license_llama.yaml
)
# same-family large target for the transfer stage
declare -A XFER=(
  [llama]=configs/m3_llama70.yaml
  [license_llama]=configs/m3_llama70.yaml
  [qwen25coder]=configs/m3_qwen.yaml
  [deepseek_coder]=configs/m3_deepseek33.yaml
)

# If a GCG supervisor is already running (the 4090 host hands off that way), wait for
# it instead of competing for the cards. On a fresh host there is none, and STAGE 1
# below runs GCG itself -- an earlier version only waited, which silently skipped GCG
# entirely when nothing was running.
while pgrep -f "supervise_gcg_cap15.sh" > /dev/null; do
  say "a GCG supervisor is already running; waiting for it..."
  sleep 900
done

# ---- generic sharded stage: launch NSHARD workers, restart any that die ----
# Same completeness-not-liveness rule as the GCG supervisor: a stage is done when every
# shard's .processed sidecar has reached its expected count, so an OOM death is retried
# rather than silently shortening the run.
run_sharded() {   # $1 label  $2 items-per-shard-unit  $3.. command template with {S}
  local label=$1 unit_total=$2; shift 2
  skipped "$label" && return 0
  local tmpl="$*"
  say "--- $label: $unit_total units over $NSHARD shards (x$KEYS_PER_UNIT keys each) ---"
  while true; do
    local pending=0
    for s in $(seq 0 $((NSHARD-1))); do
      # The runners shard with `i % NSHARD == s`, so shard s owns
      # ceil((unit_total - s) / NSHARD) units. Each unit yields KEYS_PER_UNIT progress
      # records: one per file for ident, six (one per clause probe) for clause.
      #
      # Deriving `want` any other way silently breaks: computing it from the KEY count
      # rather than the UNIT count gives every shard the same quota, which the later
      # shards -- holding one fewer file -- can never reach, so the stage spins forever
      # relaunching a process that exits immediately.
      local units=$(( (unit_total - s + NSHARD - 1) / NSHARD ))
      local want=$(( units * KEYS_PER_UNIT ))
      local have=0
      local f="${PROGRESS_PAT//\{S\}/$s}"
      [ -f "$f" ] && have=$(wc -l < "$f")
      [ "$have" -ge "$want" ] && continue
      pending=$((pending+1))
      if ! pgrep -f "${PGREP_PAT//\{S\}/$s}" > /dev/null; then
        say "  $label shard $s ($have/$want) -- (re)starting on card ${CARDS[$s]}"
        local cmd="${tmpl//\{S\}/$s}"; cmd="${cmd//\{CARD\}/${CARDS[$s]}}"
        eval "CUDA_VISIBLE_DEVICES=${CARDS[$s]} nohup $cmd >> $LOG/${label}.r${s}.log 2>&1 &"
        sleep 25
      fi
    done
    [ "$pending" -eq 0 ] && { say "--- $label DONE ---"; return 0; }
    sleep 180
  done
}


# Files per licence. 8 rather than 15 roughly halves the queue; the dataset holds at
# least 13 for every licence, so no licence is short-changed by this cap.
CAP=${LIC_CAP:-8}
TOTAL=$(python -c "
import sys,yaml; sys.path.insert(0,'src')
from lic_attack.data import build_samples_from_dir
L=yaml.safe_load(open('configs/m3_llama.yaml'))['task']['labels']
print(len(build_samples_from_dir('data/licenses_by_name',eval_licenses=L,max_per_license=$CAP)))")
# ident records one progress key per FILE; clause records one per (file, probe) pair.
# run_sharded is told the FILE count and the keys-per-file multiplier separately via
# KEYS_PER_UNIT, so each shard's quota tracks the files it actually owns.
N_PROBES=6
CLAUSE_TOTAL=$(( TOTAL * N_PROBES ))
say "dataset: $TOTAL files at cap=$CAP ($CLAUSE_TOTAL clause pairs)"

# ===================== STAGE 1: GCG, both targets =============================
# The white-box attack itself, and the largest item in the queue. ident uses PMI
# calibration (19 labels of unequal token length); clause uses none (Yes/No/Unknown
# are all one token, so the mean and summed rules coincide and there is no length
# bias to correct). Both go through the invertibility filter in run_gcg, so every
# accepted comment can actually be written into a file as text.
for tgt in ident clause; do
  CAL=none; [ "$tgt" = ident ] && CAL=pmi
  for m in "${MODELS[@]}"; do
    OUTPAT="outputs/gcg_m3_cap15_pmi/${m}_${tgt}.r{S}.jsonl"
    PROGRESS_PAT="$OUTPAT.processed"
    PGREP_PAT="run_gcg_m3.py .*--shard-index {S} .*${m}_${tgt}.r{S}.jsonl"
    KEYS_PER_UNIT=$([ "$tgt" = clause ] && echo 6 || echo 1)
    run_sharded "gcg_${m}_${tgt}" "$TOTAL" \
      "python -u scripts/run_gcg_m3.py --config ${CFG[$m]} --target $tgt \
       --max-per-license $CAP --n-steps 100 --margin-success -1.0 --batch-size 64 \
       --calibration $CAL --gpus 0 \
       --shard-index {S} --num-shards $NSHARD --out $OUTPAT"
  done
done

# ===================== STAGE 2/3: M3 seed, both targets =====================
for tgt in ident clause; do
  CAL=none; [ "$tgt" = ident ] && CAL=pmi
  for m in "${MODELS[@]}"; do
    OUTPAT="outputs/m3_seeds_v3/${m}_${tgt}.r{S}.jsonl"
    PROGRESS_PAT="$OUTPAT.processed"
    PGREP_PAT="run_m3_seed.py .*--shard-index {S} .*${m}_${tgt}.r{S}.jsonl"
    KEYS_PER_UNIT=$([ "$tgt" = clause ] && echo 6 || echo 1)
    run_sharded "seed_${m}_${tgt}" "$TOTAL" \
      "python -u scripts/run_m3_seed.py --config ${CFG[$m]} --target $tgt \
       --max-per-license $CAP --calibration $CAL --gpus 0 \
       --shard-index {S} --num-shards $NSHARD --out $OUTPAT"
  done
done

# ===================== STAGE 4/5: M3 rewrite, both targets =====================
for tgt in ident clause; do
  CAL=none; [ "$tgt" = ident ] && CAL=pmi
  for m in "${MODELS[@]}"; do
    cat outputs/m3_seeds_v3/${m}_${tgt}.r*.jsonl > outputs/m3_seeds_v3/${m}_${tgt}.all.jsonl 2>/dev/null
    n=$(wc -l < outputs/m3_seeds_v3/${m}_${tgt}.all.jsonl 2>/dev/null || echo 0)
    if [ "$n" -eq 0 ]; then say "!! no seeds for $m/$tgt -- skipping rewrite"; continue; fi
    OUTPAT="outputs/m3_rewrite_v3/${m}_${tgt}.r{S}.jsonl"
    # run_m3_rewrite writes no .processed sidecar -- it resumes by scanning its own
    # output -- so progress is the output file itself. Counting a sidecar that never
    # appears would leave `have` at 0 and relaunch a finished job every 180s forever.
    PROGRESS_PAT="$OUTPAT"
    PGREP_PAT="run_m3_rewrite.py .*--replica {S} .*${m}_${tgt}.r{S}.jsonl"
    KEYS_PER_UNIT=1
    run_sharded "rewrite_${m}_${tgt}" "$n" \
      "python -u scripts/run_m3_rewrite.py --config ${CFG[$m]} --target $tgt \
       --seeds outputs/m3_seeds_v3/${m}_${tgt}.all.jsonl --calibration $CAL --gpus 0 \
       --replica {S} --n-replicas $NSHARD --out $OUTPAT"
  done
done

# ===================== STAGE 6: transfer (small -> large, forward only) =========
# Forward-only, but the 30-70B targets are sharded across all 8 cards, so these run
# one at a time rather than 7-way.
export LIC_INFER_ONLY=1
for tgt in ident clause; do
  for m in "${MODELS[@]}"; do
    g=$(ls outputs/m3_rewrite_v3/${m}_${tgt}.r*.jsonl 2>/dev/null | head -1)
    [ -z "$g" ] && { say "!! no rewrite output for $m/$tgt -- skipping transfer"; continue; }
    say "--- transfer $m/$tgt -> ${XFER[$m]} ---"
    # The 30-70B targets are sharded over every visible card. LIC_MAX_MEMORY caps the
    # per-card budget on the 24GB host; leave it unset elsewhere so a larger card is
    # not artificially limited to 22GiB.
    ALL=$(IFS=,; echo "${CARDS[*]}")
    CUDA_VISIBLE_DEVICES=$ALL ${LIC_MAX_MEMORY:+LIC_MAX_MEMORY=$LIC_MAX_MEMORY} \
    python -u scripts/transfer_local.py --config "${XFER[$m]}" --target "$tgt" \
      --wash-glob "outputs/m3_rewrite_v3/${m}_${tgt}.r*.jsonl" --source-tag "${m}_m3" \
      --out "outputs/transfer_v3/${m}_${tgt}.jsonl" --gpus "$ALL" \
      > "$LOG/xfer_${m}_${tgt}.log" 2>&1
    say "    transfer $m/$tgt done: $(wc -l < outputs/transfer_v3/${m}_${tgt}.jsonl 2>/dev/null || echo 0) rows"
  done
done
unset LIC_INFER_ONLY

# ===================== STAGE 7: logit lens re-dump (cheap) ======================
for m in "${MODELS[@]}"; do
  CUDA_VISIBLE_DEVICES=${CARDS[0]} LIC_INFER_ONLY=1 python -u scripts/dump_logit_lens.py \
    --config "${CFG[$m]}" --model-key "$m" --target ident --gpus 0 \
    --out "outputs/logit_lens_v3/${m}_ident.jsonl" > "$LOG/lens_${m}.log" 2>&1
done

say "QUEUE COMPLETE"
