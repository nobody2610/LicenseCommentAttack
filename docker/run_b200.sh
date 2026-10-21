#!/bin/bash
# One command to take the experiment queue from a fresh B200 host to running.
#
#   ./docker/run_b200.sh build          build the image
#   ./docker/run_b200.sh check          verify GPUs, kernels and model mounts
#   ./docker/run_b200.sh smoke          3-step GCG on one file, ~3 min
#   ./docker/run_b200.sh gcg            the full GCG ident run, detached
#   ./docker/run_b200.sh queue          the whole remaining queue, detached
#   ./docker/run_b200.sh shell          interactive shell in the container
#
# Everything runs detached under `docker run -d`, so closing the SSH session does not
# kill it -- the container is owned by the daemon, not by the login shell. That is the
# main reason to prefer this over nohup here: on the 4090 host a dropped connection
# was a live risk for every long run.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=${IMAGE:-licatk:b200}
NAME=${NAME:-licatk}
# Host directory holding the four white-box model directories. Override per host.
MODELS=${MODELS:-$HOME/models}
REPO=$(pwd)

# --shm-size: the default 64 MB is not enough for dataloader/NCCL shared buffers.
# --ipc=host would also work; an explicit size keeps the container's footprint stated.
COMMON=(
  --gpus all
  --shm-size=16g
  --ulimit memlock=-1
  -v "$MODELS":/models:ro
  -v "$REPO":/workspace
  -w /workspace
  -e LIC_MODEL_ROOT=/models
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # Every card is free on this host, unlike the 4090 box whose card 0 is occupied.
  -e LIC_CARDS="0 1 2 3 4 5 6 7"
  # GCG ident runs on the 4090 host, and llama's GCG clause already finished here at
  # cap=15 -- its results are a superset of the cap=8 sample, since the loader takes
  # the first N files per licence, so it can be subset for a like-for-like table
  # rather than rerun. Unset to run the whole queue on one machine.
  -e LIC_SKIP="${LIC_SKIP-gcg_*_ident gcg_llama_clause}"
  # Files per licence. 8 halves the queue against 15; every licence holds >= 13.
  -e LIC_CAP="${LIC_CAP-8}"
)

case "${1:-}" in
  build)
    docker build -t "$IMAGE" -f Dockerfile.b200 .
    ;;

  check)
    # entrypoint.sh already asserts the arch list and does a real bf16 matmul.
    docker run --rm "${COMMON[@]}" "$IMAGE" bash -lc '
      echo "--- models visible to the container ---"
      ls -1 /models
      echo "--- config paths resolve ---"
      python - <<PY
import sys, yaml; sys.path.insert(0,"src")
from lic_attack.model import resolve_model_path
for c in ["m3_llama","m3_qwen25coder","m3_deepseek_coder","m3_license_llama"]:
    n = yaml.safe_load(open(f"configs/{c}.yaml"))["model"]["name"]
    try:    print(f"  {c:<22} -> {resolve_model_path(n)}")
    except FileNotFoundError as e: print(f"  {c:<22} !! {e}")
PY'
    ;;

  smoke)
    # Small enough to finish while you watch it, big enough to exercise the paths
    # that actually break: model load, PMI calibration, the invertibility filter.
    docker run --rm "${COMMON[@]}" -e CUDA_VISIBLE_DEVICES=0 "$IMAGE" bash -lc '
      python -u scripts/run_gcg_m3.py --config configs/m3_llama.yaml --target ident \
        --max-per-license 1 --n-steps 3 --batch-size 64 \
        --shard-index 0 --num-shards 7 --gpus 0 --out /tmp/smoke.jsonl
      python -c "
import json
rs=[json.loads(l) for l in open(\"/tmp/smoke.jsonl\")]
att=[r for r in rs if r[\"clean_correct\"]]
print(f\"rows={len(rs)} attacked={len(att)}\")
for r in att: print(\"  \", r[\"name\"], \"realizable=\", r.get(\"realizable\"))
assert all(r.get(\"realizable\") for r in att), \"unrealizable comment accepted\"
print(\"SMOKE OK\")"'
    ;;

  gcg|queue)
    script=scripts/supervise_gcg_cap15.sh
    [ "$1" = queue ] && script=scripts/queue_all.sh
    docker rm -f "$NAME" 2>/dev/null || true
    docker run -d --name "$NAME" --restart=unless-stopped "${COMMON[@]}" \
      "$IMAGE" bash -lc "bash $script"
    echo "started container '$NAME' running $script"
    echo "  docker logs -f $NAME"
    ;;

  shell)
    docker run --rm -it "${COMMON[@]}" "$IMAGE" bash
    ;;

  *)
    sed -n '2,12p' "$0"; exit 1
    ;;
esac
