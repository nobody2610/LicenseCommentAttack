#!/bin/bash
# Verify the container can actually run on this GPU before any long job starts.
#
# The failure this guards against is specific: a torch build whose compiled arch list
# does not include the device's compute capability raises "no kernel image is
# available" on the FIRST kernel launch, which on these runs is thousands of steps in
# and hours after the job was launched detached. Checking it in a second at startup
# turns a wasted night into an immediate error.
set -euo pipefail

python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("[entrypoint] no CUDA device visible -- check --gpus and the driver")
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
have = torch.cuda.get_arch_list()
print(f"[entrypoint] torch {torch.__version__}  cuda {torch.version.cuda}")
print(f"[entrypoint] device  {torch.cuda.get_device_name(0)}  ({arch}), {torch.cuda.device_count()} visible")
if arch not in have:
    sys.exit(f"[entrypoint] torch has no kernels for {arch}; compiled for {have}.\n"
             f"             Use an NGC image new enough for this GPU.")
# A real kernel launch: the arch list can be right while the install is still broken.
x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all()
print("[entrypoint] bf16 matmul ok")
PY

if [ -n "${LIC_MODEL_ROOT:-}" ] && [ ! -d "${LIC_MODEL_ROOT}" ]; then
    echo "[entrypoint] WARNING: LIC_MODEL_ROOT=${LIC_MODEL_ROOT} is not a directory." >&2
    echo "[entrypoint]          Did you forget -v /host/models:${LIC_MODEL_ROOT}?" >&2
fi

exec "$@"
