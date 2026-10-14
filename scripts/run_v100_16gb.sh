#!/usr/bin/env bash
# Experimental 16GB-per-GPU preset for the existing multi-GPU FSDP/sequence-parallel runner.
# Not validated on 16GB hardware. This is NOT single-GPU or guaranteed-OOM-free support.
set -euo pipefail
export RES=${RES:-256x448}
export CLIP_LEN=${CLIP_LEN:-25}
export WAN_SDPA_Q_CHUNK=${WAN_SDPA_Q_CHUNK:-512}
export WAN_SDPA_MATH_BUDGET_MB=${WAN_SDPA_MATH_BUDGET_MB:-256}
export GPUS=${GPUS:-0,1,2,3,4,5,6,7}
exec "$(dirname "${BASH_SOURCE[0]}")/run_v100.sh" "$@"
