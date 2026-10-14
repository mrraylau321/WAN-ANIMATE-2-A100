#!/usr/bin/env bash
# Experimental 8x NVIDIA V100 16GB (sm70) presets for Wan-Animate-2.
# Not a single-card 16GB solution. NOT GPU-verified or a guarantee of image quality.
# Usage: PRESET=minimal|balanced|detail bash scripts/run_v100_16gb.sh
set -euo pipefail

PRESET=${PRESET:-minimal}
case "$PRESET" in
  minimal)         preset_res=256x448; preset_clip_len=17 ;;
  balanced)        preset_res=256x448; preset_clip_len=25 ;;
  detail)          preset_res=288x512; preset_clip_len=25 ;;
  experimental480) preset_res=480p;    preset_clip_len=17 ;;
  *) echo "PRESET must be minimal|balanced|detail|experimental480 (got $PRESET)" >&2; exit 2 ;;
esac

export MODE=${MODE:-distill}
export RES=${RES:-$preset_res}
export CLIP_LEN=${CLIP_LEN:-$preset_clip_len}
export GPUS=${GPUS:-0,1,2,3,4,5,6,7}
export DTYPE=${DTYPE:-fp16}
export WAN_SDPA_Q_CHUNK=${WAN_SDPA_Q_CHUNK:-512}
export WAN_SDPA_MATH_BUDGET_MB=${WAN_SDPA_MATH_BUDGET_MB:-256}

if [[ "$MODE" != base && "$MODE" != distill ]]; then
  echo "MODE must be base or distill" >&2; exit 2
fi
if [[ "$DTYPE" != fp16 ]]; then
  echo "16GB presets require explicitly guarded fp16 DiT compute; DTYPE=$DTYPE is unsupported" >&2; exit 2
fi
if ! [[ "$GPUS" =~ ^[0-9]+(,[0-9]+){7}$ ]]; then
  echo "GPUS must enumerate eight unique V100 GPU indices (e.g. 0,1,2,3,4,5,6,7)" >&2; exit 2
fi
IFS=',' read -r -a cards <<<"$GPUS"
for (( i=0; i<8; ++i )); do
  for (( j=i+1; j<8; ++j )); do
    if [[ "${cards[i]}" == "${cards[j]}" ]]; then
      echo "Duplicate GPU index ${cards[i]} is not supported" >&2; exit 2
    fi
  done
done
if ! [[ "$CLIP_LEN" =~ ^[0-9]+$ ]] || (( CLIP_LEN < 5 || (CLIP_LEN - 1) % 4 != 0 )); then
  echo "CLIP_LEN must be 4n+1 (e.g. 17 or 25) for VAE temporal alignment" >&2; exit 2
fi
case "$RES" in
  256x448|288x512) ;;
  480p|540p|720p)
    if [[ "${ALLOW_UNVERIFIED_HIGH_RES:-0}" != 1 ]]; then
      echo "$RES on 8x16GB is unverified and at HIGH OOM risk; explicitly opt in with ALLOW_UNVERIFIED_HIGH_RES=1" >&2
      exit 2
    fi ;;
  *) echo "RES must be 256x448|288x512|480p|540p|720p" >&2; exit 2 ;;
esac

# Checkpoint BF16 values stay BF16 in the 40 large DiT block FSDP2 shards.
# FP16 is a compute dtype, NOT the checkpoint storage dtype. T5/VAE stay FP32.
# The timestep embeddings and output head require true FP32 operations, and
# are separately sharded as FP32 islands (checked by FP16 forward tests).
# Only the previous decoded overlap is retained between chunks on CPU.
# Enforce these safety-critical overrides AFTER optional caller arguments.
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-} test_cfg.dit_shard_dtype=bf16 test_cfg.low_vram=true test_cfg.log_gpu_memory=true test_cfg.overflow_action=abort test_cfg.fp32_modules=[time_embedding,time_projection,head]"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "PRESET=$PRESET MODE=$MODE RES=$RES CLIP_LEN=$CLIP_LEN GPUS=$GPUS DTYPE=$DTYPE"
  echo "WAN_SDPA_Q_CHUNK=$WAN_SDPA_Q_CHUNK WAN_SDPA_MATH_BUDGET_MB=$WAN_SDPA_MATH_BUDGET_MB"
  echo "EXTRA_OVERRIDES=$EXTRA_OVERRIDES"
  exit 0
fi

exec "$(dirname "${BASH_SOURCE[0]}")/run_v100.sh" "$@"
