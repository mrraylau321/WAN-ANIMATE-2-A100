#!/usr/bin/env bash
# Independent 2x A100 80GB (81920 MiB/card) native-BF16 inference launcher.
# One Python parent, two child FSDP2/context-parallel ranks. Never torchrun.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PRESET=${PRESET:-standard}
case "$PRESET" in
  smoke)    preset_res=480p; preset_len=33 ;;
  standard) preset_res=720p; preset_len=65 ;;
  full)     preset_res=720p; preset_len=81 ;;
  *) echo "PRESET must be smoke|standard|full" >&2; exit 2 ;;
esac
MODE=${MODE:-distill}; RES=${RES:-$preset_res}; CLIP_LEN=${CLIP_LEN:-$preset_len}
GPUS=${GPUS:-0,1}; DTYPE=${DTYPE:-bf16}; ATTN_BACKEND=${ATTN_BACKEND:-auto}
export WAN_SDPA_Q_CHUNK=${WAN_SDPA_Q_CHUNK:-4096}
export WAN_SDPA_MATH_BUDGET_MB=${WAN_SDPA_MATH_BUDGET_MB:-512}

[[ "$MODE" == distill || "$MODE" == base ]] || { echo "MODE must be distill|base" >&2; exit 2; }
[[ "$DTYPE" == bf16 ]] || { echo "A100 80GB recipe requires native BF16" >&2; exit 2; }
[[ "$GPUS" =~ ^[0-9]+,[0-9]+$ ]] || { echo "Select exactly two GPU indices, e.g. GPUS=0,1" >&2; exit 2; }
IFS=',' read -r -a cards <<< "$GPUS"
[[ "${cards[0]}" != "${cards[1]}" ]] || { echo "Duplicate GPU indices" >&2; exit 2; }
[[ "$CLIP_LEN" =~ ^[0-9]+$ ]] && (( CLIP_LEN >= 5 && (CLIP_LEN - 1) % 4 == 0 )) || {
  echo "CLIP_LEN must be 4n+1 (e.g. 33, 65, 81)" >&2; exit 2;
}
case "$RES" in
  256x448) W=256; H=448 ;;
  288x512) W=288; H=512 ;;
  480p) W=480; H=832 ;;
  540p) W=544; H=960 ;;
  720p) W=720; H=1280 ;;
  *) echo "RES must be 256x448|288x512|480p|540p|720p" >&2; exit 2 ;;
esac
case "$ATTN_BACKEND" in auto|flash|sdpa) ;; *) echo "ATTN_BACKEND must be auto|flash|sdpa" >&2; exit 2 ;; esac
if [[ "$MODE" == distill ]]; then
  CONFIG=wan_animate_2_distillation_a100.yaml; DIT=wan_animate_2_bf16_distillation.safetensors
  STEPS=${STEPS:-10}; GUIDE=${GUIDE:-1.0}
else
  CONFIG=wan_animate_2_a100.yaml; DIT=wan_animate_2_bf16.safetensors
  STEPS=${STEPS:-40}; GUIDE=${GUIDE:-3.0}
fi
[[ -z "${EXTRA_OVERRIDES:-}" ]] || {
  echo "EXTRA_OVERRIDES disabled for safe A100 allocation; edit the A100 YAML deliberately" >&2; exit 2;
}
echo "2x A100 80GB: PRESET=$PRESET MODE=$MODE RES=$RES (${W}x${H}) CLIP_LEN=$CLIP_LEN"
echo "GPUS=$GPUS DTYPE=$DTYPE ATTN_BACKEND=$ATTN_BACKEND STEPS=$STEPS GUIDE=$GUIDE"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "CONFIG=infer/$CONFIG sharding_size=2 sp_size=2 dit_shard_dtype=bf16"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
python "$REPO/scripts/probe_a100.py"
python "$REPO/scripts/preflight_a100_config.py" "$REPO/infer/$CONFIG"
DEMO="$REPO/examples/demo1"
CKPT_DIR=${CKPT_DIR:-"$REPO/ckpts"}
[[ -d "$CKPT_DIR" ]] || { echo "Missing CKPT_DIR: $CKPT_DIR" >&2; exit 1; }
CKPT_DIR=$(cd "$CKPT_DIR" && pwd)
REF_IMG=${REF_IMG:-"$DEMO/reference.png"}
DRIVE=${DRIVE:-"$DEMO/template.mp4"}
if [[ -z "${PROMPT:-}" ]]; then
  [[ "$REF_IMG" == "$DEMO/reference.png" ]] || { echo "PROMPT required for custom REF_IMG" >&2; exit 2; }
  PROMPT=$(< "$DEMO/prompt.txt")
fi
PROMPT_REF=${PROMPT_REF:-人物动作的参考视频}
WAN="$CKPT_DIR/videomodel/Wan-AI"
for f in "$CKPT_DIR/wan_animate_2/$DIT" "$WAN/models_t5_umt5-xxl-enc-bf16.pth" "$WAN/vae.pth" \
  "$WAN/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" "$WAN/umt5-xxl" \
  "$WAN/xlm-roberta-large" "$REF_IMG" "$DRIVE"; do
  [[ -e "$f" ]] || { echo "Missing required file/directory: $f" >&2; exit 1; }
done

OUT_DIR=${OUT_DIR:-"$REPO/outputs"}
mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)
LOG="$OUT_DIR/a100_${MODE}_${RES}_${CLIP_LEN}_$(date +%Y%m%d_%H%M%S).log"
echo "Log: $LOG"
cd "$REPO/infer"
python wan_animate_2_demo.py \
  --config "./$CONFIG" --dtype "$DTYPE" --attn-backend "$ATTN_BACKEND" \
  --prompt "$PROMPT" --prompt_ref "$PROMPT_REF" \
  --refer-img-file "$REF_IMG" --refer-video-file "$DRIVE" \
  --width "$W" --height "$H" --fps "${FPS:-24}" --clip_len "$CLIP_LEN" \
  --sample_guide_scale "$GUIDE" --step "$STEPS" --seed "${SEED:-42}" --output-dir "$OUT_DIR" \
  --cfg-override \
    "model_path=$CKPT_DIR/wan_animate_2/$DIT" \
    "model.t5_checkpoint=$WAN/models_t5_umt5-xxl-enc-bf16.pth" \
    "model.t5_tokenizer=$WAN/umt5-xxl" \
    "model.vae_checkpoint=$WAN/vae.pth" \
    "model.clip_checkpoint=$WAN/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    "model.clip_tokenizer=$WAN/xlm-roberta-large" \
    "test_cfg.sp_size=2" "test_cfg.sharding_size=2" \
  2>&1 | tee "$LOG"
