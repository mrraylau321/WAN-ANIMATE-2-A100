#!/usr/bin/env bash
# Wan-Animate-2 Distillation (10 steps, no CFG) on 8x V100. All knobs: see scripts/run_v100.sh.
MODE=distill exec "$(dirname "${BASH_SOURCE[0]}")/run_v100.sh" "$@"
