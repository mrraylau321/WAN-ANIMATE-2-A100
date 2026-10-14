#!/usr/bin/env bash
# Wan-Animate-2 Base (40 steps, CFG 3.0) on 8x V100. All knobs: see scripts/run_v100.sh.
MODE=base exec "$(dirname "${BASH_SOURCE[0]}")/run_v100.sh" "$@"
