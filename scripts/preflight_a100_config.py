#!/usr/bin/env python3
"""Fail fast on invalid A100 inference config BEFORE reading large model weights.

The GPU worker can otherwise load T5, CLIP, VAE and all 16.4B DiT weights
before discovering a trivial config/type mismatch such as YAML 'off' -> False.
"""
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from wanxiang.utils.fp16_guard import OverflowGuard
from wanxiang.utils.precision import resolve_precision


def validate_a100_config(path):
    cfg = OmegaConf.load(str(path))
    t = cfg.test_cfg
    if (int(t.sp_size), int(t.sharding_size)) != (2, 2):
        raise ValueError("A100 profile must use sp_size=2 and sharding_size=2")
    if str(t.dit_shard_dtype) != "bf16":
        raise ValueError("A100 profile must retain checkpoint-exact dit_shard_dtype=bf16")
    if str(t.sharding_strategy).upper() != "FULL_SHARD":
        raise ValueError("A100 profile must use FULL_SHARD on the two-GPU node")
    prec = resolve_precision(t, torch.device("cpu"))  # explicit BF16 profile; no CUDA allocation
    if prec.dit != torch.bfloat16 or prec.t5 != torch.bfloat16:
        raise ValueError("A100 profile must compute in native BF16 for DiT and T5")
    if prec.guard not in ("off", "first_step", "all"):
        raise ValueError(f"invalid overflow_guard after parsing YAML: {prec.guard!r}")
    # Exercise precisely the worker construction that previously failed
    # *after* the DiT checkpoint finished loading on the real A100 server.
    guard = OverflowGuard(torch.nn.Identity(), mode=prec.guard,
                          action=str(t.get("overflow_action", "abort")))
    guard.remove()
    return prec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="A100 YAML path to validate")
    args = parser.parse_args()
    prec = validate_a100_config(args.config)
    print(f"[A100 CONFIG OK] {args.config.name}: {prec.describe()}; 2-way BF16 FSDP")


if __name__ == "__main__":
    main()
