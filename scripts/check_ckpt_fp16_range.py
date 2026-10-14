#!/usr/bin/env python
"""Scan Wan-Animate-2 checkpoints for fp16 castability -- run this ON THE V100 SERVER.

Streams one tensor at a time (safetensors via safe_open; .pth via torch.load(mmap=True)), so peak
RAM stays at a few GB even for the 32.8 GB DiT files. Uses exactly the same statistics and pass/fail
rules as the load-time check inside the pipeline (wanxiang/utils/fp16_guard.py):

  overflow   original |w| > 65504 (out of fp16 range; tested on the source value, so 65504 < |w| < 65520,
             which would round down to 65504, also counts)
  underflow  nonzero value becomes 0 in fp16    (|w| <= 2**-25 = 2.98e-8; min fp16 subnormal is 5.96e-8)
  subnormal  value lands in the fp16 subnormal range (|w| < 2**-14 = 6.10e-5): reported, never fatal
  FAIL if any non-finite value, any overflow, or > --underflow-frac of a module's nonzero values
  underflow (0 = strict: any flush fails). Range rules are waived for modules in --fp32-modules.
  Every module with any flushed value is listed (WARNING) even when within the tolerance.

Usage:
  python scripts/check_ckpt_fp16_range.py ckpts/wan_animate_2/wan_animate_2_bf16.safetensors \
      ckpts/wan_animate_2/wan_animate_2_bf16_distillation.safetensors [--json report.json]
Exit code 0 = PASS for all files, 1 = FAIL.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from wanxiang.utils.fp16_guard import (FP16_MAX, aggregate_per_module, fp16_cast_stats,  # noqa: E402
                                       flush_report, fp16_violations)


def iter_tensors(path):
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                yield k, f.get_tensor(k)
    else:
        sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if isinstance(sd, dict) and "model_state" in sd:  # vae.pth layout
            sd = sd["model_state"]
        for k, v in sd.items():
            if torch.is_tensor(v) and v.is_floating_point():
                yield k, v


def unit(module):
    m = re.match(r"^(blocks\.\d+)", module)
    return m.group(1) if m else module


def scan(path, fp32_modules, underflow_frac, top):
    t0 = time.time()
    per_mod, rows = {}, []
    for name, t in iter_tensors(path):
        names, c, m, mf = fp16_cast_stats([(name, t)])
        aggregate_per_module(names, c, m, mf, per_mod)
        nonfin, over, flush, sub, nz = c[0].tolist()
        rows.append(dict(name=name, dtype=str(t.dtype).replace("torch.", ""), numel=t.numel(),
                         max_abs=float(m[0]), max_flushed=float(mf[0]), nonfinite=nonfin, overflow=over, flushed_to_zero=flush,
                         subnormal=sub, nonzero=nz))
    viol = fp16_violations(per_mod, fp32_modules, underflow_frac)
    tot = {k: sum(r[k] for r in rows) for k in ("numel", "nonfinite", "overflow", "flushed_to_zero",
                                                    "subnormal", "nonzero")}
    print(f"\n=== {path}  ({len(rows)} tensors, {tot['numel'] / 1e9:.3f}B values, "
          f"dtypes {sorted({r['dtype'] for r in rows})}, {time.time() - t0:.0f}s)")
    print(f"max|w| = {max(r['max_abs'] for r in rows):.6g} (fp16 max {FP16_MAX:g}); "
          f"over_65504={tot['overflow']}  flushed_to_0={tot['flushed_to_zero']}  "
          f"subnormal={tot['subnormal']} ({tot['subnormal'] / max(tot['nonzero'], 1):.2e} of nonzero)  "
          f"non-finite={tot['nonfinite']}")
    print(f"top {top} tensors by max|w|:")
    for r in sorted(rows, key=lambda r: -r["max_abs"])[:top]:
        print(f"  {r['max_abs']:>12.6g}  {r['name']}")
    total, nmods, lines = flush_report(per_mod)
    if total:
        print(f"WARNING: {total} nonzero weights become 0 in fp16 across {nmods} modules "
              f"(tolerance {underflow_frac:g} of each module's nonzero values):")
        for line in lines:
            print("  " + line)
    if viol:
        print("FAIL:")
        for mod, why in viol:
            print(f"  {mod}: {'; '.join(why)}")
        fixable = sorted({unit(mod) for mod, why in viol if not any("non-finite" in w for w in why)},
                         key=lambda u: (not u.startswith("blocks."), int(u.split(".")[1]) if u.startswith("blocks.") else 0, u))
        if fixable:
            print("suggested allow-list (test_cfg in infer/*_v100.yaml; whole blocks are cheapest to run "
                  "as one fp32 FSDP unit):")
            print(f"  fp32_modules: [{', '.join(fixable)}]")
        if any(any("non-finite" in w for w in why) for _, why in viol):
            print("  non-finite values cannot be fixed by fp32: the checkpoint itself is broken.")
    else:
        print("PASS: every tensor casts to fp16 without overflow / significant underflow"
              + (f" (fp32 allow-list: {fp32_modules})" if fp32_modules else ""))
    return dict(path=path, ok=not viol, totals=tot, violations=viol, tensors=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help=".safetensors (DiT) or .pth files")
    ap.add_argument("--fp32-modules", nargs="*", default=[], help="modules exempt from range rules")
    ap.add_argument("--underflow-frac", type=float, default=1e-3)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json", help="write the full per-tensor report here")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    reports = [scan(p, args.fp32_modules, args.underflow_frac, args.top) for p in args.paths]
    ok = all(r["ok"] for r in reports)
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(reports, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
