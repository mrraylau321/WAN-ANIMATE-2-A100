#!/usr/bin/env python
"""Probe what a GPU really supports for this port (run on the V100 server; never crashes).

Prints: device / torch / CUDA / cuDNN versions, torch.cuda.is_bf16_supported() with and without
emulation, timed 8192^3 matmul TFLOPS (fp16 / bf16 / fp32), a VAE-like conv3d in fp16 / bf16 / fp32,
and every SDPA backend (FLASH, EFFICIENT, CUDNN, MATH) x (fp16, bf16) at Wan head_dim 128 with
time and peak memory. Every case is caught and reported (ok / error first line).

  python scripts/probe_v100.py [--json probe.json]
  python scripts/probe_v100.py --cpu-smoke     # tiny sizes on CPU, only to check the script itself
"""
import argparse
import json
import platform
import time

import torch
import torch.nn.functional as F

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def timed(fn, dev, warmup=2, iters=5):
    """Returns (ok, ms_per_iter, peak_extra_MB, error)."""
    try:
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            base = torch.cuda.memory_allocated(dev)
        for _ in range(warmup):
            fn()
        _sync(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        _sync(dev)
        ms = (time.perf_counter() - t0) * 1e3 / iters
        peak = (torch.cuda.max_memory_allocated(dev) - base) / 2**20 if dev.type == "cuda" else float("nan")
        return True, ms, peak, ""
    except Exception as e:  # report, never crash
        msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        return False, float("nan"), float("nan"), f"{type(e).__name__}: {msg[:160]}"
    finally:
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def row(table, section, case, ok, ms, peak, err, extra=""):
    table.append(dict(section=section, case=case, ok=ok, ms=ms, peak_mb=peak, error=err, extra=extra))
    status = "ok " if ok else "FAIL"
    ms_s = f"{ms:9.2f} ms" if ok else " " * 12
    pk = f"{peak:9.0f} MB" if ok and peak == peak else " " * 12
    print(f"  {status} {case:<44} {ms_s} {pk}  {extra}{err}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="also write the table as JSON")
    ap.add_argument("--cpu-smoke", action="store_true", help="tiny shapes on CPU (script self-test only)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if not args.cpu_smoke and not torch.cuda.is_available():
        print("probe_v100: CUDA is not available on this machine - nothing to probe. "
              "Run it on the V100 server (or use --cpu-smoke to test the script).")
        return 0
    dev = torch.device("cpu") if args.cpu_smoke else torch.device(args.device)
    small = args.cpu_smoke
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(dev)}
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        p = torch.cuda.get_device_properties(dev)
        info.update(name=p.name, capability=f"sm{p.major}{p.minor}", total_mem_GB=round(p.total_memory / 2**30, 1),
                    sm_count=p.multi_processor_count, cudnn=torch.backends.cudnn.version(),
                    torch_arch_list=" ".join(torch.cuda.get_arch_list()))  # must contain sm_70 for V100
        try:
            info["is_bf16_supported()"] = torch.cuda.is_bf16_supported()
            info["is_bf16_supported(including_emulation=False)"] = torch.cuda.is_bf16_supported(
                including_emulation=False)
        except TypeError:  # older torch without the flag
            info["is_bf16_supported(including_emulation=False)"] = "n/a (torch < 2.3)"
        try:
            info["is_tf32_supported()"] = torch.cuda.is_tf32_supported()
        except AttributeError:
            pass
    print("== device")
    for k, v in info.items():
        print(f"  {k:<48} {v}")
    table = []

    print("\n== matmul (TFLOPS = 2 n^3 / time)")
    n = 256 if small else 8192
    for name, dt in DTYPES.items():
        try:
            a = torch.randn(n, n, device=dev, dtype=dt)
            b = torch.randn(n, n, device=dev, dtype=dt)
        except Exception as e:
            row(table, "matmul", f"{name} {n}^3", False, float("nan"), float("nan"), f"alloc: {e}")
            continue
        ok, ms, pk, err = timed(lambda: a @ b, dev)
        row(table, "matmul", f"{name} {n}^3", ok, ms, pk, err,
            extra=f"{2 * n ** 3 / (ms * 1e-3) / 1e12:7.2f} TFLOPS  " if ok else "")
        del a, b

    print("\n== conv3d (VAE-like: 96->96 ch, k=3, T=4, 360x640; FLOP-equivalent TFLOPS)")
    c, t, h, w = (8, 2, 16, 16) if small else (96, 4, 360, 640)
    for name, dt in DTYPES.items():
        try:
            conv = torch.nn.Conv3d(c, c, 3, padding=1).to(dev, dt)
            x = torch.randn(1, c, t, h, w, device=dev, dtype=dt)
        except Exception as e:
            row(table, "conv3d", name, False, float("nan"), float("nan"), f"alloc: {e}")
            continue
        flops = 2 * c * c * 27 * t * h * w
        with torch.no_grad():
            ok, ms, pk, err = timed(lambda: conv(x), dev)
        row(table, "conv3d", name, ok, ms, pk, err,
            extra=f"{flops / (ms * 1e-3) / 1e12:7.2f} TFLOPS  " if ok else "")
        del conv, x

    print("\n== SDPA backends (head_dim 128, 5 heads = 40 heads / 8-way context parallel)")
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        print("  torch.nn.attention.sdpa_kernel not available in this torch")
        SDPBackend = None
    if SDPBackend is not None:
        backends = [getattr(SDPBackend, b) for b in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION",
                                                      "CUDNN_ATTENTION", "MATH") if hasattr(SDPBackend, b)]
        shapes = [("self L=8192", 8192, 8192, False),
                  ("720P frame chunk Lq=3600 Lk=82800 +bias", 3600, 82800, True)]
        if small:
            shapes = [("self L=256", 256, 256, False), ("chunk Lq=64 Lk=512 +bias", 64, 512, True)]
        for label, lq, lk, with_bias in shapes:
            for dname in ("fp16", "bf16"):
                dt = DTYPES[dname]
                try:
                    q = torch.randn(1, 5, lq, 128, device=dev, dtype=dt)
                    k = torch.randn(1, 5, lk, 128, device=dev, dtype=dt)
                    v = torch.randn(1, 5, lk, 128, device=dev, dtype=dt)
                    bias = torch.zeros(1, 1, 1, lk, device=dev, dtype=dt) if with_bias else None
                    if bias is not None:
                        bias[..., : lk // 20] = -1.3
                except Exception as e:
                    row(table, "sdpa", f"{label} {dname}", False, float("nan"), float("nan"), f"alloc: {e}")
                    continue
                for be in backends:
                    def run(be=be):
                        with sdpa_kernel([be]):
                            return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
                    ok, ms, pk, err = timed(run, dev, warmup=1, iters=3)
                    flops = 4 * 5 * lq * lk * 128
                    row(table, "sdpa", f"{label} {dname} {be.name}", ok, ms, pk, err,
                        extra=f"{flops / (ms * 1e-3) / 1e12:7.2f} TFLOPS  " if ok else "")
                del q, k, v, bias

    print("\nNotes: bf16 rows that run on sm<80 are emulated / non-tensor-core paths; SDPA FLASH and "
          "CUDNN need sm80+; EFFICIENT on sm<80 accepts only fp16/fp32, so bf16 falls back to MATH "
          "(full score matrix). See NOTES_V100.md 'BF16 on V100'.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(info=info, table=table), f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
