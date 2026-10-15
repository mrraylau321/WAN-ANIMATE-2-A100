#!/usr/bin/env python3
"""Conservative offline per-card memory projection for exactly two 80GiB A100 GPUs.

Uses checkpoint-shape-derived sizes and FakeTensor traces of the VAE/DiT from
the inherited reference estimator. VAE activation traces run FP32 and the
attention trace uses FP16 SDPA as a 2-byte proxy for native A100 BF16 kernels;
this is not an exact A100 CUDA allocation trace. CUDA/NCCL workspaces and
allocator fragmentation are not traced. These numbers MUST be checked with
real per-rank [VRAM] logs before making OOM-free/quality claims.
"""
import argparse

from estimate_v100_memory import RES, GiB, estimate, param_counts


def get_estimate(res="720p", clip_len=65, frames=None, overhead_gib=3.0):
    if clip_len < 5 or (clip_len - 1) % 4 != 0:
        raise ValueError("clip_len must be 4n+1")
    if res not in RES:
        raise ValueError("unknown resolution")
    frames = frames if frames is not None else clip_len
    if frames < 1:
        raise ValueError("frames must be positive")
    P = param_counts()
    w, h = RES[res]
    e = estimate(res, w, h, frames, clip_len, 2, P, dit_shard_dtype="bf16",
                 root_fp32_islands=True)
    peak_by_phase = {name: value / GiB + overhead_gib for name, value in e["phases"].items()}
    return e, peak_by_phase


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", nargs="+", choices=list(RES), default=["480p", "540p", "720p"])
    p.add_argument("--clip-len", type=int, default=65)
    p.add_argument("--frames", type=int, default=None, help="input frames in first model chunk (default: clip-len)")
    p.add_argument("--capacity-mib", type=int, default=81920,
                   help="per-GPU physical memory, MiB (an A100 80GB reports 81920 MiB)")
    p.add_argument("--overhead-gib", type=float, default=3.0,
                   help="assumed untraced kernel/NCCL/allocator overhead, GiB")
    a = p.parse_args()
    if a.capacity_mib <= 0 or a.overhead_gib < 0:
        p.error("capacity and overhead must be positive")
    cap = a.capacity_mib / 1024
    print(f"Dual A100-80GB profile: 2 GPUs, native BF16, {a.capacity_mib} MiB / {cap:.2f} GiB each")
    print("Model sharding: DiT=FSDP2 2-way BF16; T5=FSDP1 2-way BF16; CLIP visual=FSDP1 2-way FP16; VAE=replicated")
    print("The FP32 time_embedding/time_projection/head islands are separately FSDP2-sharded.")
    print("ESTIMATE ONLY. VAE FP32 and DiT FP16 traces are proxies; CUDA kernels/NCCL/fragmentation not measured.")
    for name in a.res:
        e, phase_peaks = get_estimate(name, a.clip_len, a.frames, a.overhead_gib)
        print(f"\n{name}: input={e['g']['clip_frames']} frames, clip_len={a.clip_len}")
        print(f"  persistent weights (estimated): {sum(e['persistent'].values())/GiB:.2f} GiB/card")
        print(f"  reference K/V: {e['kv']/GiB:.2f} GiB/card; two FSDP block prefetch: {e['gathered']/GiB:.2f} GiB/card")
        for phase, peak in phase_peaks.items():
            status = "ESTIMATED OVER" if peak >= cap else ("CLOSE" if peak >= 0.8 * cap else "ESTIMATED FIT")
            print(f"  {peak:5.2f}/{cap:.2f} GiB {status}: {phase}")
        worst = max(phase_peaks.values())
        print(f"  offline peak across phases: {worst:.2f} GiB, remaining: {cap-worst:.2f} GiB (not guaranteed)")


if __name__ == "__main__":
    main()
