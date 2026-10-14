#!/usr/bin/env python3
"""Validate the two visible A100 80 GiB CUDA devices before launching the model."""
import argparse
import sys


def inspect_devices(torch_module, count=2, min_mib=78000):
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise RuntimeError("CUDA unavailable; run on the 2× A100 80GB server, not this Mac")
    visible = cuda.device_count()
    if visible != count:
        raise RuntimeError(f"Expected exactly {count} visible A100 GPUs; found {visible}")
    info = []
    for rank in range(count):
        props = cuda.get_device_properties(rank)
        major, minor = props.major, props.minor
        capacity = props.total_memory / (1024 ** 2)
        if "A100" not in props.name or (major, minor) != (8, 0):
            raise RuntimeError(
                f"rank {rank}: expected A100 sm80; got {props.name} sm{major}{minor}"
            )
        if capacity < min_mib:
            raise RuntimeError(
                f"rank {rank}: {capacity:.0f} MiB is not an 80GB A100 profile "
                f"(requires at least {min_mib} MiB; Ray reports 81920 MiB)"
            )
        info.append((rank, props.name, capacity, f"sm{major}{minor}"))
    if not cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native BF16 CUDA support not reported for A100")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-mib", type=int, default=78000)
    args = parser.parse_args()
    try:
        import torch
        for rank, name, mib, cc in inspect_devices(torch, min_mib=args.min_mib):
            print(f"[A100 preflight] rank={rank}: {name}, {mib:.0f} MiB, {cc}, native BF16")
        print(f"[A100 preflight] torch={torch.__version__} CUDA={torch.version.cuda}")
    except (ImportError, RuntimeError) as exc:
        sys.exit(f"[A100 preflight FAILED] {exc}")


if __name__ == "__main__":
    main()
