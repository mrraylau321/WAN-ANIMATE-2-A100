#!/usr/bin/env python
"""Per-phase, per-GPU estimates for V100 32GB and experimental 8x16GB recipes.

No GPU is needed, but the reported peaks are not CUDA measurements and are NOT an OOM guarantee.

How: persistent buffers are counted analytically from the real parameter counts (release
checkpoint header + model definitions on the meta device). Transient peaks are TRACED: the real
VAE encode/decode and one full-size DiT block (forward_ref / forward_gen, per-rank shapes, the
context-parallel all_to_all buffers emulated by ops.all_to_all itself) run under FakeTensorMode
and every allocation is followed until it is freed. Kernel workspaces of fused SDPA / cuBLAS /
cuDNN, the CUDA context, NCCL buffers and allocator fragmentation are NOT traced (see 'overhead').
It is an ESTIMATE - measure on the server (nvidia-smi / torch.cuda.max_memory_allocated).

  python scripts/estimate_v100_memory.py --profile 32gb [--res 480p 540p 720p]
  python scripts/estimate_v100_memory.py --profile 16gb --res 256x448 --clip-len 17
  python scripts/estimate_v100_memory.py --profile 16gb --res 288x512 --clip-len 25
"""
import argparse
import json
import math
import os
import sys
import warnings
import weakref
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import torch  # noqa: E402
from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402
from torch.distributed._tools.common_utils import get_untyped_storages  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils._pytree import tree_flatten  # noqa: E402
from torch.utils.weak import WeakIdKeyDictionary  # noqa: E402

GiB = 2 ** 30
RES = {"256x448": (256, 448), "288x512": (288, 512),
       "480p": (480, 832), "540p": (544, 960), "720p": (720, 1280)}
CAPACITY_GIB = 32510 / 1024  # V100-SXM2/PCIe 32GB as reported by nvidia-smi
CAPACITY_16G_GIB = 15.75  # conservative; override with actual nvidia-smi memory.total / 1024


class LiveMem(TorchDispatchMode):
    """Peak of live tensor storage allocated inside the mode (freed when the last ref dies)."""

    def __init__(self):
        super().__init__()
        self.live, self.cur, self.peak = WeakIdKeyDictionary(), 0, 0

    def _free(self, n):
        self.cur -= n

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor):
                for st in get_untyped_storages(t):
                    if st not in self.live:
                        n = st.nbytes()
                        self.live[st] = n
                        self.cur += n
                        weakref.finalize(st, self._free, n)
        self.peak = max(self.peak, self.cur)
        return out


@contextmanager
def cuda_like_autocast():
    """Map the model's autocast('cuda', float32) regions to CPU 'autocast off' (as on CUDA)."""
    real = torch.amp.autocast

    def shim(device_type, dtype=None, enabled=True, cache_enabled=None):
        if device_type == "cuda":
            return real("cpu", enabled=False) if dtype == torch.float32 else real("cpu", dtype=dtype, enabled=enabled)
        return real(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)

    torch.amp.autocast = shim
    try:
        yield
    finally:
        torch.amp.autocast = real


def param_counts():
    hdr = json.load(open(os.path.join(ROOT, "tests", "fixtures", "wan_animate_2_ckpt_header.json")))["tensors"]
    from wanxiang.models.clip import clip_xlm_roberta_vit_h_14
    from wanxiang.models.t5 import umt5_xxl
    from wanxiang.models.vae import VideoVAE_
    dit = sum(math.prod(s) for _, s in hdr.values())
    block = sum(math.prod(s) for k, (_, s) in hdr.items() if k.startswith("blocks.0."))
    t5 = umt5_xxl(encoder_only=True, device="meta")
    clip = clip_xlm_roberta_vit_h_14(device="meta")
    with torch.device("meta"):
        vae = VideoVAE_(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                        temperal_downsample=[False, True, True])
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    return dict(dit=dit, block=block, dit_root=dit - 40 * block, t5=n(t5),
                t5_embedding=t5.token_embedding.weight.numel(),
                t5_block=n(t5.blocks[0]), clip_vis=n(clip.visual), clip_txt=n(clip.textual), vae=n(vae),
                ffn_weight=13824 * 5120,
                root_fp32_islands=sum(math.prod(s) for k, (_, s) in hdr.items()
                                      if k.startswith(("time_embedding.", "time_projection.", "head."))))


def geometry(w, h, frames, clip_len, gpus):
    clip_frames = min(clip_len, frames)
    T = clip_frames + 1
    lat_h, lat_w = h // 8, w // 8
    lat_t = T // 4 + 1 + 1                      # generated latent frames incl. reference image
    ref_t = (clip_frames - 1) // 4 + 1          # driving-video latent frames
    tok = lat_t * lat_h * lat_w // 4
    tok_ref = ref_t * lat_h * lat_w // 4
    seq = math.ceil(tok / gpus) * gpus
    seq_ref = math.ceil(tok_ref / gpus) * gpus
    F_ = clip_len // 4 + 1
    hw = w * h // 256
    q_lim, k_lim = (F_ + 1) * hw, F_ * hw
    q_tot = math.ceil(q_lim / 128) * 128
    kv_tot = q_tot + math.ceil(k_lim / 128) * 128
    return dict(clip_frames=clip_frames, lat_t=lat_t, ref_t=ref_t, lat_h=lat_h, lat_w=lat_w,
                seq=seq, seq_ref=seq_ref, hw=hw, F=F_, q_lim=q_lim, q_tot=q_tot, kv_tot=kv_tot,
                heads=40 // gpus)


# ----------------------------------------------------------------------------- traces
def trace_vae(g, w, h):
    from wanxiang.models.vae import VideoVAE_
    res = {}
    with FakeTensorMode():
        vae = VideoVAE_(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                        temperal_downsample=[False, True, True]).eval()
        scale = [torch.zeros(16), torch.ones(16)]
        x = torch.randn(1, 3, g["clip_frames"], h, w)
        lm = LiveMem()
        with lm, torch.no_grad():
            vae.encode(x, scale)
            vae.clear_cache()
        res["encode"] = lm.peak
        z = torch.randn(1, 16, g["lat_t"] - 1, g["lat_h"], g["lat_w"])
        lm = LiveMem()
        with lm, torch.no_grad():
            out = vae.decode(z, scale)
            vae.clear_cache()
            frames = torch.stack([out[0]])           # pipeline: stack, check, clamp, then
            img = (frames + 1) * 127.5                # rearrange(((out_frames + 1) * 127.5)) on GPU
            del out, img
        res["decode"] = lm.peak
    return res


class _Row(list):
    def tolist(self):
        return list(self)


class _Grid(list):
    """Stand-in for the [1, 3] grid_sizes tensor (the model only calls .tolist() on it; a fake
    tensor cannot return data)."""

    def tolist(self):
        return [list(r) for r in self]

    def __getitem__(self, i):
        return _Row(list.__getitem__(self, i))


def trace_block(g, mode, log_scale=-1.3):
    """Transient peak of ONE Incontext_AttentionBlock forward_ref and forward_gen on one rank."""
    from wanxiang import ops
    import wanxiang.ops.distributed as D
    from wanxiang.models import wan_animate_2_model as M
    from wanxiang.models import attention as A
    from wanxiang.models import attention_sdpa as S
    G = 40 // g["heads"]
    patches = [(D, "get_world_size", lambda group=None: G), (D.dist, "all_to_all", lambda *a, **k: None),
               (ops, "get_tensor_parallel_group", lambda: None)]
    if mode == "math":
        patches.append((S, "_allowed_fused", lambda: []))
    saved = [(o, n, getattr(o, n)) for o, n, _ in patches]
    for o, n, f in patches:
        setattr(o, n, f)
    A.set_attention_backend("sdpa")
    dt = torch.float32 if mode == "fp32" else torch.float16
    S.set_compute_dtype(dt)
    out = {}
    try:
        with FakeTensorMode(), cuda_like_autocast():
            blk = M.Incontext_AttentionBlock(5120, 13824, 40, (-1, -1), True, True, 1e-6, 1,
                                             use_img_emb=True, use_context_parallel=True,
                                             log_scale=log_scale).eval().to(dt)
            d = 128
            freqs = torch.cat([M.rope_params(512, d - 4 * (d // 6)), M.rope_params(512, 2 * (d // 6)),
                               M.rope_params(512, 2 * (d // 6))], dim=1)
            e = torch.randn(1, 6, 5120)
            ctx = torch.randn(1, 257 + 512, 5120, dtype=torch.float16)
            gs = _Grid([[g["lat_t"], g["lat_h"] // 2, g["lat_w"] // 2]])  # only .tolist() is used
            gs_ref = _Grid([[g["ref_t"], g["lat_h"] // 2, g["lat_w"] // 2]])
            ac = torch.autocast("cpu", dtype=torch.float16) if dt == torch.float16 else \
                torch.autocast("cpu", enabled=False)
            # forward_ref (cache written)
            x_ref = torch.randn(1, g["seq_ref"] // G, 5120)
            kc, vc = {}, {}
            lm = LiveMem()
            with lm, ac, torch.no_grad():
                blk(x_ref, 0, kc, vc, ctx, freqs, gs_ref, e, None, method="forward_ref")
            out["forward_ref"] = lm.peak
            # forward_gen with a full cache and the In-Context mask plan
            kc = {0: torch.empty(1, g["seq_ref"], g["heads"], d, dtype=dt)}
            vc = {0: torch.empty(1, g["seq_ref"], g["heads"], d, dtype=dt)}
            runs = [(0, g["hw"], torch.empty(g["q_lim"], dtype=torch.long))]
            runs += [(f * g["hw"], (f + 1) * g["hw"], torch.empty(g["q_lim"] + g["hw"], dtype=torch.long))
                     for f in range(1, g["F"] + 1)]
            runs += [(g["q_lim"], g["q_tot"], torch.empty(0, dtype=torch.long))]
            plan = S.MaskPlan(g["q_tot"], g["kv_tot"], runs)
            sm = M._get_score_mod(hw=g["hw"], log_scale=log_scale)
            plan._bias[(sm, dt)] = [None if c.numel() == 0 else torch.zeros(1, 1, 1, c.numel(), dtype=dt)
                                    for _, _, c in runs]
            x = torch.randn(1, g["seq"] // G, 5120)
            lm = LiveMem()
            with lm, ac, torch.no_grad():
                blk(x, 0, kc, vc, plan, ctx, freqs, freqs, gs, gs_ref, 4 * (g["F"] - 1) + 1,
                    [g["hw"] * 16, 16], e, None, method="forward_gen")
            out["forward_gen"] = lm.peak
    finally:
        for o, n, f in saved:
            setattr(o, n, f)
        S.set_compute_dtype(torch.bfloat16)
    return out


# ----------------------------------------------------------------------------- phases
def estimate(name, w, h, frames, clip_len, gpus, P, dit_shard_dtype="fp32", fp32_islands=0,
             root_fp32_islands=False):
    if dit_shard_dtype not in ("fp32", "bf16"):
        raise ValueError("dit_shard_dtype must be fp32 or bf16")
    if not 0 <= fp32_islands <= 40:
        raise ValueError("fp32_islands must be between 0 and 40")
    g = geometry(w, h, frames, clip_len, gpus)
    vae = trace_vae(g, w, h)
    blocks = {m: trace_block(g, m) for m in ("fused", "math", "fp32")}
    root_islands = P["root_fp32_islands"] if root_fp32_islands else 0
    persistent = {
        f"DiT {dit_shard_dtype} shards (FSDP2 storage)": P["dit"] * (2 if dit_shard_dtype == "bf16" else 4) / gpus,
        "DiT root params unsharded after forward (fp16; excluding FP32 islands)": (P["dit_root"] - root_islands) * 2,
        "T5 bf16 shards (FSDP1)": P["t5"] * 2 / gpus,
        "CLIP visual fp16 shards (FSDP1)": P["clip_vis"] * 2 / gpus,
        "VAE fp32 weights": P["vae"] * 4,
    }
    if root_islands and dit_shard_dtype == "bf16":
        persistent["FP32 timestep/head island shard delta (from BF16)"] = root_islands * 2 / gpus
    if fp32_islands and dit_shard_dtype == "bf16":
        persistent["FP32 shard delta for allow-listed blocks"] = fp32_islands * P["block"] * 2 / gpus
    pers = sum(persistent.values())
    kv = 2 * 40 * g["seq_ref"] * g["heads"] * 128 * 2
    gathered = 2 * P["block"] * 2  # current + prefetched block, fp16
    plan = sum((g["q_lim"] + g["hw"]) * 8 for _ in range(g["F"])) + g["q_lim"] * 8
    small = (16 * g["lat_t"] * g["lat_h"] * g["lat_w"] * 4) * 6 + (40 * g["lat_h"] * g["lat_w"]
                                                                   * (g["lat_t"] + g["ref_t"]) * 2)
    pixels = 2 * 3 * g["clip_frames"] * h * w * 4  # batch tensor + the concat fed to the VAE (fp32)
    t5_gather = P["t5_embedding"] * 4 + P["t5_embedding"] * 4 / gpus  # fp32 embedding unit + cast shard
    phases = {
        "1 conditioning (VAE encode / T5 / CLIP)": pers + pixels + max(vae["encode"], t5_gather),
        "2 reference pass (forward_ref)": pers + kv + gathered + blocks["fused"]["forward_ref"],
        "3 denoising, fused SDPA fp16 (default)": pers + kv + gathered + plan + small + blocks["fused"]["forward_gen"],
        "3' denoising, MATH SDPA (no fused kernel)": pers + kv + gathered + plan + small + blocks["math"]["forward_gen"],
        "3'' denoising, a block in fp32 fallback": pers + kv + gathered + plan + small
        + blocks["fp32"]["forward_gen"] + P["ffn_weight"] * 4,
        "4 VAE decode (K/V cache freed)": pers + small + vae["decode"],
    }
    return dict(name=name, g=g, persistent=persistent, kv=kv, gathered=gathered, vae=vae, blocks=blocks,
                t5_gather=t5_gather, pixels=pixels, phases=phases)


def time_per_forward(g, gpus, P, attn_tflops, gemm_tflops, link_gbps):
    """Seconds per DiT forward per GPU from FLOPs and assumed achieved throughput (very rough)."""
    L, D, heads = 40, 128, g["heads"]
    att = L * 4 * D * heads * g["hw"] * (g["F"] * (g["q_lim"] + g["hw"]) + g["q_lim"])  # In-Context self-attn
    tok = g["seq"] / gpus
    lin = L * 2 * tok * (4 * 5120 ** 2 + 2 * 5120 ** 2 + 2 * 5120 * 13824) + L * 2 * 769 * 4 * 5120 ** 2
    comm = (P["dit"] * 2 * (gpus - 1) / gpus + L * tok * 5120 * (3 * 2 + 4) * (gpus - 1) / gpus) / (link_gbps * 1e9)
    return att / (attn_tflops * 1e12) + lin / (gemm_tflops * 1e12) + comm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=["32gb", "16gb"], default="32gb")
    ap.add_argument("--res", nargs="+", default=None, choices=list(RES))
    ap.add_argument("--frames", type=int, default=65, help="frames after fps resampling + padding")
    ap.add_argument("--clip-len", type=int, default=None)
    ap.add_argument("--capacity-gib", type=float, default=None, help="actual usable per-card GiB")
    ap.add_argument("--dit-shard-dtype", choices=["fp32", "bf16"], default=None)
    ap.add_argument("--fp32-islands", type=int, default=0, help="number of DiT blocks retaining fp32 weights")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--overhead-gib", type=float, default=1.5,
                    help="untraced: CUDA context (~0.4), NCCL buffers, kernel workspaces, fragmentation")
    ap.add_argument("--attn-tflops", type=float, default=40.0, help="assumed achieved fp16 SDPA TFLOPS")
    ap.add_argument("--gemm-tflops", type=float, default=80.0, help="assumed achieved fp16 GEMM TFLOPS")
    ap.add_argument("--link-gbps", type=float, default=100.0, help="per-GPU collective GB/s (NVLink ~100, PCIe ~10)")
    ap.add_argument("--json")
    a = ap.parse_args()
    low_vram = a.profile == "16gb"
    a.res = a.res or (["256x448", "288x512", "480p"] if low_vram else ["480p", "540p", "720p"])
    a.clip_len = a.clip_len or (17 if low_vram else 81)
    a.capacity_gib = a.capacity_gib or (CAPACITY_16G_GIB if low_vram else CAPACITY_GIB)
    a.dit_shard_dtype = a.dit_shard_dtype or ("bf16" if low_vram else "fp32")
    if a.gpus < 1 or 40 % a.gpus:
        ap.error("GPU count must divide the model's 40 attention heads")
    if a.clip_len < 5 or (a.clip_len - 1) % 4:
        ap.error("--clip-len must be 4n+1 (e.g. 17, 25 or 81)")
    if a.frames < 1 or a.capacity_gib <= 0 or not 0 <= a.fp32_islands <= 40:
        ap.error("--frames/--capacity-gib/--fp32-islands values are invalid")
    P = param_counts()
    print("params: " + ", ".join(f"{k}={v / 1e9:.3f}B" for k, v in P.items()))
    print(f"profile={a.profile}, DiT shard={a.dit_shard_dtype}, {a.gpus} GPUs, capacity={a.capacity_gib:.2f} "
          f"GiB/card (assumed; use actual nvidia-smi total), +{a.overhead_gib} GiB untraced overhead; "
          "OK < 80% of capacity, TIGHT < 100%, else OVER")
    print("WARNING: offline phase estimate is NOT a CUDA/NCCL memory measurement or OOM-free guarantee")
    report = []
    for r in a.res:
        w, h = RES[r]
        e = estimate(r, w, h, a.frames, a.clip_len, a.gpus, P,
                     dit_shard_dtype=a.dit_shard_dtype, fp32_islands=a.fp32_islands,
                     root_fp32_islands=low_vram)
        report.append(e)
        g = e["g"]
        print(f"\n== {r} ({w}x{h}), {g['clip_frames']}-frame clip: seq/rank {g['seq'] // a.gpus}, "
              f"ref seq {g['seq_ref']}, In-Context q/kv {g['q_tot']}/{g['kv_tot']}")
        for k, v in e["persistent"].items():
            print(f"  {v / GiB:6.2f}  persistent: {k}")
        print(f"  {e['kv'] / GiB:6.2f}  reference K/V cache (owned storage, fp16)")
        print(f"  {e['gathered'] / GiB:6.2f}  gathered fp16 block params (current + prefetch)")
        print(f"  {e['pixels'] / GiB:6.2f}  fp32 pixel tensors during conditioning (freed before denoising)")
        print(f"  {e['vae']['encode'] / GiB:6.2f}  traced: VAE encode transient   |  "
              f"{e['t5_gather'] / GiB:5.2f} T5 fp32 embedding gather")
        print(f"  {e['vae']['decode'] / GiB:6.2f}  traced: VAE decode transient (+ stack/scale copies)")
        for m, d in e["blocks"].items():
            print(f"  {d['forward_ref'] / GiB:6.2f} / {d['forward_gen'] / GiB:5.2f}  traced: one block "
                  f"forward_ref / forward_gen transient [{m}]")
        t = time_per_forward(g, a.gpus, P, a.attn_tflops, a.gemm_tflops, a.link_gbps)
        print(f"  time (assumed {a.attn_tflops:g}/{a.gemm_tflops:g} TFLOPS, {a.link_gbps:g} GB/s): ~{t:.1f} s per "
              f"forward -> Base 80 forwards ~{80 * t / 60:.0f} min, Distillation 10 ~{10 * t / 60:.1f} min per clip")
        print("  phase totals (+overhead):")
        for k, v in e["phases"].items():
            t = v / GiB + a.overhead_gib
            flag = "OK" if t < 0.8 * a.capacity_gib else ("TIGHT" if t < a.capacity_gib else "OVER")
            print(f"    {t:6.2f} GiB  {flag:5s}  {k}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump([{**e, "g": e["g"]} for e in report], f, indent=1, default=float)


if __name__ == "__main__":
    main()
