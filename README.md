# Wan-Animate-2 — 2× NVIDIA A100 80GB

**Unofficial dual-A100 inference adaptation.** This repository is for **two A100 cards with 81920 MiB (80 GiB) VRAM on each card**. It is separate from the [V100 project](https://github.com/mrraylau321/WAN-ANIMATE-2-V100) and does not require eight GPUs. Original Wan-Animate-2 model, paper, and code credit: [Wan-Video/Wan-Animate-2](https://github.com/Wan-Video/Wan-Animate-2) ([arXiv:2608.06009](https://arxiv.org/abs/2608.06009)). No official weights were altered or redistributed.

> **Status: CPU/gloo verified; actual A100 end-to-end inference NOT YET VERIFIED.** GPU VRAM estimates are offline proxies, not CUDA measurements or an OOM-free guarantee. Even a numerically finite output must be visually reviewed, especially at video chunk boundaries.

## Hardware and model layout

Both GPUs participate in **context parallelism (SP=2)** and **FSDP sharding (2-way)**. One parent Python process starts two worker processes; **do not start two copies or use `torchrun --nproc_per_node=2`**.

| Component | Placement / compute | Estimated resident weights, **per A100** |
| --- | --- | ---: |
| DiT, 16.395B parameters | BF16 checkpoint-exact FSDP2 shards, native BF16 compute; gather individual blocks during forward | ~15.27 GiB |
| Timestep embedding / projection + output head | 2-way **FP32 FSDP2 islands**, necessary for FP32 operations; includes BF16-to-FP32 shard overhead | ~0.17 GiB extra |
| DiT unsharded FSDP2 root remainder | BF16 root (outside the three FP32 islands) | ~0.10 GiB |
| umT5-XXL, 5.681B parameters | 2-way BF16 FSDP1 with isolated embedding/norm wrap; BF16 compute | ~5.29 GiB |
| CLIP visual encoder | 2-way FP16 FSDP1 | ~0.59 GiB |
| CLIP text encoder | **CPU only** (unused by inference) | 0 GiB on GPU |
| Video VAE | Replicated, FP32 stored weights with native BF16 autocast | ~0.47 GiB |
| **Total persistent GPU weights (estimated)** | Before transient FSDP gather, K/V, activations, CUDA overhead | **~21.90 GiB/card** |

A100 SM80 supports native BF16; **FP16 conversion is not required**. In the original 8-V100/32GB layout DiT parameters were distributed eight ways; here each A100 holds half of the BF16 parameter shards. The A100-specific `test_cfg.kv_cache_owned=true` ensures K/V caches **own their tensor storage even with FlashAttention**, avoiding the context-parallel all-to-all Q buffer remaining pinned in memory. The low-VRAM memory lifecycle also frees VAE causal caches, transfers only the inter-chunk overlap frame to CPU, and frees K/V before VAE decode. The model's FP32 timestep/head islands retain FP32 output rather than silently being converted back to BF16.

## Run on the two-A100 server

Use a PyTorch CUDA installation supporting SM80 and BF16; the inherited codebase was developed against PyTorch 2.7.1/CUDA 12.6. If compatible FlashAttention-2 is installed, the automatic attention route can use it; otherwise the port's SDPA alternative is available. **Check your actual CUDA/PyTorch driver combination before installing anything.** `ffmpeg` must be on PATH.

```bash
# Run from this new A100 repository, NOT from WAN-ANIMATE-2-V100.
git clone https://github.com/mrraylau321/WAN-ANIMATE-2-A100.git
cd WAN-ANIMATE-2-A100

conda create -n wan-a100 python=3.11 -y
conda activate wan-a100
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .
# A compatible flash-attn v2 installation is optional; SDPA is the fallback.

# Download the original official checkpoints (do not commit them).
huggingface-cli download Wan-AI/Wan2.2-Animate-2-14B --local-dir ./ckpts

# Verify exactly two A100 80GB devices are visible and native BF16 is supported.
CUDA_VISIBLE_DEVICES=0,1 python scripts/probe_a100.py

# Offline memory projection for a FULL first 720P 81-frame model chunk.
python scripts/estimate_a100_memory.py --res 720p --clip-len 81

# No-GPU launcher dry-run. Then start with a shorter 480P smoke run.
DRY_RUN=1 PRESET=smoke bash scripts/run_a100.sh
PRESET=smoke bash scripts/run_a100.sh

# Preferred initial 720P run: 65-frame model chunks, 10-step distilled model.
PRESET=standard MODE=distill bash scripts/run_a100.sh

# Longer 81-frame chunks, to retain more temporal context (higher peak).
PRESET=full MODE=distill bash scripts/run_a100.sh

# Base model: 40 steps and CFG=3, considerably more forward passes.
PRESET=standard MODE=base bash scripts/run_a100.sh
```

Supported environment variables: `MODE` = `distill|base`, `PRESET` = `smoke|standard|full`, `GPUS` = `0,1` by default (or any two distinct physical indices), `RES` = `480p|540p|720p` (also inherited small-size options), `CLIP_LEN` = `4n+1`, `REF_IMG`, `DRIVE`, `PROMPT`, `PROMPT_REF`, `CKPT_DIR`, `OUT_DIR`, `STEPS`, `GUIDE`, `FPS`, `SEED`, and `ATTN_BACKEND=auto|flash|sdpa`. The launcher deliberately disallows arbitrary `EXTRA_OVERRIDES` to avoid silently changing 2-way FSDP or BF16 storage. Its hardware preflight **rejects one GPU, V100, non-A100, and 40GB A100**. Before loading any model checkpoints it also calls `scripts/preflight_a100_config.py` to validate the actual YAML and construct a dummy `OverflowGuard`. Always quote YAML enum values such as `overflow_guard: "off"`; otherwise some YAML parsers turn `off` into Boolean `False`. Slow network-mounted checkpoints can make model initialization very long: the preflight avoids expensive late failures caused by configuration mistakes, but does not accelerate checkpoint I/O.

### Model presets

| Preset | Resolution | Chunk frames | Notes |
| --- | --- | ---: | --- |
| `smoke` | 480×832 | 33 | Initial hardware check |
| `standard` (default) | 720×1280 | 65 | Initial 720P choice |
| `full` | 720×1280 | 81 | Longer temporal context, higher peak |

For a 2×A100 80 GiB system, the offline projection uses a **3 GiB per GPU assumed untraced runtime overhead**, plus checkpoint-derived persistent sizes and FakeTensor tracing (FP16 SDPA/BF16-equivalent storage and FP32 VAE transient proxies). It projects approximately **62 GiB/card for the 720P/65-frame BF16 denoising phase** and **70 GiB/card for 720P/81-frame BF16 denoising**. These are **not actual CUDA measurements**; particular attention kernels, FSDP behavior, NCCL workspaces and allocator fragmentation can change the peak. Always inspect `[WEIGHTS cuda:*]` and `[VRAM cuda:*]` records for both ranks, including `peak_reserved`. CPU tests cannot verify A100 throughput, GPU memory fit, or visual quality.

### Validation and numeric quality

The A100 adaptation is validated with CPU distributed FSDP2 tests for two ranks: BF16 checkpoint loading, FP32 island semantics, native-BF16-style tiny forward passes, FP32 affine LayerNorm and owned FlashAttention K/V cache, plus hardware-preflight and launcher validation tests.

A real-server validation should confirm complete output, absence of NaN/Inf and OOM, stable identity/motion, and absence of exposure/color flashes at the **chunk boundaries** (particularly around the first generated frame after each chunk). Compare frames with the same source prompt, seed and geometry when possible. Passing PyTorch forward is not proof of perceptual quality.

## Repository provenance

This repo was independently cloned from the last verified V100-port commit `68e0560`, while leaving the source V100 repo unchanged. The inherited `*_v100.yaml`, launcher utilities, `NOTES_V100.md` and tests remain as technical references and regression coverage; **the supported entry point for this repo is exclusively `scripts/run_a100.sh` and the new `*_a100.yaml` files**. The upstream Apache-2.0 license remains in `LICENSE`.
