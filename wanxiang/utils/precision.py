"""Capability switch for the V100 port: choose compute dtypes per GPU.

sm>=80 (A800/A100/H100) with everything on 'auto' -> exactly the upstream recipe
(bf16 DiT/T5/VAE, fp16 CLIP). sm<80 (V100) -> fp16 DiT, fp32 T5/VAE, fp16 CLIP.

Precedence for the DiT dtype: test_cfg.dit_dtype (set by ``--dtype``) > $WAN_DTYPE > auto.
"""
import os
from dataclasses import dataclass

import torch

_DTYPES = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
    "fp32": torch.float32, "float32": torch.float32, "float": torch.float32,
}


def parse_dtype(name):
    """'auto'/None -> None, otherwise a torch dtype ('fp16', 'bfloat16', torch.float32, ...)."""
    if name is None or isinstance(name, torch.dtype):
        return name
    name = str(name).strip().lower().replace("torch.", "")
    if name in ("", "auto", "none"):
        return None
    if name not in _DTYPES:
        raise ValueError(f"unknown dtype {name!r}; use auto|fp16|bf16|fp32")
    return _DTYPES[name]


def has_native_bf16(device):
    """True only for real bf16 tensor-core hardware (sm>=80); V100 emulates bf16."""
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


@dataclass
class Precision:
    dit: torch.dtype         # DiT autocast + FSDP2 param_dtype
    dit_output: torch.dtype  # FSDP2 output_dtype of DiT units (residual stream between blocks)
    t5: torch.dtype          # T5 compute dtype (FSDP1 param_dtype); weights stay stored as loaded
    vae: torch.dtype         # VAE compute dtype (and dtype of the pixel tensors fed to it)
    clip: torch.dtype        # CLIP visual compute dtype
    guard: str = "off"               # OverflowGuard mode (off|first_step|all)
    clip_text_on_gpu: bool = True    # upstream moves the (unused) CLIP text tower to the GPU
    underflow_frac: float = 1e-3     # fp16 load check: tolerated flushed fraction per module

    @property
    def upstream(self):
        return self.dit == torch.bfloat16

    def describe(self):
        n = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
        return (f"dit={n[self.dit]} dit_output={n[self.dit_output]} t5={n[self.t5]} "
                f"vae={n[self.vae]} clip={n[self.clip]} guard={self.guard} "
                f"clip_text_on_gpu={self.clip_text_on_gpu}")


def _get(cfg, key, default=None):
    try:
        val = cfg.get(key, default)
    except AttributeError:
        val = getattr(cfg, key, default)
    return default if val is None else val


def _auto(val):
    return val is None or str(val).strip().lower() in ("", "auto", "none")


T5_FP16_ERROR = (
    "T5 in fp16 is not supported (t5_dtype / param_dtype resolved to float16). umT5 activations are "
    "known to overflow fp16, and the encoder's fp16_clamp turns the resulting inf into finite "
    "+-64512 values, so the overflow would be silent (a wrong but finite text embedding). "
    "Use t5_dtype=fp32 (the V100 default) or bf16.")


def _check_t5(dtype):
    if dtype == torch.float16:
        raise ValueError(T5_FP16_ERROR)
    return dtype


def resolve_precision(test_cfg, device):
    """Every V100-specific knob is 'auto' by default and resolves to the upstream value when the
    DiT runs in bf16 (sm>=80 or --dtype bf16), so the *_v100.yaml files behave exactly like the
    upstream YAMLs on A800/H100."""
    _check_t5(parse_dtype(_get(test_cfg, "t5_dtype", "auto")))  # refused on every path
    req = parse_dtype(_get(test_cfg, "dit_dtype", "auto")) or parse_dtype(os.environ.get("WAN_DTYPE"))
    dit = req or (torch.bfloat16 if has_native_bf16(device) else torch.float16)
    guard = _get(test_cfg, "overflow_guard", "auto")
    text_gpu = _get(test_cfg, "clip_text_on_gpu", "auto")
    frac = float(_get(test_cfg, "fp16_underflow_frac", 1e-3))
    if dit == torch.bfloat16:
        # Upstream recipe, unchanged (T5 compute follows test_cfg.param_dtype as before).
        return Precision(dit=torch.bfloat16, dit_output=torch.bfloat16,
                         t5=_check_t5(parse_dtype(_get(test_cfg, "param_dtype", "bfloat16"))),
                         vae=torch.bfloat16, clip=torch.float16,
                         guard="off" if _auto(guard) else str(guard),
                         clip_text_on_gpu=True if _auto(text_gpu) else bool(text_gpu),
                         underflow_frac=frac)
    return Precision(
        dit=dit,
        # keep the residual stream fp32 between blocks: fp16 would add an overflow point
        # that bf16 (upstream) does not have; costs ~seq*dim*4 bytes per rank.
        dit_output=torch.float32,
        # T5 runs once per prompt; fp32 avoids the known umT5 fp16 overflow (see NOTES_V100.md)
        t5=parse_dtype(_get(test_cfg, "t5_dtype", "auto")) or torch.float32,
        vae=parse_dtype(_get(test_cfg, "vae_dtype", "auto")) or torch.float32,
        clip=parse_dtype(_get(test_cfg, "clip_dtype", "auto"))
        or (torch.float32 if dit == torch.float32 else torch.float16),
        guard=("first_step" if dit == torch.float16 else "off") if _auto(guard) else str(guard),
        clip_text_on_gpu=False if _auto(text_gpu) else bool(text_gpu),
        underflow_frac=frac,
    )
