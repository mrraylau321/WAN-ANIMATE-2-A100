"""SDPA replacements for the FlashAttention / flex_attention kernels (V100 / sm<80 path).

Added for the V100 port (see README.md). Upstream kernels are unchanged and still used
on sm>=80 GPUs with flash_attn installed; this module is only reached through the backend
switch in ``attention.py``.

* ``sdpa_attention`` reproduces ``flash_attention`` (flash_attn varlen) semantics: sample b only
  sees ``k[b, :k_lens[b]]`` (padding is sliced away, as varlen packing does). ``q_lens`` is not
  used by the model; here rows >= q_lens[b] return 0 (upstream's wrapper cannot unflatten
  genuinely shortened query totals).
* ``build_mask_plan`` + ``masked_sdpa_attention`` reproduce ``flex_attention(block_mask,
  score_mod)``. The plan evaluates the *same* ``mask_mod`` function that would build the
  flex BlockMask (in row tiles, never the dense Q x KV mask), groups consecutive query rows
  whose mask rows are identical into runs, and keeps, per run, only the column indices it may
  attend to. Each run is then one dense SDPA call over the gathered keys (chunked over queries),
  with the ``score_mod`` turned into an additive bias. Rows with no visible key return 0, as
  in flex_attention.

SDPA calls honour the caller's backend flags. They first try the enabled fused kernels (on V100:
memory-efficient attention, fp16/fp32 only; flash and cuDNN need sm80+). If none accepts the inputs
(e.g. bf16 on sm70) and MATH is enabled, the MATH kernel is used with query chunks sized so its
[B, H, chunk, Lk] score buffers stay within ``WAN_SDPA_MATH_BUDGET_MB`` (default 1024 MiB).

``MONITOR`` (FiniteMonitor) checks the tensors that actually enter / leave every kernel call
(after the dtype cast, after RoPE, after k_lens slicing) for inf/NaN without a host sync; the
fp16 OverflowGuard reads it once per forward (see wanxiang/utils/fp16_guard.py).
"""
import contextlib
import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

__all__ = [
    "sdpa_attention",
    "build_mask_plan",
    "masked_sdpa_attention",
    "MaskPlan",
    "compute_dtype",
    "set_compute_dtype",
    "compute_dtype_override",
    "MONITOR",
]

HALF_DTYPES = (torch.float16, torch.bfloat16)

# dtype used when q/k/v arrive in fp32 (upstream casts those to bf16 for flash/flex).
# The worker sets this to the DiT compute dtype (fp16 on V100); fp32 blocks override it.
_COMPUTE_DTYPE = torch.bfloat16


def compute_dtype():
    return _COMPUTE_DTYPE


def set_compute_dtype(dtype):
    global _COMPUTE_DTYPE
    _COMPUTE_DTYPE = dtype


@contextlib.contextmanager
def compute_dtype_override(dtype):
    prev = _COMPUTE_DTYPE
    set_compute_dtype(dtype)
    try:
        yield
    finally:
        set_compute_dtype(prev)


def _target_dtype(v):
    # Same rule as upstream half(): keep v's dtype when it is already half precision,
    # otherwise use the configured compute dtype. Unlike upstream we cast fp32 q/k straight
    # to the target dtype (no bf16 detour on fp16 devices).
    if _COMPUTE_DTYPE == torch.float32:
        return torch.float32
    return v.dtype if v.dtype in HALF_DTYPES else _COMPUTE_DTYPE


class FiniteMonitor:
    """Device-side inf/NaN flag over every attention kernel input/output (no host sync).

    Disabled unless an OverflowGuard enables it. With ``sink`` set (detailed guard steps), each
    checked tensor also appends ``(label, dtype, max|x|)`` to the guard's record list, labelled
    by the innermost watched module on ``scope`` (pushed/popped by the guard's hooks)."""

    def __init__(self):
        self.enabled = False
        self.sink = None
        self.scope = []   # [[label, calls_in_this_scope], ...]
        self.flags = {}   # device -> bool scalar tensor

    def begin(self, sink=None):
        self.flags = {}
        self.scope = []
        self.sink = sink

    def end(self):
        """Stop labelling (keeps the flags for the guard's check)."""
        self.scope = []
        self.sink = None

    def push(self, label):
        self.scope.append([label, 0])

    def pop(self):
        if self.scope:
            self.scope.pop()

    def take(self, device):
        """Return (and clear) the flag for ``device`` as a float32 0/1 scalar tensor."""
        f = self.flags.pop(torch.device(device), None)
        if f is None:
            return torch.zeros((), dtype=torch.float32, device=device)
        return f.float()

    def check(self, kind, new_call=True, **tensors):
        """new_call=True: first check of a new kernel call (numbers calls per scope);
        new_call=False: further tensors (e.g. the output) of the same call."""
        if not self.enabled:
            return
        if self.sink is not None:
            where, n, outer = "<unscoped>", 0, ""
            if self.scope:
                entry = self.scope[-1]
                if new_call:
                    entry[1] += 1  # attention calls inside this module (cross-attn: img=0, text=1)
                where, n = entry[0], entry[1] - 1
                if len(self.scope) > 1:
                    outer = f" (in {self.scope[0][0]})"
        for name, t in tensors.items():
            if not t.numel():
                continue
            lo, hi = torch.aminmax(t.detach())
            amax = torch.maximum(hi.float(), -lo.float())  # NaN propagates, inf stays inf
            bad = ~torch.isfinite(amax)
            f = self.flags.get(t.device)
            if f is None:
                self.flags[t.device] = bad.clone()
            else:
                f.logical_or_(bad)
            if self.sink is not None:
                self.sink.append((f"{where}:{kind}{n}.{name}{outer}",
                                  str(t.dtype).replace("torch.", ""), amax))


MONITOR = FiniteMonitor()


def _q_chunk():
    return int(os.environ.get("WAN_SDPA_Q_CHUNK", "4096"))


_FUSED_FLAGS = (("FLASH_ATTENTION", "flash_sdp_enabled"), ("EFFICIENT_ATTENTION", "mem_efficient_sdp_enabled"),
                ("CUDNN_ATTENTION", "cudnn_sdp_enabled"))
_NO_KERNEL = ("No available kernel", "No viable backend")
_WARNED = set()


def _allowed_fused():
    """Fused backends still enabled by the caller (respects an outer sdpa_kernel(...) context)."""
    out = []
    for name, flag in _FUSED_FLAGS:
        b, f = getattr(SDPBackend, name, None), getattr(torch.backends.cuda, flag, None)
        if b is not None and (f is None or f()):
            out.append(b)
    return out


def _math_allowed():
    f = getattr(torch.backends.cuda, "math_sdp_enabled", None)
    return f is None or f()


def _sdpa_math_rows(q, k):
    """Query rows per MATH call so that its fp32 scores + softmax (+ half copy) fit the budget."""
    budget = float(os.environ.get("WAN_SDPA_MATH_BUDGET_MB", "1024")) * 2 ** 20
    per_row = q.size(0) * q.size(1) * k.size(2) * 10  # 4 B scores + 4 B softmax + 2 B cast
    return max(1, int(budget // per_row))


def _sdpa_chunked(q, k, v, attn_mask=None, scale=None, dropout_p=0.0):
    """q/k/v: [B, H, L, D]; attn_mask broadcastable to [B, H, Lq, Lk]. Chunked along Lq.

    Honours every backend flag of the caller (an outer sdpa_kernel(...) context): the enabled fused
    kernels are tried first (O(L) memory, chunk = WAN_SDPA_Q_CHUNK rows). Only if none of them
    accepts these inputs AND the MATH kernel is enabled, MATH runs with budget-sized chunks (it
    materialises [B, H, chunk, Lk] fp32 scores); otherwise PyTorch's error is raised. No
    availability is cached: a rejected call costs one dispatch-time exception and does not affect
    later calls with different inputs.
    """
    fused, math_ok = _allowed_fused(), _math_allowed()
    if fused:
        try:
            with sdpa_kernel(fused):
                return _sdpa_rows(q, k, v, attn_mask, scale, dropout_p, _q_chunk())
        except RuntimeError as e:
            if not math_ok or not any(m in str(e) for m in _NO_KERNEL):
                raise
            key = (q.device.type, q.dtype, None if attn_mask is None else attn_mask.dtype, dropout_p > 0)
            if key not in _WARNED:
                _WARNED.add(key)
                import logging
                logging.getLogger("wan.attention").warning(
                    f"[sdpa] no fused SDPA kernel for (device, dtype, mask dtype, dropout)={key}; using "
                    f"the MATH kernel in chunks of {_sdpa_math_rows(q, k)} query rows (WAN_SDPA_MATH_BUDGET_MB)")
    if not math_ok:  # nothing the caller allows: let PyTorch raise its own error
        return _sdpa_rows(q, k, v, attn_mask, scale, dropout_p, _q_chunk())
    with sdpa_kernel([SDPBackend.MATH]):
        return _sdpa_rows(q, k, v, attn_mask, scale, dropout_p, _sdpa_math_rows(q, k))


def _sdpa_rows(q, k, v, attn_mask, scale, dropout_p, chunk):
    lq = q.size(2)
    if lq <= chunk:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale)
    out = q.new_empty(*q.shape[:-1], v.size(-1))
    for s in range(0, lq, chunk):
        e = min(s + chunk, lq)
        m = attn_mask
        if m is not None and m.size(-2) != 1:
            m = m[..., s:e, :]
        out[:, :, s:e] = F.scaled_dot_product_attention(
            q[:, :, s:e], k, v, attn_mask=m, dropout_p=dropout_p, scale=scale)
    return out


def sdpa_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
):
    """Drop-in for ``flash_attention``. q: [B, Lq, N, D], k/v: [B, Lk, N, D] -> [B, Lq, N, Dv]."""
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("sliding-window attention is not used by Wan-Animate-2")
    if causal:
        raise NotImplementedError("causal attention is not used by Wan-Animate-2")

    out_dtype = q.dtype
    b, lq, lk = q.size(0), q.size(1), k.size(1)
    dt = _target_dtype(v)
    q, k, v = q.to(dt), k.to(dt), v.to(dt)
    if q_scale is not None:
        q = q * q_scale

    if k_lens is None:
        groups = [(None, lk)]
    else:
        k_lens = [int(x) for x in torch.as_tensor(k_lens).flatten().tolist()]
        if len(k_lens) != b or min(k_lens) < 1 or max(k_lens) > lk:
            raise ValueError(f"k_lens {k_lens} must have {b} entries in [1, {lk}]")
        # like flash varlen packing: each sample only ever sees k[b, :k_lens[b]] (padding is
        # sliced away, never multiplied by a zero weight); samples grouped by length
        groups = [(None, k_lens[0])] if len(set(k_lens)) == 1 else \
            [([i for i, n in enumerate(k_lens) if n == L], L) for L in sorted(set(k_lens))]

    x = None
    for g, (idx, L) in enumerate(groups):
        qi, ki, vi = (q, k[:, :L], v[:, :L]) if idx is None else (q[idx], k[idx, :L], v[idx, :L])
        MONITOR.check("attn", g == 0, q=qi, k=ki, v=vi)
        o = _sdpa_chunked(qi.transpose(1, 2), ki.transpose(1, 2), vi.transpose(1, 2),
                          scale=softmax_scale, dropout_p=dropout_p).transpose(1, 2)
        if idx is None:
            x = o
        else:
            if x is None:
                x = o.new_empty(b, *o.shape[1:])
            x[idx] = o
    MONITOR.check("attn", False, out=x)

    if q_lens is not None:
        # Not used by the model. Defined extension: rows >= q_lens[b] return 0. (Upstream's
        # wrapper packs shortened queries, then fails in unflatten(0, (b, lq)).)
        q_lens = torch.as_tensor(q_lens, device=x.device).flatten()
        keep = torch.arange(lq, device=x.device)[None, :] < q_lens[:, None]
        x = x * keep[:, :, None, None].to(x.dtype)
    return x.type(out_dtype)


class MaskPlan:
    """SDPA-friendly form of a flex BlockMask: runs of identical query rows -> visible key columns."""

    def __init__(self, q_len, kv_len, runs):
        self.q_len = q_len
        self.kv_len = kv_len
        self.runs = runs  # list of (q_start, q_end, cols: LongTensor); empty cols = fully masked
        self._bias = {}

    def __repr__(self):
        return f"MaskPlan(q_len={self.q_len}, kv_len={self.kv_len}, runs={len(self.runs)})"

    def bias(self, score_mod, dtype, num_heads):
        """Additive bias per run from an additive score_mod (cached per score_mod/dtype)."""
        key = (score_mod, dtype)
        if key not in self._bias:
            self._bias[key] = [self._run_bias(score_mod, q0, q1, cols, dtype, num_heads)
                               for q0, q1, cols in self.runs]
        return self._bias[key]

    @staticmethod
    def _run_bias(score_mod, q0, q1, cols, dtype, num_heads):
        if cols.numel() == 0:
            return None
        dev = cols.device
        zero = torch.zeros((), dtype=torch.long, device=dev)
        kv = cols[None, :]
        first = torch.arange(q0, q0 + 1, device=dev)[:, None]
        s0 = torch.zeros((), dtype=torch.float32, device=dev)
        row = score_mod(s0, zero, zero, first, kv).expand(1, cols.numel()).float()
        # The plan supports additive, batch/head-independent score_mods (the only kind used
        # by the model: score + log_scale on a key range). Verify instead of assuming.
        shifted = score_mod(s0 + 1.0, zero, zero, first, kv).expand_as(row).float()
        last_h = torch.full((), num_heads - 1, dtype=torch.long, device=dev)
        other_h = score_mod(s0, zero, last_h, first, kv).expand_as(row).float()
        if not (torch.allclose(shifted - row, torch.ones_like(row), rtol=0, atol=1e-5)
                and torch.equal(other_h, row)):
            raise NotImplementedError("masked_sdpa_attention needs an additive, head-independent score_mod")
        uniform = True
        tile = max(1, (1 << 22) // cols.numel())
        for r0 in range(q0, q1, tile):
            qi = torch.arange(r0, min(r0 + tile, q1), device=dev)[:, None]
            blk = score_mod(s0, zero, zero, qi, kv).expand(qi.size(0), cols.numel()).float()
            if not torch.equal(blk, row.expand_as(blk)):
                uniform = False
                break
        if uniform:
            bias = row
        else:  # general case: full [rows, cols] bias
            qi = torch.arange(q0, q1, device=dev)[:, None]
            bias = score_mod(s0, zero, zero, qi, kv).expand(q1 - q0, cols.numel()).float()
        if not bias.any():
            return None
        return bias.to(dtype)[None, None]


@torch.no_grad()
def build_mask_plan(mask_mod, q_len, kv_len, device, max_tile_elems=1 << 24):
    """Evaluate a flex ``mask_mod(b, h, q_idx, kv_idx)`` (B=None/H=None semantics) into a MaskPlan.

    Peak extra memory is a few [tile, kv_len] tensors (~16M elements per tile).
    """
    device = torch.device(device)
    kv = torch.arange(kv_len, device=device)
    zero = torch.zeros((), dtype=torch.long, device=device)
    tile = max(1, max_tile_elems // kv_len)
    runs, start, cur = [], 0, None
    for r0 in range(0, q_len, tile):
        r1 = min(r0 + tile, q_len)
        qi = torch.arange(r0, r1, device=device)
        m = mask_mod(zero, zero, qi[:, None], kv[None, :]).expand(r1 - r0, kv_len)
        change = torch.ones(r1 - r0, dtype=torch.bool, device=device)
        change[1:] = (m[1:] != m[:-1]).any(dim=1)
        if cur is not None:
            change[0] = (m[0] != cur).any()
        for i in change.nonzero().flatten().tolist():
            if cur is not None:
                runs.append((start, r0 + i, cur.nonzero().flatten()))
            start, cur = r0 + i, m[i].clone()
    runs.append((start, q_len, cur.nonzero().flatten()))
    return MaskPlan(q_len, kv_len, runs)


def masked_sdpa_attention(q, k, v, plan, score_mod=None):
    """``flex_attention(q, k, v, block_mask, score_mod)`` via SDPA. q/k/v: [B, L, N, D] (model layout)."""
    assert q.size(1) == plan.q_len and k.size(1) == plan.kv_len, (q.shape, k.shape, plan)
    out_dtype = q.dtype
    dt = _target_dtype(v)
    q, k, v = (t.to(dt) for t in (q, k, v))
    MONITOR.check("masked_attn", q=q, k=k, v=v)  # post-RoPE, post-cast kernel inputs
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, N, L, D]
    out = q.new_zeros(*q.shape[:-1], v.size(-1))
    biases = plan.bias(score_mod, dt, q.size(1)) if score_mod is not None else [None] * len(plan.runs)
    for (q0, q1, cols), bias in zip(plan.runs, biases):
        if cols.numel() == 0:
            continue  # fully masked rows -> 0 (flex_attention semantics)
        kc, vc = k.index_select(2, cols), v.index_select(2, cols)
        out[:, :, q0:q1] = _sdpa_chunked(q[:, :, q0:q1], kc, vc, attn_mask=bias)
    MONITOR.check("masked_attn", False, out=out)
    return out.transpose(1, 2).type(out_dtype)
