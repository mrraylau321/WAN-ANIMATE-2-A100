"""Two-A100 native BF16 allocation regression tests (CPU/gloo, no GPU needed)."""
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from conftest import REPO
from test_v100_state_dict import _run, _save_bf16_ckpt


@pytest.mark.parametrize("name", [
    "wan_animate_2_a100.yaml",
    "wan_animate_2_distillation_a100.yaml",
])
def test_native_a100_config_is_two_gpu_bf16(name):
    cfg = OmegaConf.load(str(Path(REPO) / "infer" / name))
    t = cfg.test_cfg
    assert t.sp_size == t.sharding_size == 2
    assert t.sharding_strategy == "FULL_SHARD"
    assert t.param_dtype == "bfloat16" and t.reduce_dtype == "bfloat16"
    assert t.dit_dtype == "bf16" and t.dit_shard_dtype == "bf16"
    assert t.t5_dtype == "bf16" and t.vae_dtype == "bf16"
    assert t.clip_dtype == "fp16" and t.clip_text_on_gpu is False
    assert t.low_vram and t.log_gpu_memory
    assert t.kv_cache_owned
    assert list(t.fp32_modules) == ["time_embedding", "time_projection", "head"]
    assert cfg.model.transformer.num_heads % t.sp_size == 0
    assert cfg.model.transformer.log_scale == (
        -1.3 if "distillation" in name else 0.0
    )


def test_native_a100_two_rank_bf16_shards_forward(tmp_path):
    """Two FSDP2 ranks use BF16 persistent shards + BF16 compute, FP32 root islands."""
    ckpt = str(tmp_path / "tiny_bf16.safetensors")
    original = _save_bf16_ckpt(ckpt)
    result = _run(tmp_path, "a100_bf16_forward", ckpt, world=2)
    assert all(r["ok"] for r in result), [r.get("tb") for r in result]
    for r in result:
        assert r["prec"].startswith("dit=bf16 dit_output=bf16")
        assert r["storage_dtype"] == "torch.bfloat16"
        assert r["local_allocation"].get("bfloat16", 0) > 0
        assert r["local_allocation"].get("float32", 0) > 0
        assert set(r["fp32_units"]) == {"time_embedding", "time_projection", "head"}
    full = torch.load(tmp_path / "a100_bf16_forward_sd.pt", weights_only=True)
    assert set(full) == set(original)
    for name, value in original.items():
        is_fp32 = name.startswith(("time_embedding.", "time_projection.", "head."))
        assert full[name].dtype == (torch.float32 if is_fp32 else torch.bfloat16)
        assert torch.equal(full[name], value.float() if is_fp32 else value), name
    a, b = (
        torch.load(tmp_path / f"a100_bf16_forward_out_rank{i}.pt", weights_only=True)
        for i in (0, 1)
    )
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_affine_layernorm_accepts_fp32_residuals(dtype):
    from wanxiang.models.wan_animate_2_model import LayerNorm
    norm = LayerNorm(16, elementwise_affine=True).to(dtype=dtype).eval()
    x = torch.randn(3, 16, dtype=torch.float32)
    with torch.no_grad():
        y = norm(x)
        ref = torch.nn.functional.layer_norm(
            x, (16,), norm.weight.float(), norm.bias.float(), norm.eps
        )
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, ref, rtol=1e-6, atol=1e-6)


def test_a100_flash_kv_cache_owns_storage(monkeypatch):
    """A100 must not retain the all_to_all Q buffer via K/V views on FlashAttention."""
    from conftest import TINY, make_forward_inputs
    from wanxiang.models import wan_animate_2_model as M
    from wanxiang.models import attention as A
    from wanxiang.models.wan_animate_2_model import WanAnimate2Transformer

    def flash_ref(q, k, v, k_lens=None, **_):
        length = k.shape[1] if k_lens is None else int(k_lens[0])
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k[:, :length].transpose(1, 2),
            v[:, :length].transpose(1, 2)).transpose(1, 2)

    monkeypatch.setattr(M, "attention_backend", lambda device: "flash")
    monkeypatch.setattr(M, "flash_attention", flash_ref)
    A.set_compute_dtype(torch.float32)
    M.set_own_kv_cache(True)
    try:
        model = WanAnimate2Transformer(**TINY).eval()
        inp = make_forward_inputs()
        keys, values = {}, {}
        with torch.no_grad():
            model(inp["cond_lat"], grid_sizes=inp["grid_sizes"],
                  k_cache=keys, v_cache=values, t=inp["t"], method="forward_ref",
                  context_ref=inp["ctx_ref"], seq_len_ref=inp["seq_len_ref"],
                  clip_fea_ref=inp["clip_fea_ref"], y_ref=inp["cond_y"])
        assert keys and values
        for tensor in list(keys.values()) + list(values.values()):
            assert tensor._base is None
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    finally:
        M.set_own_kv_cache(False)
        A.set_compute_dtype(torch.bfloat16)
