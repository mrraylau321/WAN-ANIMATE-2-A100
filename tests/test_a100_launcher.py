"""2×A100-80GB launcher/preflight contract tests (no CUDA hardware required)."""
import os
import pathlib
import subprocess
from types import SimpleNamespace

import pytest

from scripts.probe_a100 import inspect_devices

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_a100.sh"


class FakeCuda:
    def __init__(self, specs):
        self.specs = specs

    def is_available(self):
        return True

    def device_count(self):
        return len(self.specs)

    def get_device_properties(self, i):
        name, mib, cc = self.specs[i]
        return SimpleNamespace(name=name, total_memory=mib * 1024 * 1024,
                               major=cc[0], minor=cc[1])

    def is_bf16_supported(self, including_emulation=True):
        return True


def run_launcher(**overrides):
    env = os.environ.copy()
    for key in ("PRESET", "MODE", "RES", "CLIP_LEN", "GPUS", "DTYPE",
                "ATTN_BACKEND", "EXTRA_OVERRIDES", "DRY_RUN"):
        env.pop(key, None)
    env.update({"DRY_RUN": "1", **overrides})
    return subprocess.run(["bash", str(LAUNCHER)], cwd=ROOT, env=env,
                          capture_output=True, text=True, check=False)


def test_two_a100_80gb_preflight_accepts_reported_81920_mib():
    cuda = FakeCuda([("NVIDIA A100-SXM4-80GB", 81920, (8, 0)),
                     ("NVIDIA A100-SXM4-80GB", 81920, (8, 0))])
    result = inspect_devices(SimpleNamespace(cuda=cuda))
    assert len(result) == 2 and all(x[2] == 81920 for x in result)


@pytest.mark.parametrize("specs,expected", [
    ([("A100", 81920, (8, 0))], "exactly 2"),
    ([("A100", 40960, (8, 0))] * 2, "80GB"),
    ([("V100", 81920, (7, 0))] * 2, "expected A100"),
    ([("H100", 81920, (9, 0))] * 2, "expected A100"),
])
def test_a100_preflight_rejects_other_hardware(specs, expected):
    with pytest.raises(RuntimeError, match=expected):
        inspect_devices(SimpleNamespace(cuda=FakeCuda(specs)))


@pytest.mark.parametrize("preset,res,length", [
    ("smoke", "480p", 33),
    ("standard", "720p", 65),
    ("full", "720p", 81),
])
def test_launcher_presets(preset, res, length):
    result = run_launcher(PRESET=preset)
    assert result.returncode == 0, result.stderr
    assert f"RES={res}" in result.stdout
    assert f"CLIP_LEN={length}" in result.stdout
    assert "GPUS=0,1" in result.stdout and "DTYPE=bf16" in result.stdout
    assert "sp_size=2" in result.stdout and "sharding_size=2" in result.stdout
    assert "wan_animate_2_distillation_a100.yaml" in result.stdout


def test_launcher_base_mode_and_smoke():
    result = run_launcher(PRESET="smoke", MODE="base")
    assert result.returncode == 0, result.stderr
    assert "wan_animate_2_a100.yaml" in result.stdout
    assert "STEPS=40" in result.stdout and "GUIDE=3.0" in result.stdout


@pytest.mark.parametrize("overrides,needle", [
    ({"GPUS": "0"}, "two GPU indices"),
    ({"GPUS": "0,0"}, "Duplicate"),
    ({"DTYPE": "fp16"}, "native BF16"),
    ({"MODE": "invalid"}, "MODE must"),
    ({"PRESET": "invalid"}, "PRESET must"),
    ({"CLIP_LEN": "64"}, "4n+1"),
    ({"EXTRA_OVERRIDES": "test_cfg.sp_size=8"}, "EXTRA_OVERRIDES disabled"),
])
def test_launcher_rejects_invalid_hardware_or_unsafe_overrides(overrides, needle):
    result = run_launcher(**overrides)
    assert result.returncode != 0
    assert needle in result.stderr
