"""Dependency-free smoke tests for the opt-in 8x V100 16GB preset dispatcher."""
import os
import pathlib
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v100_16gb.sh"


def invoke(**overrides):
    env = dict(os.environ)
    for key in ("PRESET", "MODE", "RES", "CLIP_LEN", "DTYPE", "GPUS",
                "ALLOW_UNVERIFIED_HIGH_RES", "EXTRA_OVERRIDES"):
        env.pop(key, None)
    env.update({"DRY_RUN": "1", **overrides})
    return subprocess.run(["bash", str(SCRIPT)], cwd=str(ROOT),
                          env=env, text=True, capture_output=True, check=False)


class Presets(unittest.TestCase):
    def test_three_regular_presets_resolve(self):
        cases = [
            ("minimal", "RES=256x448 CLIP_LEN=17"),
            ("balanced", "RES=256x448 CLIP_LEN=25"),
            ("detail", "RES=288x512 CLIP_LEN=25"),
        ]
        for preset, expected in cases:
            with self.subTest(preset=preset):
                p = invoke(PRESET=preset)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertIn(expected, p.stdout)
                self.assertIn("GPUS=0,1,2,3,4,5,6,7", p.stdout)
                self.assertIn("MODE=distill", p.stdout)
                self.assertIn("DTYPE=fp16", p.stdout)
                for item in ("test_cfg.dit_shard_dtype=bf16", "test_cfg.low_vram=true",
                             "test_cfg.log_gpu_memory=true", "test_cfg.overflow_action=abort",
                             "test_cfg.fp32_modules=[time_embedding,time_projection,head]"):
                    self.assertIn(item, p.stdout)

    def test_base_model_uses_same_safety_defaults(self):
        p = invoke(PRESET="minimal", MODE="base")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("MODE=base", p.stdout)
        self.assertIn("test_cfg.overflow_action=abort", p.stdout)

    def test_high_res_is_explicitly_gated(self):
        p = invoke(PRESET="experimental480")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("HIGH OOM risk", p.stderr)
        p = invoke(PRESET="experimental480", ALLOW_UNVERIFIED_HIGH_RES="1")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("RES=480p CLIP_LEN=17", p.stdout)
        p = invoke(RES="540p")
        self.assertNotEqual(p.returncode, 0)

    def test_bad_hardware_and_precision_are_refused(self):
        cases = [
            ({"GPUS": "0"}, "eight"),
            ({"GPUS": "0,1,2,3,4,5,6,6"}, "Duplicate GPU"),
            ({"DTYPE": "bf16"}, "guarded fp16"),
            ({"DTYPE": "fp32"}, "guarded fp16"),
            ({"CLIP_LEN": "20"}, "4n+1"),
            ({"MODE": "invalid"}, "MODE must"),
            ({"PRESET": "invalid"}, "PRESET must"),
        ]
        for options, error in cases:
            with self.subTest(options=options):
                p = invoke(**options)
                self.assertNotEqual(p.returncode, 0)
                self.assertIn(error, p.stderr)


if __name__ == "__main__":
    unittest.main()
