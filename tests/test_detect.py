# Iris 3B support for Forge Neo
# detect() against the real published checkpoints (headers only, no weights) and against a non-Iris state dict.

import gzip
import json
import unittest
from pathlib import Path

from forge_iris.native import detect

FIXTURES = Path(__file__).parent / "fixtures"
CHECKPOINTS = ["iris-3b_bf16", "iris-3b_int8_convrot"]

EXPECTED = {
    "hidden_size": 2560, "patch_size": 16, "depth": 24, "dual_depth": 8,
    "num_heads": 20, "num_kv_heads": 5, "text_len": 300, "text_dim": 2560,
    "text_lap_num_layers": 12, "text_lap_ffn_hidden_size": 3328,
    "pixel_hidden_size": 16, "pixel_depth": 4, "pixel_attn_hidden_size": 1280,
    "pixel_num_heads": 10, "pixel_ffn_hidden_size": 64,
}


def header(name):
    return json.load(gzip.open(FIXTURES / f"{name}_header.json.gz", "rt"))


class TestDetect(unittest.TestCase):
    def test_real_checkpoints(self):
        for name in CHECKPOINTS:
            with self.subTest(checkpoint=name):
                config = detect.detect(header(name), "")
                self.assertEqual(config["image_model"], "iris")
                for key, value in EXPECTED.items():
                    self.assertEqual(config[key], value, f"{name}: {key}")

    def test_int8_reads_its_padded_ffn(self):
        # the INT8 checkpoint zero-pads the SwiGLU FFN to a rotation-group multiple; Forge must build with
        # exactly the width the checkpoint carries, so the two checkpoints detect different FFN sizes
        self.assertEqual(detect.detect(header("iris-3b_bf16"), "")["ffn_hidden_size"], 6826)
        self.assertEqual(detect.detect(header("iris-3b_int8_convrot"), "")["ffn_hidden_size"], 6912)

    def test_key_prefix(self):
        config = detect.detect({f"model.diffusion_model.{k}": v for k, v in header("iris-3b_bf16").items()
                                if k != "__metadata__"}, "model.diffusion_model.")
        self.assertEqual(config["depth"], 24)

    def test_rejects_other_models(self):
        self.assertIsNone(detect.detect({"input_blocks.0.weight": {"shape": [320, 4, 3, 3]}}, ""))
        self.assertIsNone(detect.detect({}, ""))


if __name__ == "__main__":
    unittest.main()
