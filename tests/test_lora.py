# Iris 3B support for Forge Neo
# Iris LoRAs fitted for Forge's matcher, from the published files' real headers: the Turbo LoRA's bare keys and
# metadata alpha, and FFN widths fitted both ways (6826 <-> 6912), checked layer by layer against the checkpoints.

import unittest

import headers
import torch

from forge_iris.native import lora


def tensors(fixture, rank=1):
    """The file's keys with tensors of its shapes, the rank axis cut to `rank` (the real ones are 64-128)."""
    header = headers.header(fixture)
    metadata = header.pop("__metadata__", {})
    out = {}
    for key, entry in header.items():
        shape = list(entry["shape"])
        if key.endswith(("lora_down", "lora_down.weight")):
            shape[0] = rank
        elif key.endswith(("lora_up", "lora_up.weight")):
            shape[1] = rank
        out[key] = torch.ones(shape)
    return out, metadata


def model_shapes(fixture):
    return {f"diffusion_model.{k}": tuple(v["shape"]) for k, v in headers.header(fixture).items() if k != "__metadata__"}


class TestConvert(unittest.TestCase):
    def check_fits(self, converted, checkpoint):
        model = model_shapes(checkpoint)
        layers = {k[: -len(".lora_down.weight")] for k in converted if k.endswith(".lora_down.weight")}
        self.assertTrue(layers)
        for layer in layers:
            weight = model.get(f"{layer}.weight")
            self.assertIsNotNone(weight, layer)
            down, up = converted[f"{layer}.lora_down.weight"], converted[f"{layer}.lora_up.weight"]
            self.assertEqual((up.shape[0], down.shape[1]), weight, layer)

    def test_turbo_keys_and_alpha(self):
        lora_sd, metadata = tensors("iris-3b-turbo-lora_r2-1000")
        self.assertTrue(lora.is_bare_format(lora_sd))
        alpha_rank = (float(metadata["alpha"]), int(metadata["rank"]))
        converted = lora.convert(lora_sd, 6826, alpha_rank)
        self.assertFalse(lora.is_bare_format(converted))
        # Forge scales by alpha / rows of down: the file's alpha / rank (64 / 64) must survive the cut rank
        alpha = converted["diffusion_model.blocks.0.attn.q_proj_x.alpha"].item()
        self.assertAlmostEqual(alpha / converted["diffusion_model.blocks.0.attn.q_proj_x.lora_down.weight"].shape[0],
                               1.0)

    def test_turbo_fits_both_builds(self):
        lora_sd, metadata = tensors("iris-3b-turbo-lora_r2-1000")
        alpha_rank = (float(metadata["alpha"]), int(metadata["rank"]))
        self.check_fits(lora.convert(lora_sd, 6826, alpha_rank), "iris-3b_bf16")
        self.check_fits(lora.convert(lora_sd, 6912, alpha_rank), "iris-3b_int8_convrot")

    def test_upscaler_fits_both_builds(self):
        # extracted at the padded 6912: cropped for bf16, as is for INT8
        lora_sd, _ = tensors("iris-3b-upscaler_r128")
        self.check_fits(lora.convert(lora_sd, 6826), "iris-3b_bf16")
        self.check_fits(lora.convert(lora_sd, 6912), "iris-3b_int8_convrot")

    def test_padding_is_exact(self):
        # a LoRA fitted 6826 -> 6912 -> 6826 comes back unchanged, and the padding is zero
        up = torch.randn(6826, 4)
        padded = lora.convert({"diffusion_model.blocks.0.mlp_x.w1.lora_up.weight": up}, 6912)
        value = padded["diffusion_model.blocks.0.mlp_x.w1.lora_up.weight"]
        self.assertTrue(bool((value[6826:] == 0).all()))
        back = lora.convert(padded, 6826)["diffusion_model.blocks.0.mlp_x.w1.lora_up.weight"]
        self.assertTrue(torch.equal(back, up))


if __name__ == "__main__":
    unittest.main()
