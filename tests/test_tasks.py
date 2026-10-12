# Iris 3B support for Forge Neo
# The single-pass tasks against speridlabs/iris-3b's own code (a8d15239, src/iris3b/downstream/restoration.py and
# depth.py, transcribed here), the script's preparation of Forge's img2img, and the DiT's task modes.

import math
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image

from forge_iris import contracts, tasks


def upstream_window(tile):
    index = torch.arange(tile, dtype=torch.float32)
    cols = torch.exp(-((index - (tile - 1) / 2) ** 2) / (tile * tile) / 0.02)
    rows = torch.exp(-((index - tile / 2) ** 2) / (tile * tile) / 0.02)
    return (rows[:, None] * cols[None, :])[None, None]


def upstream_tiled(fn, x, tile, stride):
    def positions(size):
        return [0] if size <= tile else [*range(0, size - tile, stride), size - tile]

    window, out, weight = upstream_window(tile), torch.zeros_like(x), torch.zeros_like(x[:, :1])
    for top in positions(x.shape[-2]):
        for left in positions(x.shape[-1]):
            crop = (..., slice(top, top + tile), slice(left, left + tile))
            out[crop] += fn(x[crop].contiguous()) * window
            weight[crop] += window
    return out / weight


class TestRestoreMath(unittest.TestCase):
    def test_sizes(self):
        # a 256 input is exactly one tile; 384x256 works at 1536x1024; 512 needs no enlargement
        self.assertEqual(tasks.restore_sizes(256, 256), ((1024, 1024), (1024, 1024)))
        self.assertEqual(tasks.restore_sizes(384, 256), ((1024, 1536), (1024, 1536)))
        self.assertEqual(tasks.restore_sizes(200, 100), ((400, 800), (1024, 2048)))
        self.assertEqual(tasks.restore_sizes(512, 512), ((2048, 2048), (2048, 2048)))

    def test_tiling_matches_upstream(self):
        torch.manual_seed(0)
        x = torch.rand(1, 3, 70, 100)

        def fn(t):
            return t * 2 + t.mean()

        torch.testing.assert_close(tasks.tiled(fn, x, 32, 16), upstream_tiled(fn, x, 32, 16))
        torch.testing.assert_close(tasks.tiled(lambda t: t, x, 32, 16), x)

    def test_tiling_in_float16_stays_finite(self):
        # Forge computes in float16 on GPUs without bfloat16; the window's ~1e-11 corners must not reach 0 / 0
        x = (torch.rand(1, 3, 1024, 1536) * 2 - 1).half()
        out = tasks.tiled(lambda t: t, x)
        self.assertEqual(out.dtype, torch.float16)
        self.assertTrue(bool(torch.isfinite(out).all()))
        torch.testing.assert_close(out, x)

    def test_color_fix_is_detail_plus_colour(self):
        torch.manual_seed(0)
        content, reference = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
        high, low = tasks._split(content)
        torch.testing.assert_close(high + low, content)
        torch.testing.assert_close(tasks.wavelet_color_fix(content, reference), high + tasks._split(reference)[1])
        torch.testing.assert_close(tasks.wavelet_color_fix(content, content), content)

    def test_inputs_and_output(self):
        torch.manual_seed(0)
        image = Image.fromarray((torch.rand(64, 96, 3) * 255).byte().numpy())
        reference, work = tasks.restore_inputs(image)
        self.assertEqual(tuple(reference.shape), (1, 3, 256, 384))
        self.assertEqual(tuple(work.shape[-2:]), (1024, 1536))
        out = tasks.restore_output(work.clamp(0, 1), reference)
        self.assertEqual(out.shape, reference.shape)
        # back from the work size the bicubic round trip stays close, and the colour fix pulls it further in
        self.assertLess(float((out - reference).abs().mean()), 0.05)


class TestDepthMath(unittest.TestCase):
    def test_size(self):
        self.assertEqual(tasks.depth_size(640, 480), (640, 480))
        self.assertEqual(tasks.depth_size(4000, 3000), (1024, 768))
        self.assertEqual(tasks.depth_size(1000, 700), (992, 704))

    def test_map_near_is_bright(self):
        # relative log depth: -1 near (top), +1 far (bottom), carried in [0, 1] image space
        ramp = torch.linspace(0, 1, 64)[None, None, :, None].expand(1, 3, 64, 32)
        out = tasks.depth_map(ramp, (128, 64))
        self.assertEqual(tuple(out.shape), (1, 3, 128, 64))
        self.assertGreater(float(out[0, 0, 5].mean()), 0.9)
        self.assertLess(float(out[0, 0, -5].mean()), 0.1)


class TestDiTTaskModes(unittest.TestCase):
    def setUp(self):
        from test_transformer import toy_config

        from forge_iris.native.transformer import IrisDiT

        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.dit = IrisDiT(toy_config()).eval()

    def tearDown(self):
        contracts.set_task(None)

    def test_depth_returns_x_minus_output(self):
        # Forge's denoised = x - sigma * out at sigma 1 must be the model's output itself
        x, t, context = torch.rand(1, 3, 32, 32) * 2 - 1, torch.tensor([1000.0]), torch.rand(1, 300, 32)
        with torch.no_grad():
            plain = self.dit(x, t, context)
            contracts.set_task(tasks.DEPTH)
            depth = self.dit(x, t, context)
        torch.testing.assert_close(x - depth, plain)

    def test_restore_tiles(self):
        x, t, context = torch.rand(1, 3, 40, 56) * 2 - 1, torch.tensor([500.0]), torch.rand(1, 300, 32)
        with torch.no_grad(), patch.object(tasks, "TILE", 32):
            contracts.set_task(tasks.RESTORE)
            out = self.dit(x, t, context)
            contracts.set_task(None)
            padded = F.pad(x, (0, -56 % 16, 0, -40 % 16))
            expected = upstream_tiled(lambda tile: self.dit(tile, t, context), padded, 32, 16)[..., :40, :56]
        torch.testing.assert_close(out, expected)


class TestPrepare(unittest.TestCase):
    def tearDown(self):
        contracts.set_task(None)

    def test_restore_prepares_forge_img2img(self):
        from forge_iris import script

        image = Image.new("RGB", (200, 100), (120, 60, 30))
        p = types.SimpleNamespace(init_images=[image], image_mask=None,
                                  prompt="sharp photo <lora:iris-3b-upscaler_r128:1>", negative_prompt="blurry",
                                  override_settings={}, extra_generation_params={})
        script._prepare_task(p, tasks.RESTORE, script.MODE_RESTORE)
        self.assertEqual((p.width, p.height), (2048, 1024))
        # the fine-tune's empty prompt, with the LoRA tag kept for Forge to apply
        self.assertEqual(p.prompt, "<lora:iris-3b-upscaler_r128:1>")
        self.assertEqual((p.negative_prompt, p.cfg_scale, p.steps, p.denoising_strength), ("", 1.0, 1, 1.0))
        self.assertEqual(p.sampler_noise_scheduler_override(1).tolist(), [0.5, 0.0])
        self.assertEqual(p.override_settings["img2img_extra_noise"], 0)
        self.assertEqual(tuple(p.iris_task["reference"].shape), (1, 3, 400, 800))
        self.assertEqual(contracts.current_task(), tasks.RESTORE)

    def test_depth_prepares_forge_img2img(self):
        from forge_iris import script

        p = types.SimpleNamespace(init_images=[Image.new("RGB", (1000, 700))], image_mask=None, prompt="",
                                  negative_prompt="", override_settings={}, extra_generation_params={})
        script._prepare_task(p, tasks.DEPTH, script.MODE_DEPTH)
        self.assertEqual((p.width, p.height), (992, 704))
        # no tags: a space, which sd-dynamic-prompts accepts and the text engine encodes as the empty prompt
        self.assertEqual(p.prompt, " ")
        self.assertEqual(p.sampler_noise_scheduler_override(1).tolist(), [1.0, 0.0])
        self.assertEqual(p.iris_task["size"], (700, 1000))

    def test_noise_is_the_image(self):
        from forge_iris import script

        config_type = type("Config", (), {"memory_usage_factor": 0.05})
        config = config_type()
        p = types.SimpleNamespace(sd_model=types.SimpleNamespace(is_iris=True, model_config=config),
                                  is_hr_pass=False, iris_task={"task": tasks.RESTORE}, modified_noise=None)
        x = torch.rand(1, 3, 2048, 1024)
        script.before_sampling(p, torch.zeros(1, 3, 256, 128), x)
        torch.testing.assert_close(p.modified_noise, x)
        # sigma * noise + (1 - sigma) * image with the image as noise: the image
        torch.testing.assert_close(0.5 * p.modified_noise + 0.5 * x, x)
        # the activation reserve covers one tile, not the image's two tiles' worth
        self.assertAlmostEqual(config.memory_usage_factor, 0.05 * 0.5)
        script._reset_memory_factor(p.sd_model)
        self.assertTrue(math.isclose(config.memory_usage_factor, 0.05))


if __name__ == "__main__":
    unittest.main()
