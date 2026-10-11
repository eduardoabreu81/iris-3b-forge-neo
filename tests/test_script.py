# Iris 3B support for Forge Neo
# The pixel-space guards: hires and latent-resize blocked, encode method checked, and the txt2img noise
# replaced with a full-resolution RNG.

import sys
import unittest
from unittest.mock import patch

from forge_iris.contracts import IrisError, raise_pending_error, set_pending_error
from forge_iris.script import before_sampling, script_before_process


class FakeModel:
    def __init__(self, is_iris=True):
        self.is_iris = is_iris


class FakeP:
    def __init__(self, is_iris=True, **kwargs):
        self.sd_model = FakeModel(is_iris)
        self.enable_hr = False
        self.is_img2img = False
        self.resize_mode = 0
        self.height, self.width = 512, 768
        self.seeds, self.subseeds = [1], [0]
        self.subseed_strength = 0.0
        self.seed_resize_from_h = self.seed_resize_from_w = -1
        self.rng = None
        self.modified_noise = None
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeImageRNG:
    shapes = []

    def __init__(self, shape, seeds, **kwargs):
        self.shape = shape

    def next(self):
        import torch

        return torch.zeros(self.shape)


def stub_modules():
    import types

    rng_module = types.ModuleType("modules.rng")
    rng_module.ImageRNG = FakeImageRNG
    shared_module = types.ModuleType("modules.shared")
    shared_module.opts = types.SimpleNamespace(sd_vae_encode_method="Full")
    modules = types.ModuleType("modules")
    modules.rng = rng_module
    modules.shared = shared_module
    return {
        "modules": modules,
        "modules.rng": rng_module,
        "modules.shared": shared_module,
    }


class TestBeforeProcess(unittest.TestCase):
    def tearDown(self):
        set_pending_error(None)

    def test_non_iris_untouched(self):
        p = FakeP(is_iris=False, enable_hr=True, resize_mode=3)
        with patch.dict(sys.modules, stub_modules()):
            script_before_process(p)
        self.assertIsNone(p.rng)
        self.assertEqual(p.resize_mode, 3)

    def test_hires_is_refused(self):
        p = FakeP(enable_hr=True)
        with patch.dict(sys.modules, stub_modules()):
            script_before_process(p)
            with self.assertRaises(IrisError):
                raise_pending_error()

    def test_encode_method_is_checked(self):
        p = FakeP(is_img2img=True)
        stubs = stub_modules()
        stubs["modules.shared"].opts.sd_vae_encode_method = "TAESD"
        with patch.dict(sys.modules, stubs):
            script_before_process(p)
            with self.assertRaises(IrisError):
                raise_pending_error()

    def test_latent_resize_mode_falls_back(self):
        p = FakeP(is_img2img=True, resize_mode=3)
        with patch.dict(sys.modules, stub_modules()):
            script_before_process(p)
        self.assertEqual(p.resize_mode, 0)
        self.assertIsNone(p.rng)  # no error: nothing pending


class TestBeforeSampling(unittest.TestCase):
    def test_txt2img_noise_becomes_full_resolution(self):
        import torch

        p = FakeP()
        noise = torch.zeros(1, 3, 64, 96)  # Forge's 1/8-resolution latent noise
        with patch.dict(sys.modules, stub_modules()):
            before_sampling(p, noise)
        self.assertEqual(tuple(p.modified_noise.shape), (1, 3, 512, 768))
        self.assertEqual(p.rng.shape, (1, 3, 512, 768))

    def test_img2img_noise_untouched(self):
        import torch

        p = FakeP(is_img2img=True)
        noise = torch.zeros(1, 3, 512, 768)  # the init latent sets the right shape already
        with patch.dict(sys.modules, stub_modules()):
            before_sampling(p, noise)
        self.assertIsNone(p.modified_noise)

    def test_non_iris_untouched(self):
        import torch

        p = FakeP(is_iris=False)
        noise = torch.zeros(1, 3, 64, 96)
        with patch.dict(sys.modules, stub_modules()):
            before_sampling(p, noise)
        self.assertIsNone(p.modified_noise)


if __name__ == "__main__":
    unittest.main()
