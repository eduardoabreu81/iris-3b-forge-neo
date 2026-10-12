# Iris 3B support for Forge Neo
# The pixel-space guards: hires and latent-resize blocked, encode/decode methods checked, the Iris (and its depth
# fine-tune) recognized from the selected checkpoint before Forge loads it, and the noise replaced with a
# full-resolution RNG.

import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import headers
import torch

from forge_iris.contracts import IrisError, raise_pending_error, set_pending_error
from forge_iris.script import before_sampling, script_before_process


class FakeModel:
    def __init__(self, is_iris=True):
        self.is_iris = is_iris


class FakeP:
    def __init__(self, is_iris=True, **kwargs):
        self.sd_model = FakeModel(is_iris)
        self.override_settings = {}
        self.enable_hr = False
        self.is_hr_pass = False
        self.resize_mode = 0
        self.height, self.width = 512, 768
        self.seeds, self.subseeds = [1, 2], [0, 0]
        self.subseed_strength = 0.0
        self.seed_resize_from_h = self.seed_resize_from_w = -1
        self.rng = None
        self.modified_noise = None
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeImageRNG:
    """Forge's ImageRNG contract: shape is one sample's, next() stacks one noise per seed."""

    def __init__(self, shape, seeds, **kwargs):
        self.shape = tuple(shape)
        self.seeds = seeds
        self.kwargs = kwargs

    def next(self):
        return torch.stack([torch.zeros(self.shape) for _ in self.seeds])


class Checkpoints:
    """modules.sd_models.get_closet_checkpoint_match over a name -> file map."""

    def __init__(self, files):
        self.files = files

    def get_closet_checkpoint_match(self, name):
        path = self.files.get(name)
        return types.SimpleNamespace(filename=str(path)) if path else None


def stub_modules(checkpoint="iris", files=None, encode="Full", decode="Full"):
    rng_module = types.ModuleType("modules.rng")
    rng_module.ImageRNG = FakeImageRNG
    shared_module = types.ModuleType("modules.shared")
    shared_module.opts = types.SimpleNamespace(sd_vae_encode_method=encode, sd_vae_decode_method=decode,
                                               sd_model_checkpoint=checkpoint)
    sd_models = types.ModuleType("modules.sd_models")
    sd_models.get_closet_checkpoint_match = Checkpoints(files or {}).get_closet_checkpoint_match
    modules = types.ModuleType("modules")
    modules.rng, modules.shared, modules.sd_models = rng_module, shared_module, sd_models
    return {"modules": modules, "modules.rng": rng_module, "modules.shared": shared_module,
            "modules.sd_models": sd_models}


class TestBeforeProcess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        write = headers.write_safetensors
        cls.files = {"iris": write(cls.directory.name, "iris-3b_bf16"),
                     "iris int8": write(cls.directory.name, "iris-3b_int8_convrot"),
                     "iris depth": write(cls.directory.name, "iris-3b-depth_bf16"),
                     "iris gguf": headers.write_gguf(cls.directory.name, "iris-3b-q4_k_s"),
                     "aikimi": write(cls.directory.name, "aikimi_iris-3b-int8"),
                     "qwen": write(cls.directory.name, "qwen3vl_4b_bf16")}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def tearDown(self):
        set_pending_error(None)

    def run_guard(self, p, **stubs):
        with patch.dict(sys.modules, stub_modules(files=self.files, **stubs)):
            script_before_process(p)

    def assert_refused(self):
        with self.assertRaises(IrisError):
            raise_pending_error()

    def test_selected_checkpoint_decides_not_the_loaded_model(self):
        # Forge loads the checkpoint after before_process: switching to Iris still runs the guards...
        self.run_guard(FakeP(is_iris=False, enable_hr=True), checkpoint="iris")
        self.assert_refused()
        set_pending_error(None)
        # ...and switching away from Iris does not refuse the other model's hires fix
        p = FakeP(is_iris=True, enable_hr=True, init_images=[], resize_mode=3)
        self.run_guard(p, checkpoint="qwen")
        raise_pending_error()
        self.assertEqual(p.resize_mode, 3)

    def test_int8_checkpoint_is_recognized(self):
        self.run_guard(FakeP(is_iris=False, enable_hr=True), checkpoint="iris int8")
        self.assert_refused()

    def test_gguf_checkpoint_is_recognized(self):
        self.run_guard(FakeP(is_iris=False, enable_hr=True), checkpoint="iris gguf")
        self.assert_refused()

    def test_depth_model_is_refused(self):
        self.run_guard(FakeP(is_iris=False), checkpoint="iris depth")
        with self.assertRaises(IrisError) as raised:
            raise_pending_error()
        self.assertIn("depth", str(raised.exception))

    def test_foreign_build_is_left_to_the_loader(self):
        # it never loads (the detection hook explains why), so nothing may stay pending
        self.run_guard(FakeP(is_iris=False, enable_hr=True), checkpoint="aikimi")
        raise_pending_error()

    def test_override_checkpoint_wins(self):
        self.run_guard(FakeP(enable_hr=True, override_settings={"sd_model_checkpoint": "qwen"}), checkpoint="iris")
        raise_pending_error()

    def test_hires_is_refused(self):
        self.run_guard(FakeP(enable_hr=True))
        self.assert_refused()

    def test_refusal_does_not_outlive_its_generation(self):
        self.run_guard(FakeP(enable_hr=True))
        self.assert_refused()
        self.run_guard(FakeP())
        raise_pending_error()

    def test_decode_method_is_checked(self):
        self.run_guard(FakeP(), decode="TAESD")
        self.assert_refused()

    def test_encode_method_is_checked(self):
        self.run_guard(FakeP(init_images=[]), encode="TAESD")
        self.assert_refused()

    def test_latent_resize_mode_falls_back(self):
        p = FakeP(init_images=[], resize_mode=3)
        self.run_guard(p)
        self.assertEqual(p.resize_mode, 0)
        raise_pending_error()  # no error: nothing pending


class TestBeforeSampling(unittest.TestCase):
    def sample(self, p, noise):
        with patch.dict(sys.modules, stub_modules()):
            before_sampling(p, noise)

    def test_noise_becomes_full_resolution(self):
        # Forge's 1/8-resolution latent noise, for txt2img and img2img alike
        p = FakeP()
        self.sample(p, torch.zeros(2, 3, 64, 96))
        self.assertEqual(tuple(p.modified_noise.shape), (2, 3, 512, 768))
        self.assertEqual(p.rng.shape, (3, 512, 768))

    def test_full_resolution_noise_untouched(self):
        p = FakeP()
        self.sample(p, torch.zeros(2, 3, 512, 768))
        self.assertIsNone(p.modified_noise)

    def test_hires_pass_untouched(self):
        p = FakeP(is_hr_pass=True)
        self.sample(p, torch.zeros(2, 3, 1024, 1536))
        self.assertIsNone(p.modified_noise)

    def test_seed_resize_is_full_resolution(self):
        p = FakeP(seed_resize_from_h=256, seed_resize_from_w=384)
        self.sample(p, torch.zeros(2, 3, 64, 96))
        # ImageRNG divides by 8 for an image latent
        self.assertEqual((p.rng.kwargs["seed_resize_from_h"], p.rng.kwargs["seed_resize_from_w"]), (2048, 3072))

    def test_non_iris_untouched(self):
        p = FakeP(is_iris=False)
        self.sample(p, torch.zeros(1, 4, 64, 96))
        self.assertIsNone(p.modified_noise)


if __name__ == "__main__":
    unittest.main()
