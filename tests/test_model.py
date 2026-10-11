# Iris 3B support for Forge Neo
# The guess class: model identity, flow sampling settings, RGB latent, and the same text encoder bucket the
# Forge loads natively for Krea 2.

import sys
import unittest

try:
    import torch
except ImportError:
    torch = None

from unittest.mock import patch

import forge_stubs


def load_model():
    with patch.dict(sys.modules, forge_stubs.engine_modules()):
        import importlib

        return importlib.import_module("forge_iris.native.model")


@unittest.skipIf(torch is None, "torch not installed")
class TestGuess(unittest.TestCase):
    def setUp(self):
        self.model = load_model()

    def test_identity_and_recognition(self):
        self.assertEqual(self.model.Iris.unet_config, {"image_model": "iris"})

    def test_sampling_settings(self):
        # discrete flow: the DiT embeds timestep = sigma * 1000; shift 4 belongs to the Iris scheduler
        self.assertEqual(self.model.Iris.sampling_settings, {"shift": 4.0, "multiplier": 1000})
        guess = self.model.Iris.__new__(self.model.Iris)
        self.assertEqual(guess.model_type({}), "flow")

    def test_pixel_space_latent(self):
        # the latent format is the guess package's RGB (the pixels themselves); the stubs rebuild per patch,
        # so compare by identity within one patch round only
        self.assertEqual(self.model.Iris.latent_format.__name__, "RGB")
        self.assertEqual(self.model.Iris.latent_format.latent_channels, 3)

    def test_clip_target_is_the_native_qwen3vl_4b(self):
        guess = self.model.Iris.__new__(self.model.Iris)
        self.assertEqual(guess.clip_target({}), {"qwen3vl_4b.transformer": "text_encoder"})
        self.assertEqual(self.model.Iris.text_encoder_key_prefix, ["text_encoders."])
        self.assertEqual(self.model.Iris.unet_target, "transformer")
        # the vae bucket stays empty on purpose; the loader hook answers it with the stand-in
        self.assertEqual(self.model.Iris.vae_key_prefix, ["vae."])


if __name__ == "__main__":
    unittest.main()
