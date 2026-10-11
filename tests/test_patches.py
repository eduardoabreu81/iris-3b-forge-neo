# Iris 3B support for Forge Neo
# The loader hook's routing: Iris components take the extension paths, everything else reaches the original
# function untouched, and the DiT build swaps Forge's Krea 2 transformer class for ours for one call only.

import sys
import unittest

try:
    import torch
except ImportError:
    torch = None

from unittest.mock import patch

import forge_stubs


def load_patches():
    """Import forge_iris.native.patches and forge_iris.native.model with Forge stubbed."""
    with patch.dict(sys.modules, forge_stubs.patch_modules()):
        import importlib

        patches = importlib.import_module("forge_iris.native.patches")
        model = importlib.import_module("forge_iris.native.model")
    return patches, model


class FakeGuess:
    pass


@unittest.skipIf(torch is None, "torch not installed")
class TestHookRouting(unittest.TestCase):
    def setUp(self):
        self.patches, self.model = load_patches()
        self.patches._applied = False
        self.seen = []
        self.patches.loader.load_huggingface_component = (
            lambda guess, component_name, lib_name, cls_name, repo_path, state_dict:
            self.seen.append(cls_name) or "built")
        self.patches._hook_components()
        self.hook = self.patches.loader.load_huggingface_component
        # the hook checks isinstance against the model module it captured at import time: use that one
        self.guess = self.patches.model.Iris.__new__(self.patches.model.Iris)

    def tearDown(self):
        self.patches._applied = False

    def test_dummy_vae(self):
        result = self.hook(self.guess, "vae", "diffusers", "AutoencoderKLIris", "path", {})
        self.assertIsInstance(result, self.patches._PixelPassthrough)
        self.assertEqual(result.config.latent_channels, 3)
        self.assertEqual(self.seen, [])

    def test_transformer_routes_through_krea_branch(self):
        from forge_iris.native.transformer import IrisDiT

        self.assertEqual(self.hook(self.guess, "transformer", "diffusers", "IrisTransformer2DModel", "path", {}), "built")
        self.assertEqual(self.seen, ["Krea2Transformer2DModel"])
        # the swap lasted one call: Forge's own class is back
        self.assertIsNot(self.patches.krea.SingleStreamDiT, IrisDiT)

    def test_missing_text_encoder_explains_itself(self):
        with self.assertRaises(ValueError) as raised:
            self.hook(self.guess, "text_encoder", "transformers", "Qwen3VLModel", "path", {})
        self.assertIn("qwen3vl_4b", str(raised.exception))
        self.assertEqual(str(raised.exception), self.patches.MISSING_TE)

    def test_other_models_reach_original_untouched(self):
        self.assertEqual(self.hook(FakeGuess(), "transformer", "diffusers", "Krea2Transformer2DModel", "path", {}), "built")
        self.assertEqual(self.seen, ["Krea2Transformer2DModel"])


if __name__ == "__main__":
    unittest.main()
