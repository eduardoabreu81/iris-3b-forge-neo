# Iris 3B support for Forge Neo
# The loader hook's routing: Iris components take the extension paths, everything else reaches the original
# function untouched, and the DiT build swaps Forge's Krea 2 transformer class for ours for one call only.

import gzip
import json
import sys
import unittest
from pathlib import Path

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


class Shape:
    def __init__(self, *shape):
        self.shape = shape


def text_encoder_state_dict():
    header = json.load(gzip.open(Path(__file__).parent / "fixtures" / "qwen3vl_4b_bf16_header.json.gz", "rt"))
    return {k: Shape(*v["shape"]) for k, v in header.items() if k != "__metadata__"}


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

    def test_real_text_encoder_is_accepted(self):
        # Comfy-Org's qwen3vl_4b_bf16 keys, as the loader hands them over (before Forge's own renaming)
        state_dict = text_encoder_state_dict()
        self.assertEqual(self.hook(self.guess, "text_encoder", "transformers", "Qwen3VLModel", "path", state_dict), "built")
        self.assertEqual(self.seen, ["Qwen3VLModel"])

    def test_flat_text_encoder_keys_are_accepted(self):
        state_dict = {k.replace("model.language_model.", "model."): v for k, v in text_encoder_state_dict().items()}
        self.assertEqual(self.hook(self.guess, "text_encoder", "transformers", "Qwen3VLModel", "path", state_dict), "built")

    def test_wrong_size_text_encoder_explains_itself(self):
        state_dict = text_encoder_state_dict()
        state_dict["model.language_model.layers.0.input_layernorm.weight"] = Shape(4096)
        with self.assertRaises(ValueError) as raised:
            self.hook(self.guess, "text_encoder", "transformers", "Qwen3VLModel", "path", state_dict)
        self.assertEqual(str(raised.exception), self.patches.WRONG_TE)

    def test_foreign_iris_format_explains_itself(self):
        self.patches._hook_detection()
        state_dict = text_encoder_state_dict()  # anything that is not Iris...
        state_dict.update({k: Shape(1) for k in ("s_embedder.proj.qweight", "y_pos_embedding",
                                                   "pixel_embedder.proj.weight")})  # ...plus Aikimi's layout
        with self.assertRaises(ValueError) as raised:
            self.patches.detection.detect_unet_config(state_dict, "")
        self.assertIn("Aikimi", str(raised.exception))

    def test_other_models_reach_original_untouched(self):
        self.assertEqual(self.hook(FakeGuess(), "transformer", "diffusers", "Krea2Transformer2DModel", "path", {}), "built")
        self.assertEqual(self.seen, ["Krea2Transformer2DModel"])


if __name__ == "__main__":
    unittest.main()
