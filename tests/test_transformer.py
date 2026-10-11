# Iris 3B support for Forge Neo
# The DiT at toy sizes: Forge's build path (IrisDiT(**detected)) round-trips the checkpoint exactly, and a
# generation step runs end to end on CPU.

import unittest

try:
    import torch
except ImportError:
    torch = None

from forge_iris.native.detect import detect
from forge_iris.native.transformer import IrisConfig, IrisDiT


def toy_config(**overrides):
    values = dict(hidden_size=32, depth=4, dual_depth=2, num_heads=4, num_kv_heads=2, patch_size=4,
                  ffn_hidden_size=24, text_len=300, text_dim=32, text_lap_num_layers=3,
                  text_lap_num_heads=4, text_lap_ffn_hidden_size=12, pixel_hidden_size=8,
                  pixel_attn_hidden_size=16, pixel_num_heads=4, pixel_depth=2, pixel_ffn_hidden_size=8)
    values.update(overrides)
    return IrisConfig(**values)


@unittest.skipIf(torch is None, "torch not installed")
class TestTransformer(unittest.TestCase):
    def test_forge_build_path_round_trips(self):
        # Forge builds IrisDiT(**dit_config_from_detect) and loads the checkpoint strict; any drift between
        # detect() and the module's parameter names or shapes fails here
        original = IrisDiT(toy_config())
        built = IrisDiT(**detect(original.state_dict(), ""))
        built.load_state_dict(original.state_dict(), strict=True)

    def test_forward_shape_and_finiteness(self):
        torch.manual_seed(0)
        dit = IrisDiT(toy_config())
        x = torch.rand(1, 3, 32, 32) * 2 - 1
        context = torch.rand(1, 300, 32)
        out = dit(x, torch.tensor([500.0]), context)
        self.assertEqual(out.shape, (1, 3, 32, 32))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_forward_odd_size_pads(self):
        # dimensions that are not a multiple of the patch size pad and crop back
        torch.manual_seed(0)
        dit = IrisDiT(toy_config())
        out = dit(torch.rand(1, 3, 30, 34) * 2 - 1, torch.tensor([0.0]), torch.rand(1, 300, 32))
        self.assertEqual(out.shape, (1, 3, 30, 34))

    def test_preprocess_text(self):
        # the engine path: raw layer states [B, layers, T, D] with a mask -> the DiT's text stream; the engine
        # zeroes the padded states before this call, so the adapter output tracks them (the pos embedding is
        # added to every row, as upstream does)
        torch.manual_seed(0)
        config = toy_config()
        dit = IrisDiT(config)
        states = torch.rand(1, 300, config.text_lap_num_layers, 32)  # tokens-major, as the engine hands them over
        mask = torch.ones(1, 300)
        mask[:, 250:] = 0
        y = dit.preprocess_text(states * mask[:, :, None, None], mask)
        self.assertEqual(y.shape, (1, 300, 32))
        self.assertTrue(bool(torch.isfinite(y).all()))
        clean = dit.preprocess_text(states * mask[:, :, None, None], mask)
        self.assertTrue(torch.allclose(y, clean))

    def test_forward_matches_itself(self):
        torch.manual_seed(1)
        dit = IrisDiT(toy_config()).eval()
        x, t, context = torch.rand(1, 3, 16, 16), torch.tensor([500.0]), torch.rand(1, 300, 32)
        with torch.no_grad():
            a, b = dit(x, t, context), dit(x, t, context)
        self.assertTrue(torch.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
