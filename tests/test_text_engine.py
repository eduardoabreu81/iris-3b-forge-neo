# Iris 3B support for Forge Neo
# The text engine's token assembly: template + 300-token window + pads, and the encoding path through the
# stand-in SDClipModel into the DiT's preprocess_text.

import unittest

try:
    import torch
except ImportError:
    torch = None

from unittest.mock import patch

import forge_stubs
from forge_stubs import FakeHFTokenizer

CONSTANTS = {}


def _load_module():
    with patch.dict("sys.modules", forge_stubs.text_processing()):
        import importlib

        module = importlib.import_module("forge_iris.native.text_engine")
    CONSTANTS.update(PAD=module.PAD, PREFIX=module.PREFIX, SUFFIX=module.SUFFIX,
                     TEXT_LENGTH=module.TEXT_LENGTH, IrisTextEngine=module.IrisTextEngine)
    return module


_load_module()
PAD, PREFIX, SUFFIX, TEXT_LENGTH = CONSTANTS["PAD"], CONSTANTS["PREFIX"], CONSTANTS["SUFFIX"], CONSTANTS["TEXT_LENGTH"]


def make_engine():
    import torch

    class Dit:
        # the engine reads the adapter's device from this parameter
        y_pos_embedding = torch.nn.Parameter(torch.zeros(1, 300, 8))

        def preprocess_text(self, states, mask):
            # the real adapter stays with the DiT; here the engine's layout contract is what matters:
            # states (B, tokens, layers, dim), mask (B, tokens)
            return states, mask

    engine = CONSTANTS["IrisTextEngine"].__new__(CONSTANTS["IrisTextEngine"])
    engine.tokenizer = forge_stubs.FakeSDTokenizer(FakeHFTokenizer())
    engine.text_encoder = forge_stubs.FakeSDClipModel()
    engine.dit = Dit()
    engine.prefix_ids = engine.tokenizer.tokenizer.encode(PREFIX, add_special_tokens=False)
    engine.suffix_ids = engine.tokenizer.tokenizer.encode(SUFFIX, add_special_tokens=False)
    return engine


class TestTokenIds(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_window_size(self):
        # system + TEXT_LENGTH tokens in total, whatever the text length
        for text in ("", "a bird takes off", "word " * 400):
            with self.subTest(text=text[:20]):
                ids = self.engine.token_ids(text)
                self.assertEqual(len(ids), len(self.engine.prefix_ids) + TEXT_LENGTH)

    def test_template_and_padding(self):
        ids = self.engine.token_ids("a bird")
        text = self.engine.tokenizer.tokenizer.encode("a bird", add_special_tokens=False)
        self.assertEqual(ids[:len(self.engine.prefix_ids)], self.engine.prefix_ids)
        self.assertEqual(ids[len(self.engine.prefix_ids):len(self.engine.prefix_ids) + 6], text)
        # the suffix follows the text, pads fill the tail of the system + TEXT_LENGTH window
        start = len(self.engine.prefix_ids) + 6
        self.assertEqual(ids[start:start + len(self.engine.suffix_ids)], self.engine.suffix_ids)
        self.assertNotIn(PAD, ids[:start + len(self.engine.suffix_ids)])
        self.assertTrue(all(token == PAD for token in ids[start + len(self.engine.suffix_ids):]))

    def test_long_text_truncates(self):
        ids = self.engine.token_ids("word " * 400)
        budget = TEXT_LENGTH - len(self.engine.suffix_ids)
        text = self.engine.tokenizer.tokenizer.encode("word " * 400, add_special_tokens=False)
        kept = ids[len(self.engine.prefix_ids):len(self.engine.prefix_ids) + budget]
        self.assertEqual(kept, text[:budget])


@unittest.skipIf(torch is None, "torch not installed")
class TestEncode(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()

    def test_encode_window_and_mask(self):
        # the stand-in model returns layers-major states (1, 12, T, 24) marked with T and a mask ending in pads;
        # the engine must transpose, keep the last TEXT_LENGTH tokens, zero the pads and hand the DiT both
        states, mask = self.engine.encode("a bird")
        total = len(self.engine.prefix_ids) + TEXT_LENGTH
        self.assertEqual(states.shape, (1, TEXT_LENGTH, 12, 24))
        self.assertEqual(states[0, 0, 0, 0].item(), float(total))
        self.assertEqual(mask.shape, (1, TEXT_LENGTH))
        self.assertEqual(mask[0, 0].item(), 1.0)
        self.assertEqual(mask[0, -1].item(), 0.0)

    def test_encode_zeroes_pads(self):
        states, _ = self.engine.encode("a bird")
        # every layer of a padded token is zeroed by the mask multiply
        self.assertTrue(bool((states[:, -1] == 0).all()))

    def test_call_caches_per_prompt(self):
        engine = self.engine
        engine.encode = lambda line: torch.tensor([len(line)])
        zs = engine(["same prompt", "same prompt", "other"])
        self.assertEqual([tensor.item() for tensor in zs], [11, 11, 5])


if __name__ == "__main__":
    unittest.main()
