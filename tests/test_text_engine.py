# Iris 3B support for Forge Neo
# The text engine's token assembly: template + 300-token window + pads, and the encoding path through the
# stand-in SDClipModel to one prompt's raw conditioning (no batch dimension: Forge stacks the batch).

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
    engine = CONSTANTS["IrisTextEngine"].__new__(CONSTANTS["IrisTextEngine"])
    engine.tokenizer = forge_stubs.FakeSDTokenizer(FakeHFTokenizer())
    engine.text_encoder = forge_stubs.FakeSDClipModel()
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

    def test_whitespace_is_the_empty_prompt(self):
        # "", a space, and what Forge leaves of a prompt that only held LoRA tags all encode as the empty prompt
        empty = self.engine.token_ids("")
        for text in (" ", "  ", "\n", " \t "):
            with self.subTest(text=repr(text)):
                self.assertEqual(self.engine.token_ids(text), empty)
                self.assertEqual(self.engine.tokenize(text), 0)
        self.assertEqual(self.engine.token_ids(" a bird "), self.engine.token_ids("a bird"))

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

    def test_encode_window(self):
        # the stand-in model returns layers-major states (1, 12, T, 24) marked with T and a mask ending in pads;
        # the engine must transpose, keep the last TEXT_LENGTH tokens and drop the batch dimension
        states = self.engine.encode("a bird")
        total = len(self.engine.prefix_ids) + TEXT_LENGTH
        self.assertEqual(states.shape, (TEXT_LENGTH, 12, 24))
        self.assertEqual(states[0, 0, 0].item(), float(total))

    def test_encode_zeroes_pads(self):
        # every layer of a padded token is zero, and only those: the DiT reads the mask back from them
        states = self.engine.encode("a bird")
        used = len(self.engine.tokenizer.tokenizer.encode("a bird", add_special_tokens=False)) + len(self.engine.suffix_ids)
        padded = states.flatten(1).eq(0).all(-1)
        self.assertFalse(bool(padded[:used].any()))
        self.assertTrue(bool(padded[used:].all()))

    def test_batch_stacks_like_forge(self):
        # Forge's prompt_parser stacks one tensor per prompt into the batch
        zs = self.engine(["a bird", "a cat"])
        self.assertEqual(torch.stack(zs).shape, (2, TEXT_LENGTH, 12, 24))

    def test_call_caches_per_prompt(self):
        engine = self.engine
        engine.encode = lambda line: torch.tensor([len(line)])
        zs = engine(["same prompt", "same prompt", "other"])
        self.assertEqual([tensor.item() for tensor in zs], [11, 11, 5])


if __name__ == "__main__":
    unittest.main()
