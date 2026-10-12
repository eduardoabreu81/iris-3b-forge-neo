# Iris 3B support for Forge Neo
# kernels.attention on Forge's path: the function is looked up on backend.attention, and GQA reaches backends that
# only accept equal head counts (Forge's pytorch/basic backends without enable_gqa, its flash/sage fallbacks).

import importlib
import sys
import types
import unittest
from unittest.mock import patch

import torch


def strict_attention(q, k, v, heads, mask=None, skip_reshape=False, transformer_options=None, **kwargs):
    """Forge's attention_pytorch as called without enable_gqa: SDPA refuses mismatched head counts."""
    assert skip_reshape and q.shape[1] == heads
    if k.shape[1] != q.shape[1]:
        raise RuntimeError("head count mismatch")
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def load_kernels(attention_function):
    attention = types.ModuleType("backend.attention")
    attention.attention_function = attention_function
    memory = types.ModuleType("backend.memory_management")
    memory.cast_to = lambda weight, dtype=None, device=None: weight.to(dtype=dtype, device=device)
    operations = types.ModuleType("backend.operations")
    operations.main_stream_worker = operations.weights_manual_cast = None
    backend = types.ModuleType("backend")
    backend.attention, backend.memory_management, backend.operations = attention, memory, operations
    stubs = {"backend": backend, "backend.attention": attention, "backend.memory_management": memory,
             "backend.operations": operations}
    package = importlib.import_module("forge_iris.native")
    original = getattr(package, "kernels", None)
    # patch.dict puts the real kernels module back on exit; the package attribute is restored by hand
    with patch.dict(sys.modules, stubs):
        sys.modules.pop("forge_iris.native.kernels", None)
        kernels = importlib.import_module("forge_iris.native.kernels")
    if original is not None:
        package.kernels = original
    return kernels


class TestForgeAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.q = torch.randn(2, 8, 10, 16)
        self.k = torch.randn(2, 2, 10, 16)
        self.v = torch.randn(2, 2, 10, 16)

    def reference(self):
        out = torch.nn.functional.scaled_dot_product_attention(self.q, self.k, self.v, enable_gqa=True)
        return out.transpose(1, 2).reshape(2, 10, -1)

    def test_gqa_matches_sdpa(self):
        kernels = load_kernels(strict_attention)
        out = kernels.attention(self.q, self.k, self.v)
        torch.testing.assert_close(out, self.reference())

    def test_function_is_looked_up_per_call(self):
        calls = []
        kernels = load_kernels(strict_attention)
        kernels._forge_attention.attention_function = lambda *a, **k: calls.append(k) or strict_attention(*a, **k)
        kernels.attention(self.q, self.k, self.v, transformer_options={"x": 1})
        self.assertEqual(calls[0]["transformer_options"], {"x": 1})


if __name__ == "__main__":
    unittest.main()
