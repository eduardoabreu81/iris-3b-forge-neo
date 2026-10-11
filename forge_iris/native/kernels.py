# Iris 3B support for Forge Neo
# The few Forge Neo runtime pieces the DiT calls, with plain torch fallbacks outside Forge (CPU tests).

import contextlib

import torch

try:
    from backend import attention as _forge_attention
    from backend.memory_management import cast_to as _forge_cast_to
    from backend.operations import main_stream_worker, weights_manual_cast
except ImportError:  # outside Forge Neo: eager backend on CPU
    _forge_attention = None
    _forge_cast_to = None
    main_stream_worker = None
    weights_manual_cast = None

__all__ = ["attention", "cast_to", "weight_and_bias"]


def cast_to(weight, dtype=None, device=None):
    if _forge_cast_to is not None:
        return _forge_cast_to(weight, dtype=dtype, device=device)
    return weight.to(dtype=dtype, device=device)


def attention(q, k, v, mask=None, transformer_options=None):
    """q: [B, Hq, S, D]; k, v: [B, Hkv, S, D] (GQA) -> [B, S, Hq * D].

    transformer_options reach Forge's attention wrapper, which applies attention overrides such as the Sparse
    Attention Integrated extension. With no mask, skip_reshape lets Forge pick its fastest kernel.
    """
    if _forge_attention is not None:
        return _forge_attention(q, k, v, q.shape[1], mask=mask, skip_reshape=True,
                                transformer_options=transformer_options or {})
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=q.shape[1] != k.shape[1])
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


@contextlib.contextmanager
def weight_and_bias(layer, ref):
    """A linear layer's weight and bias on ref's device and dtype, LoRA patches applied, as Forge's Linear does."""
    if weights_manual_cast is not None and hasattr(layer, "weight_function"):
        weight, bias, signal = weights_manual_cast(layer, ref)
        with main_stream_worker(weight, bias, signal):
            yield weight, bias
        return
    yield layer.weight.to(ref), layer.bias.to(ref)
