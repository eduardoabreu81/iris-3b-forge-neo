# Iris 3B support for Forge Neo
# Iris LoRAs on Forge's own LoRA path: the community formats are converted before Forge matches the keys.
#
# - The Turbo LoRA (pentacoxian-dev/iris-3b-turbo-lora, from the pentaCoxian/iris-3b-turbo fork) stores bare
#   "<layer>.lora_down" / "<layer>.lora_up" with alpha and rank in the file's metadata; Forge expects
#   "diffusion_model.<layer>.lora_down.weight" and an alpha tensor.
# - The trunk SwiGLU width differs between builds (6826 in bf16/FP32, zero-padded to 6912 in INT8 ConvRot; the text
#   refiner's stays 6826), and a LoRA carries the width it was extracted at. The padded units are always zero, so a
#   LoRA is fitted to the loaded DiT exactly: its w1/w3 up rows and w2 down columns are zero-padded or cropped to the
#   width of the model's own layer.
#
# The hook wraps sd_forge_lora's networks.load_lora_for_models (Forge's built-in LoRA extension) and only touches
# LoRAs applied to an Iris DiT; everything else reaches Forge untouched.

import json
import re
import struct
import sys

import torch

_FFN = re.compile(r"(?:^|\.)mlp(?:_[xy])?\.(w1|w2|w3)\.lora_(up|down)\.weight$")
_hooked = False


def _turbo_metadata(filename):
    """alpha and rank from a safetensors header's metadata, or None."""
    try:
        with open(filename, "rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 64 * 1024 * 1024:
                return None
            metadata = json.loads(stream.read(length)).get("__metadata__") or {}
        return float(metadata["alpha"]), int(metadata["rank"])
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return None


def is_bare_format(lora: dict) -> bool:
    return any(k.endswith(".lora_down") for k in lora) and not any(k.endswith(".lora_down.weight") for k in lora)


def convert(lora: dict, ffn_width: int, alpha_rank=None, refiner_width: int = 6826) -> dict:
    """The LoRA as Forge matches it for an Iris DiT whose trunk FFN is ffn_width wide (refiner_width in the text
    refiner)."""
    if is_bare_format(lora):
        converted = {}
        for key, value in lora.items():
            layer, _, kind = key.rpartition(".")
            if kind not in ("lora_down", "lora_up"):
                converted[key] = value
                continue
            converted[f"diffusion_model.{layer}.{kind}.weight"] = value
            if kind == "lora_down" and alpha_rank is not None:
                # Forge scales by alpha / rank, where rank is the down matrix's row count
                converted[f"diffusion_model.{layer}.alpha"] = torch.tensor(alpha_rank[0] / alpha_rank[1] * value.shape[0])
        lora = converted
    out = {}
    for key, value in lora.items():
        match = _FFN.search(key)
        if match is not None and value.ndim == 2:
            weight, kind = match.groups()
            width = refiner_width if "y_embedder.refiner." in key else ffn_width
            # w1/w3 up: rows are the FFN units; w2 down: columns are; the other two face the hidden size
            axis = 0 if weight != "w2" and kind == "up" else (1 if weight == "w2" and kind == "down" else None)
            if axis is not None and value.shape[axis] != width:
                if value.shape[axis] > width:
                    value = value.narrow(axis, 0, width)
                else:
                    pad = width - value.shape[axis]
                    value = torch.nn.functional.pad(value, (0, pad) if axis == 1 else (0, 0, 0, pad))
        out[key] = value
    return out


def _iris_dit(model):
    from .transformer import IrisDiT

    dit = getattr(getattr(model, "model", None), "diffusion_model", None)
    return dit if isinstance(dit, IrisDiT) else None


def install() -> bool:
    """Wrap networks.load_lora_for_models once sd_forge_lora is imported; True when the hook is in place."""
    global _hooked
    if _hooked:
        return True
    networks = sys.modules.get("networks")
    original = getattr(networks, "load_lora_for_models", None)
    if original is None:
        return False

    def load_lora_for_models(model, clip, lora, *args, **kwargs):
        dit = _iris_dit(model)
        if dit is not None:
            filename = kwargs.get("filename", args[2] if len(args) > 2 else None)
            alpha_rank = _turbo_metadata(filename) if filename and is_bare_format(lora) else None
            lora = convert(lora, dit.cfg.ffn_hidden_size, alpha_rank, dit.cfg.refiner_ffn_hidden_size)
        return original(model, clip, lora, *args, **kwargs)

    networks.load_lora_for_models = load_lora_for_models
    _hooked = True
    return True
