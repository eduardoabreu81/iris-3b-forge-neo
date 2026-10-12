# Iris 3B support for Forge Neo
# What a checkpoint file is, from its header alone (no weights read): an Iris text-to-image DiT, the Iris depth
# fine-tune, an Iris build in a format Forge cannot load, or something else.
#
# Read before Forge loads the checkpoint (the generation guards) and from the loader's detection hook. safetensors
# and GGUF (city96's layout, as realrebelai/iris-3b_GGUFs) are both read.

import json
import os
import struct
from functools import lru_cache

from .native import detect

IRIS = "iris"
DEPTH = "iris_depth"
FOREIGN = "iris_foreign"

MAX_HEADER_BYTES = 64 * 1024 * 1024
UNET_PREFIXES = ("", "model.diffusion_model.")
# Aikimi Forge Neo's own Iris formats: per-layer qweight/scale(/codebook) instead of weight
FOREIGN_KEYS = ("s_embedder.proj.qweight", "y_pos_embedding", "pixel_embedder.proj.weight")

DEPTH_MESSAGE = ("[Iris 3B] this is the Iris depth model: use it in img2img with the Iris panel set to Depth. For "
                 "images, select iris-3b_bf16 or iris-3b_int8_convrot.")
FOREIGN_MESSAGE = ("[Iris 3B] this Iris build uses Aikimi Forge Neo's own quantization format (iris-rowwise-int8 or "
                   "iris-convrot-w4a8), which only that program loads. Use the bf16, int8_convrot or GGUF builds from "
                   "nynxz/iris-3b or realrebelai/iris-3b_GGUFs instead.")


def kind_of(names, metadata=None, key_prefix="") -> str | None:
    """IRIS, DEPTH, FOREIGN or None for a state dict's (or header's) tensor names."""
    if any(detect.is_iris(names, prefix) for prefix in ({key_prefix} | set(UNET_PREFIXES))):
        return DEPTH if (metadata or {}).get("iris_task") == "depth" else IRIS
    if any(all(f"{prefix}{k}" in names for k in FOREIGN_KEYS) for prefix in ({key_prefix} | set(UNET_PREFIXES))):
        return FOREIGN
    return None


def _safetensors(stream, size):
    length = struct.unpack("<Q", stream.read(8))[0]
    if not 2 <= length <= min(MAX_HEADER_BYTES, size - 8):
        raise ValueError("invalid safetensors header")
    header = json.loads(stream.read(length))
    metadata = header.pop("__metadata__", None) or {}
    return set(header), metadata


class _GGUF:
    SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def __init__(self, stream):
        self.stream = stream

    def read(self, fmt):
        size = struct.calcsize(fmt)
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF header")
        return struct.unpack(fmt, data)[0]

    def string(self):
        length = self.read("<Q")
        if length > MAX_HEADER_BYTES:
            raise ValueError("invalid GGUF string")
        return self.stream.read(length).decode("utf-8", "replace")

    def value(self, kind):
        if kind == 8:
            return self.string()
        if kind == 9:
            item, count = self.read("<I"), self.read("<Q")
            return [self.value(item) for _ in range(count)]
        return self.read("<" + self.SCALARS[kind])

    def header(self):
        if self.stream.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        self.read("<I")  # version
        tensors, entries = self.read("<Q"), self.read("<Q")
        metadata = {}
        for _ in range(entries):
            key = self.string()
            metadata[key] = self.value(self.read("<I"))
        shapes = {}
        for _ in range(tensors):
            name = self.string()
            dims = [self.read("<Q") for _ in range(self.read("<I"))]
            self.stream.read(4 + 8)  # type, offset
            # GGUF stores dims innermost first; ComfyUI-GGUF records the shapes it had to reshape
            shapes[name] = tuple(metadata.get(f"comfy.gguf.orig_shape.{name}") or reversed(dims))
        return shapes, metadata


@lru_cache(maxsize=32)
def _file_kind(path: str, size: int, mtime: int) -> str | None:
    try:
        with open(path, "rb") as stream:
            if path.lower().endswith(".gguf"):
                names, metadata = _GGUF(stream).header()
            else:
                names, metadata = _safetensors(stream, size)
    except (OSError, UnicodeError, ValueError, KeyError, struct.error):
        return None
    return kind_of(names, metadata)


def file_kind(path) -> str | None:
    """kind_of for a checkpoint file, read from its header and cached by size and modification time."""
    if not str(path).lower().endswith((".safetensors", ".gguf")):
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return _file_kind(os.path.abspath(path), stat.st_size, stat.st_mtime_ns)
