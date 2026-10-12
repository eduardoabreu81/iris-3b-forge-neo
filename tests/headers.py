"""Checkpoint files holding only a real header (fixtures/*_header.*.gz): enough for anything that reads headers."""

import gzip
import json
import struct
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def header(fixture):
    return json.load(gzip.open(FIXTURES / f"{fixture}_header.json.gz", "rt"))


def write_safetensors(directory, fixture):
    data = json.dumps(header(fixture)).encode()
    path = Path(directory) / f"{fixture}.safetensors"
    path.write_bytes(struct.pack("<Q", len(data)) + data)
    return path


def write_gguf(directory, fixture):
    path = Path(directory) / f"{fixture}.gguf"
    path.write_bytes(gzip.open(FIXTURES / f"{fixture}_header.gguf.gz", "rb").read())
    return path
