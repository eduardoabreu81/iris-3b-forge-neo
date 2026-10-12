# Iris 3B support for Forge Neo
# What a checkpoint is, from the published files' real headers: the Iris builds Forge loads (bf16, INT8 ConvRot,
# W4A8, FP32, GGUF), the depth fine-tune, Aikimi's own formats, and files that are not Iris at all.

import io
import tempfile
import unittest

import headers

from forge_iris import checkpoints
from forge_iris.native import detect


class TestFileKind(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def kind(self, fixture, gguf=False):
        write = headers.write_gguf if gguf else headers.write_safetensors
        return checkpoints.file_kind(write(self.directory.name, fixture))

    def test_iris_builds(self):
        for fixture in ("iris-3b_bf16", "iris-3b_int8_convrot", "iris-3b-w4a8_rebel"):
            with self.subTest(fixture=fixture):
                self.assertEqual(self.kind(fixture), checkpoints.IRIS)

    def test_gguf_build(self):
        self.assertEqual(self.kind("iris-3b-q4_k_s", gguf=True), checkpoints.IRIS)

    def test_depth_is_told_apart(self):
        # nynxz tags the folded depth fine-tune in its metadata; its tensors are a plain Iris DiT
        self.assertEqual(self.kind("iris-3b-depth_bf16"), checkpoints.DEPTH)

    def test_aikimi_format_is_foreign(self):
        self.assertEqual(self.kind("aikimi_iris-3b-int8"), checkpoints.FOREIGN)

    def test_other_files(self):
        self.assertIsNone(self.kind("qwen3vl_4b_bf16"))
        self.assertIsNone(checkpoints.file_kind(self.directory.name + "/missing.safetensors"))
        self.assertIsNone(checkpoints.file_kind(self.directory.name + "/model.ckpt"))


class TestDetectOnPackedBuilds(unittest.TestCase):
    """detect() reads rows and norms only: packed 4-bit columns and GGUF shapes give the bf16 configuration."""

    def setUp(self):
        self.reference = detect.detect(headers.header("iris-3b_bf16"), "")

    def test_w4a8(self):
        self.assertEqual(detect.detect(headers.header("iris-3b-w4a8_rebel"), ""), self.reference)

    def test_gguf(self):
        data = (headers.FIXTURES / "iris-3b-q4_k_s_header.gguf.gz").read_bytes()
        import gzip

        shapes, metadata = checkpoints._GGUF(io.BytesIO(gzip.decompress(data))).header()
        self.assertEqual(metadata["general.architecture"], "iris")
        # Forge's GGUF loader hands detect the same original shapes (backend.utils get_orig_shape)
        state_dict = {name: {"shape": list(shape)} for name, shape in shapes.items()}
        self.assertEqual(detect.detect(state_dict, ""), self.reference)


if __name__ == "__main__":
    unittest.main()


class TestFullSizeBuild(unittest.TestCase):
    """IrisDiT(**detect(checkpoint)) at full size on the meta device: every tensor the checkpoint stores must have
    the shape the built module expects (what Forge's load_state_dict needs), for each published build."""

    def build(self, shapes):
        import torch

        from forge_iris.native.transformer import IrisDiT

        with torch.device("meta"):
            return {k: tuple(v.shape) for k, v in IrisDiT(**detect.detect(shapes, "")).state_dict().items()}

    def check(self, shapes, packed=()):
        built = self.build({k: {"shape": list(v)} for k, v in shapes.items()})
        stored = {k: tuple(v) for k, v in shapes.items() if k in built and k not in packed}
        self.assertEqual(set(built) - set(shapes), set(), "parameters the checkpoint lacks")
        mismatched = {k: (stored[k], built[k]) for k in stored if stored[k] != built[k]}
        self.assertEqual(mismatched, {})

    def test_bf16_and_int8(self):
        for fixture in ("iris-3b_bf16", "iris-3b_int8_convrot"):
            with self.subTest(fixture=fixture):
                header = headers.header(fixture)
                header.pop("__metadata__", None)
                self.check({k: v["shape"] for k, v in header.items()})

    def test_w4a8(self):
        # 4-bit weights are stored packed (Forge unpacks them by their comfy_quant orig_shape)
        header = headers.header("iris-3b-w4a8_rebel")
        header.pop("__metadata__", None)
        packed = {k[: -len(".comfy_quant")] + ".weight" for k in header if k.endswith(".comfy_quant")}
        self.check({k: v["shape"] for k, v in header.items()}, packed)

    def test_gguf(self):
        import gzip

        data = gzip.decompress((headers.FIXTURES / "iris-3b-q4_k_s_header.gguf.gz").read_bytes())
        shapes, _ = checkpoints._GGUF(io.BytesIO(data)).header()
        self.check(shapes)
